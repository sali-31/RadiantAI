import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
from .product_data_cleaner import ProductDataCleaner
import re
import re


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProductRecommender:
    """
    Match skin analysis to product recommendations
    
    Functions: 
        - recommend_for_condition: Get product recommendations for a specific condition
        - recommend_for_analysis: Extracts condition and severity from Gemini analysis and recommends products.
    """
    # Weights when customer prioritizes ratings
    RATING_PRIORITY_WEIGHTS = {
        'rating': 0.6,
        'reviews': 0.3,
        'price': 0.1
    }

    # Weights when customer prioritizes price
    PRICE_PRIORITY_WEIGHTS = {
        'rating': 0.4,
        'reviews': 0.2,
        'price': 0.4
    }
    
    
    def __init__(self):
        self.cleaner = ProductDataCleaner()
        self.products = self.cleaner.clean_all_products(save_cleaned=False)
        logger.info("Product data cleaned and loaded")


    def create_product_bundle(
        self,
        condition: str,
        budget_max: float = None,
        categories: List[str] = None,
        prioritize_rating: bool = True,
        keywords: List[str] = None
    ):
        """
        Create a product bundle for a specific condition.

        Args:
            - condition: Skin condition to match (e.g. "acne")
            - budget_max: Maximum budget to filter price (default: None)
            - categories: List of product categories to filter
            - prioritize_rating: Weight rating higher than price (default: True)
            - keywords: List of keywords to filter products (optional)

        Returns:
            Dict with bundle products, total cost and savings
        """
        # Configure the weights depending on customer preference (price vs ratings)
        weights = self.RATING_PRIORITY_WEIGHTS if prioritize_rating else self.PRICE_PRIORITY_WEIGHTS

        if condition not in self.products:
            return {"error": "Condition not found", "bundle": []}
        
        df = self.products[condition].copy()

        if df.empty:
            return {f"error": "No products found for condition: {condition}", "bundle": []}
        
        # Filter by keywords if provided
        if keywords:
            import re
            # Create a regex pattern for case-insensitive matching of ANY keyword
            escaped_keywords = [re.escape(k) for k in keywords]
            pattern = '|'.join(escaped_keywords)
            df = df[df['title'].str.contains(pattern, case=False, na=False)]
            
            if df.empty:
                 # If keywords filter everything out, return empty or maybe fallback? 
                 # For now, return empty to respect the filter.
                 return {f"error": f"No products found matching keywords: {keywords}", "bundle": []}

        # Categorize products by keywords in title
        df = self._categorize_products(df)

        # Default categories for skincare routine
        if categories is None:
            categories = ['cleanser', 'treatment', 'moisturizer']

        # Set up your knapsack and remaining space
        bundle = []
        remaining_budget = budget_max if budget_max is not None else float('inf')

        # Filter products based on category
        for category in categories:
            category_products = df[df['category'] == category]

            # Move to the next category if the product list is empty
            if category_products.empty:
                continue

            # Filter by remaining budget (use .copy() to avoid SettingWithCopyWarning)
            affordable = category_products[category_products['price_numeric'] <= remaining_budget].copy()

            # Move to the next category if there are no affordable items of that category
            # that can fit within the budget
            if affordable.empty:
                continue

            """ Calculate value score:
            Value scores are adjusted weights of the three most important columns depending on whether we should prioritize rating or not.
            """
            
            # Normalize rating to 0-1 scale so it's comparable to the normalized reviews score
            normalized_rating = affordable['rating'] / 5.0
            
            max_reviews = affordable['reviews'].max()
            reviews_score = (affordable['reviews'] / max_reviews) * weights['reviews'] if max_reviews > 0 else 0

            # If prioritizing rating, we treat price only as a constraint (must fit in budget),
            # not as a ranking factor. This allows higher-priced (but high-rated) items to be selected
            # if the budget allows.
            if prioritize_rating:
                price_score = 0
            else:
                max_price = affordable['price_numeric'].max()
                price_score = (affordable['price_numeric'] / max_price) * weights['price'] if max_price > 0 else 0

            affordable['value_score'] = (
                normalized_rating * weights['rating'] + reviews_score - price_score)
                
            

            # Select the best value product
            # OLD: best_product_idx = affordable['value_score'].idxmax()
            # NEW: Randomize among top 3 value scores to add variety
            top_candidates = affordable.nlargest(3, 'value_score')
            
            # Randomly sample 1 from the top candidates
            best_product = top_candidates.sample(n=1).iloc[0].to_dict()


            bundle.append({
                'category': category,
                'name': best_product['title'],
                'asin': best_product['asin'],
                'price': best_product['price'],
                'price_numeric': best_product['price_numeric'],
                'rating': best_product['rating'],
                'reviews': best_product['reviews'],
                'link': best_product['link'],
                'thumbnail': best_product['thumbnail'],
                'directions': best_product.get('directions', 'See product packaging for directions'),
                'value_score': best_product['value_score']
            })

            remaining_budget -= best_product['price_numeric']

        total_cost = sum(item['price_numeric'] for item in bundle)

        return {
            'bundle': bundle,
            'total_cost': total_cost,
            'budget_max': budget_max,
            'budget_utilized_pct': round((total_cost / budget_max) * 100, 2) if budget_max else 0,
            'categories_included': [item['category'] for item in bundle]
        }

    def _categorize_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize products by keywords in title.
        """
        category_keywords = {
            'cleanser': ['cleanser', 'face wash', 'wash', 'cleansing'],
            'treatment': ['treatment', 'serum', 'gel', 'spot treatment', 'adapalene', 'benzoyl', 'retinol'],
            'moisturizer': ['moisturizer', 'cream', 'lotion', 'hydrating'],
            'toner': ['toner', 'essence'],
            'sunscreen': ['sunscreen', 'spf', 'sun protection'],
            'mask': ['mask', 'peel'],
            'patch': ['patch', 'sticker']
        }

        def get_category(title):
            if pd.isna(title):
                return 'other'

            title_lower = title.lower()

            for category, keywords in category_keywords.items():
                if any(keyword in title_lower for keyword in keywords):
                    return category

            return 'other'

        df['category'] = df['title'].apply(get_category)
        return df

    def recommend_for_condition(
        self,
        condition: str,
        severity: str = 'moderate',
        budget_max: float = None,
        top_n: int = 5,
        keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        This function will get product recommendations for a specific condition
        based on facial analysis data.

        Args:
            - condition: Skin condition to match (e.g. "acne")
            - severity: Skin condition severity (expected: "mild"/"moderate"/"severe")
            - budget_max: Maximum budget to filter price
            - top_n: Number of products to return
            - keywords: List of keywords to filter products (optional)

        Returns:
            List of recommended products
        """

        if condition not in self.products:
            logger.warning(f"No products found for condition: {condition}")
            return []

        df = self.products[condition].copy()

        if df.empty:
            return []

        # Apply filters
        if budget_max is not None:
            df = df[df['price_numeric'] <= budget_max]

        # Apply keyword filter
        if keywords:
            import re
            # Create a regex pattern for case-insensitive matching of ANY keyword
            escaped_keywords = [re.escape(k) for k in keywords]
            pattern = '|'.join(escaped_keywords)
            df = df[df['title'].str.contains(pattern, case=False, na=False)]

        # Sort by rating (primary) and reviews (secondary)
        df = df.sort_values(['rating','reviews'], ascending=[False, False])

        # Randomization Logic:
        # Instead of just taking the top N, we take a larger pool (e.g. top 3*N)
        # and randomly sample from them. This ensures quality (high ratings)
        # but provides variety ("freshness") on subsequent calls.
        
        candidate_pool_size = top_n * 3
        candidates = df.head(candidate_pool_size)
        
        if candidates.empty:
            return []

        # Randomly sample from the top candidates
        # We use min() to handle cases where we have fewer candidates than top_n
        recommendations_df = candidates.sample(n=min(top_n, len(candidates)))

        # Convert to list of dicts
        recommendations = recommendations_df.to_dict('records')

        return recommendations

    def extract_conditions_and_severities(self, gemini_analysis: str) -> Dict[str, str]:
        """
        Extracts condition and severity from Gemini JSON analysis.

        Now handles structured JSON output from Gemini 2.0 Flash.

        Args:
            gemini_analysis: JSON string from Gemini API

        Returns:
            Dict of {condition: severity} or empty dict if not found
        """
        try:
            # Parse JSON response
            analysis_data = json.loads(gemini_analysis)
            detected_conditions = {}

            condition_keywords = {
                'acne': ['acne', 'pimple', 'blemish', 'breakout', 'comedone', 'whitehead', 'blackhead', 'papule', 'pustule'],
                'rosacea': ['rosacea', 'facial redness', 'flushing', 'telangiectasia'],
                'eczema': ['eczema', 'atopic dermatitis', 'dry patches', 'dermatitis'],
                'dry_skin': ['dry skin', 'dehydrated', 'flaky', 'xerosis'],
                'oily_skin': ['oily', 'sebum', 'shine', 'sebaceous'],
                'hyperpigmentation': ['hyperpigmentation', 'dark spots', 'discoloration', 'melasma', 'post-inflammatory'],
                'melasma': ['melasma', 'brown patches', 'chloasma']
            }

            # 1. Try New Schema (List of conditions)
            if 'detected_conditions' in analysis_data and isinstance(analysis_data['detected_conditions'], list):
                for item in analysis_data['detected_conditions']:
                    raw_condition = item.get('condition', '').lower()
                    severity = item.get('severity', 'moderate').lower()
                    
                    # Map raw condition string to our internal keys
                    for key, keywords in condition_keywords.items():
                        if any(kw in raw_condition for kw in keywords):
                            detected_conditions[key] = severity
                            logger.info(f"Detected condition (structured): {key} ({severity})")
                            break
            
            # 2. Fallback/Legacy Schema (Global severity + Characterization text)
            if not detected_conditions:
                # Extract severity directly from JSON (already normalized)
                severity = analysis_data.get('severity', '').lower()
                if severity not in ['mild', 'moderate', 'severe']:
                    # logger.warning(f"Invalid severity '{severity}', defaulting to moderate")
                    severity = 'moderate'

                # Extract characterization and search for condition
                characterization = analysis_data.get('characterization', '').lower()

                # Search characterization for condition keywords
                for condition, keywords in condition_keywords.items():
                    if any(keyword in characterization for keyword in keywords):
                        detected_conditions[condition] = severity
                        logger.info(f"Detected condition: {condition} (severity: {severity})")

            if not detected_conditions:
                # logger.warning(f"No condition detected in: {characterization[:100]}")
                return {}

            return detected_conditions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}")
            logger.error(f"Raw response: {gemini_analysis[:200]}")

            # Fallback: try keyword search on raw text (old method)
            # Note: Fallback still returns tuple, we wrap it in dict
            cond, sev = self._extract_condition_from_text_fallback(gemini_analysis)
            if cond:
                return {cond: sev}
            return {}

        except Exception as e:
            logger.error(f"Unexpected error extracting condition: {e}")
            return {}

    def _extract_condition_from_text_fallback(self, text: str) -> Tuple[Optional[str], str]:
        """
        Fallback method for parsing unstructured text responses.
        Used when JSON parsing fails.
        """
        logger.warning("Using fallback text parsing (JSON parse failed)")

        text_lower = text.lower()

        condition_keywords = {
            'acne': ['acne', 'pimple', 'blemish', 'breakout', 'comedone', 'whitehead', 'blackhead'],
            'rosacea': ['rosacea', 'facial redness', 'flushing'],
            'eczema': ['eczema', 'atopic dermatitis', 'dry patches'],
            'dry_skin': ['dry skin', 'dehydrated', 'flaky'],
            'oily_skin': ['oily', 'sebum', 'shine'],
            'hyperpigmentation': ['hyperpigmentation', 'dark spots', 'discoloration'],
            'melasma': ['melasma', 'brown patches']
        }

        severity_keywords = {
            'mild': ['mild', 'light', 'minor'],
            'moderate': ['moderate', 'medium'],
            'severe': ['severe', 'serious', 'heavy', 'extensive']
        }

        # Find condition
        detected_condition = None
        for condition, keywords in condition_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_condition = condition
                break

        # Find severity
        detected_severity = 'moderate'  # default
        for severity_level, keywords in severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_severity = severity_level
                break

        if detected_condition:
            logger.info(f"Fallback detected: {detected_condition} ({detected_severity})")

        return detected_condition, detected_severity
    
    def get_combined_products(self, conditions: List[str], budget_max: float = None) -> pd.DataFrame:
        """
        Function to gather available products for the list of conditions 
        extracted from the gemini analysis
        
        @params:
            - self,
            - conditions: A list of strings containing the user skin conditions
        """
        dfs = []
        for condition in conditions:
            if condition in self.products:
                dfs.append(self.products[condition])

        if not dfs:
            return pd.DataFrame()
        
        # 1. Concatenate all matching dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # 2. Deduplicate by ASIN (prevents adding the same product twice)
        combined_df = combined_df.drop_duplicates(subset='asin')

        # 3. Apply Budget Filter
        if budget_max:
            combined_df = combined_df[combined_df['price_numeric'] <= budget_max]

        return combined_df

    def get_randomized_top_n(self, df: pd.DataFrame, sort_col: str, n: int = 5, pool_size: int = 15) -> List[Dict[str, Any]]:
        """
        This Function offers a Freshness factor to the returned list of products by using stratified sampling.
        Instead of taking just the top 5 products, we'll grab the top tier of 15 products and randomly sample
        that tier.
        1. Sort by metric (Rating, Price, Reviews, etc)
        2. We'll take the top 15 Products within that metric
        3. Randomly pick 5 from those 15
        """
        if df.empty:
            return []
        
        # Sort the list
        ascending = True if sort_col == 'price_numeric' else False
        sorted_df = df.sort_values(sort_col, ascending=ascending)

        # Pool the list
        candidate_pool = sorted_df.head(pool_size)

        # Randomly sample from the top candidates (*Use min to handle edge case where pool < n)
        sample_size = min(n, len(candidate_pool))
        return candidate_pool.sample(n=sample_size).to_dict('records')


    def create_product_bundle_from_analysis(
        self,
        gemini_analysis: str,
        budget_max: float = None,
        categories: List[str] = None,
        prioritize_rating: bool = True,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create a product bundle based on Gemini analysis results.

        Args:
            gemini_analysis: Text from Gemini API
            budget_max: Maximum budget for the bundle (None = Infinite)
            categories: List of product categories (default: cleanser, treatment, moisturizer)
            prioritize_rating: Weight rating higher than price
            keywords: List of keywords to filter products

        Returns:
            Dict with bundle products, individual recommendations, and metadata
        """
        conditions_dict = self.extract_conditions_and_severities(gemini_analysis)

        if not conditions_dict:
            return {
                "error": "Could not detect skin condition from analysis",
                "bundle": [],
                "recommendations": [],
                "total_cost": 0
            }

        # Primary condition for the bundle (e.g. the first one found)
        primary_condition = list(conditions_dict.keys())[0]

        # 1. Get the Bundle (Sum <= Budget)
        bundle_result = self.create_product_bundle(
            condition=primary_condition,
            budget_max=budget_max,
            categories=categories,
            prioritize_rating=prioritize_rating,
            keywords=keywords
        )
        
        # 2. Combined Catalog (based on ALL conditions)
        combined_df = self.get_combined_products(list(conditions_dict.keys()), budget_max)
        
        # 3. Shelves
        return {
            "bundle": bundle_result['bundle'],
            "total_cost": bundle_result['total_cost'],
            "recommendations": self.get_randomized_top_n(combined_df, 'rating', n=5),
            "top_rated": self.get_randomized_top_n(combined_df, 'rating', n=5),
            "best_value": self.get_randomized_top_n(combined_df, 'price_numeric', n=5),
            "full_catalog": combined_df.sample(frac=1).to_dict('records')
        }
    
    def recommend_from_analysis(
        self,
        gemini_analysis: str,
        budget_max: float = None,
        top_n: int = 5,
        keywords: List[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts condition and severity from Gemini analysis and recommends products.

        Args:
        gemini_analysis: Text from Gemini API
        budget_max: Maximum budget to filter price
        top_n: Number of products to return
        keywords: List of keywords to filter products

        Returns:
        Dict with catalog and shelves
        """
        conditions_dict = self.extract_conditions_and_severities(gemini_analysis)
        
        if not conditions_dict:
            return {
                "error": "Could not detect skin condition from analysis",
                "full_catalog": [],
                "top_rated": [],
                "best_value": []
            }

        # Combined Catalog (based on ALL conditions)
        combined_df = self.get_combined_products(list(conditions_dict.keys()), budget_max)
        
        # Shelves
        return {
            "top_rated": self.get_randomized_top_n(combined_df, 'rating', n=top_n),
            "best_value": self.get_randomized_top_n(combined_df, 'price_numeric', n=top_n),
            "full_catalog": combined_df.sample(frac=1).to_dict('records') if not combined_df.empty else []
        }
    
    def find_products_by_names(self, product_names: List[str]) -> List[Dict]:
        """
        Find products in the catalog that match the given names using fuzzy matching.
        """
        found_products = []
        
        # Flatten all products into one DataFrame for searching
        # In a real production app, you'd want to cache this or use a proper search engine
        all_products_list = []
        for condition, df in self.products.items():
            # Add condition to the product data so we know where it came from
            df_copy = df.copy()
            df_copy['condition_tag'] = condition
            all_products_list.append(df_copy)
            
        if not all_products_list:
            return []
            
        all_products_df = pd.concat(all_products_list, ignore_index=True)
        
        # Remove duplicates based on title (since same product might appear in multiple conditions)
        all_products_df = all_products_df.drop_duplicates(subset=['title'])
        
        for name in product_names:
            # 1. Try simple case-insensitive substring match
            # "CeraVe" in "CeraVe Hydrating Cleanser" -> Match
            # We use regex=False for speed and safety
            match = all_products_df[all_products_df['title'].str.contains(name, case=False, regex=False)]
            
            if not match.empty:
                # Take the highest rated one from the matches
                best_match = match.sort_values('rating', ascending=False).iloc[0]
                found_products.append(best_match.to_dict())
            else:
                # 2. Optional: Try splitting words if no full match
                # e.g. "CeraVe Cleanser" -> match "CeraVe" AND "Cleanser"
                words = name.split()
                if len(words) > 1:
                    # Create a regex that requires ALL words to be present
                    # (?=.*word1)(?=.*word2)
                    pattern = "".join([f"(?=.*{re.escape(w)})" for w in words])
                    match = all_products_df[all_products_df['title'].str.contains(pattern, case=False, regex=True)]
                    
                    if not match.empty:
                        best_match = match.sort_values('rating', ascending=False).iloc[0]
                        found_products.append(best_match.to_dict())

        return found_products
