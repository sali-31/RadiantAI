import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProductDataCleaner:
    """Clean and validate Amazon product data
    Features:
        - Remove duplicates (by ASIN)
        - Validate required fields
        - Parse price strings to floats
        - Standardize data types
    """

    # Use absolute path relative to this file to avoid CWD issues
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
    REQUIRED_FIELDS = [
        "title",
        "asin",
        "price",
        "rating",
        "link",
        "thumbnail"
    ]

    def __init__(self):
        self.conditions = [
          "acne", "rosacea", "eczema", "psoriasis",
            "melasma", "dark_spots", "hyperpigmentation",
            "dry_skin", "oily_skin", "sensitive_skin"  
        ]

    def load_all_products(self) -> Dict[str, pd.DataFrame]:
        """
        Load all product CSVs

        Returns:
            Dict mapping condition -> DataFrame 
        """
        products = {}

        for condition in self.conditions:
            # Try to load enriched dataset first, fall back to original
            enriched_path = self.DATA_DIR/f"{condition}_products_enriched.csv"
            csv_path = self.DATA_DIR/f"{condition}_products_dataset.csv"

            load_path = enriched_path if enriched_path.exists() else csv_path

            if load_path.exists():
                try:
                    products[condition] = pd.read_csv(load_path)
                    dataset_type = "enriched" if load_path == enriched_path else "original"
                    logger.info(f" Loaded {len(products[condition])} {dataset_type} products for condition: {condition}")
                
                except Exception as e:
                    logger.error(f"Failed to load {condition} products: {e}")
                    products[condition] = pd.DataFrame()
            else:
                logger.warning(f"File not found: {csv_path}")
                products[condition] = pd.DataFrame()
        return products
    

    def clean_product_data(self, df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Clean and validate a single condition's product data
        
        Steps:
        1. Remove duplicates
        2. Drop rows with missing critical fields
        3. Convert price strings to floats
        4. Validate ratings
        5. Filter low-quality products
        6. Add metadata
        """
        original_count = len(df)
        logger.info(f"Cleaning {condition} products (original: {original_count})")

        if df.empty:
            return df
        
        # 1. Remove exact duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_count - len(df)} exact duplicates")

        # 2. Remove duplicates by ASIN (Amazon's unique product ID) 
        if 'asin' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset="asin")
            logger.info(f"Removed {before - len(df)} duplicates by ASIN")

        # 3. Check for required fields:
        missing_fields = [field for field in self.REQUIRED_FIELDS if field not in df.columns]
        if missing_fields:
            logger.warning(f"Missing columns: {', '.join(missing_fields)}")

        # 4. Drop rows with missing critical data
        critical_fields = ['title', 'asin', 'link', 'price']
        before = len(df)
        df = df.dropna(subset=critical_fields, how='any')
        logger.info(f"Removed {before - len(df)} rows with missing critical data")

        # 5. Clean price field
        df = self._clean_price(df)

        # 6. Clean ratings field
        df = self._clean_ratings(df)

        # 7. Clean reviews field
        df = self._clean_reviews(df)

        # 8. Filter low-quality products
        df = self._filter_low_quality(df)

        # 9. Add metadata
        df['condition'] = condition
        df['data_source'] = 'amazon'

        # 10. Standardize column order (keep all existing columns, including category and directions)
        base_columns = ['asin', 'title', 'price', 'price_numeric', 'rating',
                        'reviews', 'link', 'thumbnail', 'condition', 'data_source']

        # Preserve additional columns like 'category' and 'directions' if they exist
        extra_columns = [col for col in df.columns if col not in base_columns]
        column_order = base_columns + extra_columns

        df = df[[col for col in column_order if col in df.columns]]

        logger.info(f"Cleaned {condition} products dataset (final count: {len(df)})")
        return df
    
    def _clean_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse price strings to a numeric value
        Example:
        "$15.73" -> 15.73
        "$2,372.99 -> 2372.99
        "" -> NaN
        """
        if 'price' not in df.columns:
            return df
        
        def parse_price(price_str: str) -> Optional[float]:
            if pd.isna(price_str) or price_str == "":
                return np.nan
            
            # Remove dollar sign ($), commas(,)
            cleaned_price = str(price_str).replace("$", "").replace(",", "").strip()

            try:
                return float(cleaned_price)
            except ValueError:
                logger.warning(f"Failed to parse price: {price_str}")
                return np.nan
        
        df['price_numeric'] = df['price'].apply(parse_price)

        # Count how many rows failed to parse price
        failed = df['price_numeric'].isna().sum()
        if failed > 0:
            logger.warning(f"Failed to parse {failed} prices")

        return df
    
    def _clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and cleans the ratings field

        Ratings should be between 0.0 - 5.0
        """

        if 'rating' not in df.columns:
            return df
        
        # Convert to numeric
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Validate range
        invalid = ((df['rating'] < 0.0) | (df['rating'] > 5.0)).sum()
        if invalid > 0 :
            logger.warning(f"Found {invalid} invalid ratings")
            df.loc[(df['rating'] < 0) | (df['rating'] > 5), 'rating'] = np.nan

        return df
    
    def _clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans reviews count field
        Convert to numeric, handle edge cases
        """

        if 'reviews' not in df.columns:
            return df
        
        df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
        df['reviews'] = df['reviews'].fillna(0).astype(int)

        return df

    def _filter_low_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out low-quality products

        Criteria:
        - Must have a rating >= 3.5
        - Must have at least 50 reviews (or no filter if too few products)
        - Must have a valid price
        """
        before = len(df)
        
        # Filter by rating
        if 'rating' in df.columns:
            df = df[df['rating'] >= 3.5]

        # Filter by reviews
        if 'reviews' in df.columns:
            df = df[df['reviews'] >= 50]

        # Filter by valid price
        if 'price_numeric' in df.columns:
            df = df[df['price_numeric'].notna()]
            df = df[df['price_numeric'] > 0]

        logger.info(f"Filtered out {before - len(df)} low-quality products")

        return df
    
    def get_duplicate_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a report on duplicates

        Returns:
            Dict with duplicate statistics
        """
        report = {
            'total_products': len(df),
            'exact_duplicates': df.duplicated().sum(),
            'asin_duplicates': df.duplicated(subset='asin').sum(),
            'duplicate_asins': 0,
            'duplicate_titles': 0
        }

        if 'asin' in df.columns:
            report['duplicate_asins'] = df['asin'].duplicated().sum()

        if 'title' in df.columns:
            report['duplicate_titles'] = df['title'].duplicated().sum()

        return report
    
    def get_missing_data_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a report on missing data
        
        Returns:
            Dict with missing data statistics
        """
        report = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                report[col] = {
                    'count': missing_count,
                    'percentage': round(missing_pct, 2)
                }
        
        return report
    
    def clean_all_products(self, save_cleaned=True) -> Dict[str, pd.DataFrame]:
        """
        Load and clean all product datasets
        
        Args:
            save_cleaned: Whether to save cleaned data to new CSVs
        
        Returns:
            Dict mapping condition -> cleaned DataFrame
        """
        all_products = self.load_all_products()
        cleaned_products = {}
        
        for condition, df in all_products.items():
            cleaned_df = self.clean_product_data(df, condition)
            cleaned_products[condition] = cleaned_df
            
            if save_cleaned and not cleaned_df.empty:
                output_path = self.DATA_DIR / f"{condition}_products_cleaned.csv"
                cleaned_df.to_csv(output_path, index=False)
                logger.info(f"Saved cleaned data to {output_path}")
        
        return cleaned_products
    
    def get_summary_statistics(self, cleaned_products: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics across all conditions
        
        Returns:
            DataFrame with statistics per condition
        """
        stats = []
        
        for condition, df in cleaned_products.items():
            if df.empty:
                continue
            
            stat = {
                'condition': condition,
                'total_products': len(df),
                'avg_price': df['price_numeric'].mean() if 'price_numeric' in df.columns else np.nan,
                'avg_rating': df['rating'].mean() if 'rating' in df.columns else np.nan,
                'avg_reviews': df['reviews'].mean() if 'reviews' in df.columns else np.nan,
                'price_range': f"${df['price_numeric'].min():.2f} - ${df['price_numeric'].max():.2f}" if 'price_numeric' in df.columns else "N/A"
            }
            stats.append(stat)
        
        return pd.DataFrame(stats)
    

if __name__ == "__main__":
    cleaner = ProductDataCleaner()

    # Clean all products
    cleaned = cleaner.clean_all_products()

    # Generate summary
    summary = cleaner.get_summary_statistics(cleaned)
    print("\n"+"="*60)
    print("PRODUCT DATA SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))

    # Get detailed report for each condition
    for condition in cleaner.conditions:
        if condition not in cleaned:
            print("="*60)
            print(f"{condition.upper().replace('_', ' ')} - NO DATA AVAILABLE")
            print("="*60)
            continue
        condition_df = cleaned[condition]
        
        print("="*60)
        print(f"{condition.upper()} - DATA QUALITY REPORT")
        print("="*60)

        # Duplicate  data report
        dupe_report = cleaner.get_duplicate_report(condition_df)
        print(f"\nDuplicates: {dupe_report}")

        # Missing data report
        missing_report = cleaner.get_missing_data_report(condition_df)
        print(f"\nMissing Data: {missing_report}" if missing_report else f"Missing Data: None - All fields complete")

        # Top 10 products by rating
        if len(condition_df) >= 10:
            top_10_products = condition_df.nlargest(10, 'rating')[['title', 'price', 'rating', 'reviews']]
            print("\nTop 10 Products by Rating:")
        else:
            top_10_products = condition_df.sort_values('rating', ascending=False)[['title', 'price', 'rating', 'reviews']]

        print(top_10_products.to_string(index=False))
        
        if 'price_numeric' in condition_df.columns:
            print(f"\nPrice Statistics:")
            print(f"  Min: ${condition_df['price_numeric'].min():.2f}")
            print(f"  Max: ${condition_df['price_numeric'].max():.2f}")
            print(f"  Average: ${condition_df['price_numeric'].mean():.2f}")
            print(f"  Median: ${condition_df['price_numeric'].median():.2f}")

    print("="*60+"\n")
