import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProductDirectionsEnricher:
    """
    Add 'directions' field to product datasets using category-based templates.

    This is a practical alternative to web scraping, providing generic but accurate
    usage instructions based on product category and active ingredients.
    """

    DATA_DIR = Path("./backend/data")

    # Generic directions templates by product category
    CATEGORY_DIRECTIONS = {
        'cleanser': {
            'default': "Wet face with lukewarm water. Apply a small amount to fingertips and gently massage onto face in circular motions for 30-60 seconds, avoiding eye area. Rinse thoroughly with water and pat dry. Use morning and evening.",
            'salicylic': "Use once daily initially (evening). Wet face, apply to fingertips, massage gently for 30 seconds, rinse thoroughly. Increase to twice daily if tolerated. Follow with moisturizer.",
            'benzoyl': "Start using once daily (evening). Apply to damp skin, lather gently, leave on for 1-2 minutes, rinse thoroughly. May bleach fabrics. Use white towels and pillowcases."
        },
        'treatment': {
            'default': "Apply a thin layer to clean, dry skin. Start with 2-3 times per week, gradually increase frequency as tolerated. Use sunscreen during the day.",
            'retinoid': "Apply a pea-sized amount to entire face (not just spots) at night after cleansing. Start 2-3 times per week for first 2 weeks, then increase to nightly if tolerated. Avoid eye area. Always use SPF 30+ during the day.",
            'benzoyl': "Apply a thin layer to affected areas 1-2 times daily after cleansing. Start once daily to assess tolerance. May bleach clothing and bedding.",
            'salicylic': "Apply to affected areas after cleansing. Start once daily (evening), increase to twice daily if tolerated. Follow with moisturizer.",
            'azelaic': "Apply a thin layer to clean, dry skin morning and/or evening. Start once daily, increase to twice daily as tolerated.",
            'niacinamide': "Apply 3-5 drops to entire face after cleansing, before moisturizer. Can be used morning and evening.",
            'vitamin_c': "Apply 3-5 drops to clean, dry face in the morning before moisturizer and sunscreen. Start every other day, increase to daily use."
        },
        'serum': {
            'default': "Apply 3-5 drops to clean, dry face before moisturizer. Gently press into skin, don't rub vigorously. Use morning and/or evening as directed.",
            'hydrating': "Apply to damp skin after cleansing. Pat gently into face and neck. Follow with moisturizer to seal in hydration.",
            'exfoliating': "Apply to clean, dry skin in the evening. Start 2-3 times per week, increase as tolerated. Always follow with SPF in the morning."
        },
        'moisturizer': {
            'default': "Apply a dime-sized amount to clean face and neck, morning and evening. Gently massage in upward motions until absorbed. Use as the final step before sunscreen (AM) or after treatments (PM).",
            'day': "Apply liberally to face and neck every morning as the final skincare step, 15 minutes before sun exposure. Reapply every 2 hours if outdoors.",
            'night': "Apply to clean, dry skin every evening after treatments/serums. Massage gently into face and neck. Use as final step in PM routine."
        },
        'cream': {
            'default': "Apply to clean, dry skin morning and/or evening. Take a small amount, warm between fingertips, and gently press onto face and neck. Allow to absorb before applying makeup or sunscreen."
        },
        'lotion': {
            'default': "Apply to clean skin morning and/or evening. Dispense 2-3 pumps, apply to face and neck in gentle upward strokes. Allow to absorb fully."
        },
        'gel': {
            'default': "Apply a thin layer to clean, dry skin. Start with once daily application, increase to twice daily as tolerated. Allow to absorb completely before applying other products.",
            'spot_treatment': "Apply a small amount directly to individual blemishes after cleansing. Use 1-2 times daily. Do not apply to entire face unless directed."
        },
        'toner': {
            'default': "After cleansing, apply to cotton pad and gently sweep over face and neck, avoiding eye area. Or dispense into palms and press into skin. Use morning and evening before serums.",
            'exfoliating': "Use in the evening after cleansing. Apply with cotton pad or fingertips, avoiding eye area. Start 2-3 times per week, increase as tolerated. Follow with hydrating products."
        },
        'essence': {
            'default': "After cleansing and toning, dispense 2-3 pumps into palms. Gently press into face and neck. Pat until absorbed. Follow with serum and moisturizer."
        },
        'sunscreen': {
            'default': "Apply liberally to all exposed skin 15 minutes before sun exposure. Use approximately 1/4 teaspoon for face alone. Reapply every 2 hours, or immediately after swimming/sweating. Use as final step in morning routine.",
            'chemical': "Apply to clean, dry skin as the final step in morning routine. Wait 15 minutes before sun exposure. Reapply every 2 hours when outdoors.",
            'physical': "Apply as last step in AM routine. May be applied immediately before sun exposure. Reapply every 2 hours. Can leave white cast."
        },
        'mask': {
            'default': "Apply an even layer to clean, dry skin. Avoid eye and lip area. Leave on for 10-15 minutes or as directed. Rinse thoroughly with lukewarm water. Use 1-3 times per week.",
            'clay': "Apply to clean skin, avoiding eye area. Leave on until nearly dry (10-15 minutes). Do not let completely dry. Rinse with lukewarm water. Use 1-2 times per week.",
            'sheet': "After cleansing, apply sheet mask to face, smoothing out air bubbles. Leave on for 15-20 minutes. Remove and gently pat remaining serum into skin. Do not rinse. Use 1-3 times per week."
        },
        'patch': {
            'default': "Cleanse and completely dry the affected area. Apply patch directly over blemish, pressing edges to ensure adhesion. Leave on for 6-8 hours or overnight. Remove gently and discard. Use on individual blemishes as needed.",
            'hydrocolloid': "Apply to clean, dry blemish (works best on surface pustules). Leave on for 6-12 hours or until patch turns white/opaque. Remove gently, discard, and cleanse area."
        },
        'spot_treatment': {
            'default': "After cleansing, apply a small amount directly to individual blemishes. Do not apply to entire face. Use 1-2 times daily. Allow to dry completely before applying other products."
        },
        'oil': {
            'default': "Apply 2-4 drops to clean, damp skin. Gently press into face and neck. Can be mixed with moisturizer or used as final step. Use evening only initially, add morning use if desired.",
            'cleansing': "Apply to dry face, massage gently for 1-2 minutes to dissolve makeup and impurities. Add water to emulsify, then rinse thoroughly. Follow with water-based cleanser if double cleansing."
        },
        'other': {
            'default': "Follow product-specific instructions on packaging. If unsure, consult with a dermatologist or skincare professional. Start with minimal use and increase as tolerated."
        }
    }

    # Ingredient-specific modifiers
    INGREDIENT_KEYWORDS = {
        'retinoid': ['retinol', 'retinoid', 'adapalene', 'differin', 'tretinoin', 'retin-a'],
        'benzoyl': ['benzoyl peroxide', 'benzoyl', 'peroxide'],
        'salicylic': ['salicylic acid', 'salicylic', 'bha'],
        'azelaic': ['azelaic acid', 'azelaic'],
        'niacinamide': ['niacinamide', 'vitamin b3'],
        'vitamin_c': ['vitamin c', 'ascorbic acid', 'l-ascorbic'],
        'aha': ['glycolic', 'lactic', 'mandelic', 'aha', 'alpha hydroxy'],
        'hydrating': ['hyaluronic', 'hydrating', 'moisture', 'ceramide'],
        'exfoliating': ['exfoliat', 'peel', 'acid']
    }

    def __init__(self):
        self.conditions = [
            "acne", "rosacea", "eczema", "psoriasis",
            "melasma", "dark_spots", "hyperpigmentation",
            "dry_skin", "oily_skin", "sensitive_skin"
        ]

    def _categorize_product(self, title: str) -> str:
        """Determine product category from title."""
        if pd.isna(title):
            return 'other'

        title_lower = title.lower()

        # Priority order matters - more specific first
        category_keywords = {
            'patch': ['patch', 'sticker', 'pimple patch', 'acne patch'],
            'spot_treatment': ['spot treatment', 'spot corrector', 'blemish treatment'],
            'sunscreen': ['sunscreen', 'spf', 'sun protection', 'sun block'],
            'mask': ['mask', 'masque'],
            'toner': ['toner', 'astringent'],
            'essence': ['essence'],
            'serum': ['serum'],
            'cleanser': ['cleanser', 'face wash', 'wash', 'cleansing', 'facial cleanser'],
            'treatment': ['treatment', 'gel', 'adapalene', 'differin'],
            'cream': ['cream', 'creme'],
            'lotion': ['lotion'],
            'moisturizer': ['moisturizer', 'moisturiser', 'hydrating cream', 'hydrating lotion'],
            'oil': ['oil', 'facial oil']
        }

        for category, keywords in category_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                return category

        return 'other'

    def _detect_ingredients(self, title: str, product_name: str = '') -> list:
        """Detect key ingredients from product title/name."""
        detected = []
        combined_text = f"{title} {product_name}".lower()

        for ingredient_type, keywords in self.INGREDIENT_KEYWORDS.items():
            if any(keyword in combined_text for keyword in keywords):
                detected.append(ingredient_type)

        return detected

    def _get_directions(self, category: str, ingredients: list) -> str:
        """Get appropriate directions based on category and ingredients."""

        # Get category-specific directions
        category_directions = self.CATEGORY_DIRECTIONS.get(category, {})

        # Check for ingredient-specific directions first
        for ingredient in ingredients:
            if ingredient in category_directions:
                return category_directions[ingredient]

        # Fall back to default for category
        if 'default' in category_directions:
            return category_directions['default']

        # Ultimate fallback
        return self.CATEGORY_DIRECTIONS['other']['default']

    def enrich_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'directions' column to product dataframe."""

        if df.empty:
            return df

        # Make a copy to avoid modifying original
        enriched_df = df.copy()

        # Detect category and ingredients for each product
        enriched_df['category'] = enriched_df['title'].apply(self._categorize_product)

        # Create directions based on category and title
        def create_directions(row):
            category = row.get('category', 'other')
            title = row.get('title', '')
            ingredients = self._detect_ingredients(title)
            return self._get_directions(category, ingredients)

        enriched_df['directions'] = enriched_df.apply(create_directions, axis=1)

        logger.info(f"Added directions to {len(enriched_df)} products")

        return enriched_df

    def enrich_all_datasets(self, save_enriched: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all product datasets, add directions, and optionally save.

        Args:
            save_enriched: Whether to save enriched datasets to new CSV files

        Returns:
            Dict mapping condition -> enriched DataFrame
        """
        enriched_products = {}

        for condition in self.conditions:
            csv_path = self.DATA_DIR / f"{condition}_products_dataset.csv"

            if not csv_path.exists():
                logger.warning(f"File not found: {csv_path}")
                continue

            try:
                # Load original dataset
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded {len(df)} products for {condition}")

                # Add directions
                enriched_df = self.enrich_dataset(df)
                enriched_products[condition] = enriched_df

                # Save enriched dataset
                if save_enriched:
                    output_path = self.DATA_DIR / f"{condition}_products_enriched.csv"
                    enriched_df.to_csv(output_path, index=False)
                    logger.info(f"✓ Saved enriched dataset: {output_path}")

            except Exception as e:
                logger.error(f"Failed to process {condition}: {e}")
                continue

        return enriched_products

    def get_directions_summary(self, enriched_products: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate summary of direction types across all products."""

        summary_data = []

        for condition, df in enriched_products.items():
            if df.empty:
                continue

            category_counts = df['category'].value_counts().to_dict()

            summary_data.append({
                'condition': condition,
                'total_products': len(df),
                'with_directions': df['directions'].notna().sum(),
                'categories': ', '.join([f"{cat}({count})" for cat, count in list(category_counts.items())[:5]])
            })

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    enricher = ProductDirectionsEnricher()

    print("\n" + "="*70)
    print("PRODUCT DIRECTIONS ENRICHMENT")
    print("="*70)

    # Enrich all datasets
    enriched = enricher.enrich_all_datasets(save_enriched=True)

    # Generate summary
    summary = enricher.get_directions_summary(enriched)
    print("\nEnrichment Summary:")
    print(summary.to_string(index=False))

    # Show sample directions for each category
    print("\n" + "="*70)
    print("SAMPLE DIRECTIONS BY CATEGORY")
    print("="*70)

    for condition, df in enriched.items():
        if df.empty:
            continue

        print(f"\n{condition.upper().replace('_', ' ')}:")
        print("-" * 70)

        # Sample one product from each category
        for category in df['category'].unique()[:3]:
            sample = df[df['category'] == category].iloc[0]
            print(f"\nCategory: {category}")
            print(f"Product: {sample['title'][:60]}...")
            print(f"Directions: {sample['directions'][:150]}...")

    print("\n" + "="*70 + "\n")
