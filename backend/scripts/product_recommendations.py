"""
Product recommendation system based on acne detection results

Maps detected acne types and severity to specific skincare product recommendations.

Usage:
    python scripts/product_recommendations.py --lesions '{"papules": 5, "pustules": 2}'
    python scripts/product_recommendations.py --severity moderate --dominant-type papules
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class ProductRecommendationSystem:
    """Recommend skincare products based on acne analysis"""

    def __init__(self, products_db_path: str = "data/products.json"):
        """Load product database"""
        db_path = Path(products_db_path)

        if not db_path.exists():
            raise FileNotFoundError(f"Product database not found: {products_db_path}")

        with open(db_path, 'r') as f:
            self.products_db = json.load(f)

    def recommend(
        self,
        detection_results: Dict[str, Any],
        budget: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Generate product recommendations based on detection results

        Args:
            detection_results: {
                'comedones_count': int,
                'papules_count': int,
                'pustules_count': int,
                'nodules_count': int,
                'total_detections': int
            }
            budget: 'budget' | 'moderate' | 'premium'

        Returns:
            Recommendation package with products, routine, timeline
        """
        # Determine dominant lesion type
        lesion_counts = {
            'comedones': detection_results.get('comedones_count', 0),
            'papules': detection_results.get('papules_count', 0),
            'pustules': detection_results.get('pustules_count', 0),
            'nodules': detection_results.get('nodules_count', 0)
        }

        total_lesions = sum(lesion_counts.values())

        # Determine severity
        severity = self._calculate_severity(total_lesions)

        # Find dominant type
        dominant_type = max(lesion_counts, key=lesion_counts.get)

        if lesion_counts[dominant_type] == 0:
            # No acne detected
            return self._clear_skin_routine()

        # Check for nodules (requires medical intervention)
        if lesion_counts['nodules'] > 0:
            return self._nodule_warning()

        # Get recommendations for dominant type
        recommendations = self._get_recommendations(
            dominant_type,
            severity,
            lesion_counts,
            budget
        )

        return recommendations

    def _calculate_severity(self, total_count: int) -> str:
        """Calculate severity based on lesion count"""
        if total_count >= 30:
            return 'severe'
        elif total_count >= 10:
            return 'moderate'
        elif total_count > 0:
            return 'mild'
        else:
            return 'clear'

    def _get_recommendations(
        self,
        dominant_type: str,
        severity: str,
        lesion_counts: Dict[str, int],
        budget: str
    ) -> Dict[str, Any]:
        """Get product recommendations"""

        # Get products for dominant type
        type_products = self.products_db.get(dominant_type, {})

        # Build recommendation package
        recommendations = {
            'analysis': {
                'dominant_type': dominant_type,
                'severity': severity,
                'lesion_breakdown': lesion_counts,
                'total_lesions': sum(lesion_counts.values())
            },
            'description': type_products.get('description', ''),
            'products': {
                'cleanser': self._select_products(
                    type_products.get('cleansers', []),
                    budget,
                    count=1
                ),
                'treatment': self._select_products(
                    type_products.get('treatments', []),
                    budget,
                    count=2
                ),
                'moisturizer': self._select_products(
                    type_products.get('moisturizers', []),
                    budget,
                    count=1
                )
            },
            'routine': {
                'morning': [
                    "1. Cleanse with recommended cleanser",
                    "2. Apply treatment (if AM-safe)",
                    "3. Moisturize",
                    "4. SPF 30+ (ESSENTIAL with retinoids/acids)"
                ],
                'evening': [
                    "1. Remove makeup/sunscreen",
                    "2. Cleanse",
                    "3. Apply treatment (retinoid/BHA/BP)",
                    "4. Moisturize",
                    "5. Spot treat if needed"
                ],
                'summary': type_products.get('routine', '')
            },
            'timeline': type_products.get('timeline', '8-12 weeks for visible results'),
            'pro_tip': type_products.get('pro_tip', ''),
            'when_to_see_doctor': type_products.get('when_to_see_doctor', ''),
            'general_tips': self.products_db['general_tips'],
            'severity_info': self.products_db['severity_guidelines'].get(severity, {})
        }

        return recommendations

    def _select_products(
        self,
        products: List[Dict],
        budget: str,
        count: int = 1
    ) -> List[Dict]:
        """Select products based on budget"""
        if not products:
            return []

        # Price ranges (rough USD estimates)
        budget_ranges = {
            'budget': (0, 15),
            'moderate': (10, 30),
            'premium': (20, 100)
        }

        min_price, max_price = budget_ranges.get(budget, (0, 100))

        # Filter by budget
        filtered = []
        for product in products:
            price_str = product.get('price_range', '$0')
            # Extract max price from range like "$12-15" or "$32"
            try:
                prices = [int(p.strip('$')) for p in price_str.replace('-', ' ').split()]
                max_product_price = max(prices)

                if min_price <= max_product_price <= max_price:
                    filtered.append(product)
            except:
                filtered.append(product)  # Include if can't parse price

        # If no products in budget, return cheapest options
        if not filtered:
            filtered = sorted(products, key=lambda x: x.get('price_range', '$0'))[:count]

        return filtered[:count]

    def _clear_skin_routine(self) -> Dict[str, Any]:
        """Recommendations for clear skin (maintenance)"""
        return {
            'analysis': {
                'dominant_type': 'none',
                'severity': 'clear',
                'message': 'No acne detected! Focus on prevention and maintenance.'
            },
            'products': {
                'cleanser': [{
                    'name': 'Gentle, pH-balanced cleanser',
                    'why_it_works': 'Maintains skin barrier without stripping',
                    'examples': 'CeraVe Hydrating Cleanser, Vanicream Gentle Cleanser'
                }],
                'treatment': [{
                    'name': 'Optional: Niacinamide or Gentle BHA',
                    'why_it_works': 'Prevents future breakouts, maintains pore clarity',
                    'examples': 'The Ordinary Niacinamide, Paula\'s Choice BHA 2%'
                }],
                'moisturizer': [{
                    'name': 'Lightweight, non-comedogenic moisturizer',
                    'why_it_works': 'Keeps skin hydrated without clogging pores',
                    'examples': 'Neutrogena Hydro Boost, CeraVe PM'
                }]
            },
            'routine': {
                'summary': 'Cleanse → (Treatment) → Moisturize → SPF 30+',
                'pro_tip': 'Maintain consistent routine even with clear skin to prevent future breakouts'
            }
        }

    def _nodule_warning(self) -> Dict[str, Any]:
        """Special warning for nodular acne"""
        nodule_info = self.products_db.get('nodules', {})

        return {
            'analysis': {
                'dominant_type': 'nodules',
                'severity': 'severe',
                'warning': '⚠️ NODULAR ACNE DETECTED - REQUIRES MEDICAL TREATMENT'
            },
            'message': '''
Nodular acne is a severe form that requires professional medical treatment.
Over-the-counter products are insufficient for this condition.

IMMEDIATE ACTION REQUIRED:
1. Schedule appointment with dermatologist ASAP
2. Do NOT attempt to pop or squeeze nodules (causes scarring)
3. While waiting for appointment, use gentle skincare only

Nodular acne can cause permanent scarring and typically requires prescription medication.
            ''',
            'products': nodule_info,
            'prescription_options': nodule_info.get('prescription_options', []),
            'immediate_relief': [
                'Apply ice wrapped in cloth for 5-10 min, 3x daily (reduces pain/swelling)',
                'Use gentle, non-irritating cleanser',
                'Avoid harsh scrubs or exfoliants',
                'Take over-the-counter ibuprofen for pain (if medically safe for you)'
            ]
        }

    def format_recommendations(self, recommendations: Dict[str, Any]) -> str:
        """Format recommendations as readable text"""
        output = []

        # Header
        analysis = recommendations.get('analysis', {})
        output.append("="*70)
        output.append("PERSONALIZED ACNE TREATMENT RECOMMENDATIONS")
        output.append("="*70)
        output.append("")

        # Analysis summary
        output.append("📊 Analysis:")
        output.append(f"  Severity: {analysis.get('severity', 'unknown').upper()}")

        if 'lesion_breakdown' in analysis:
            output.append(f"  Total lesions: {analysis['total_lesions']}")
            output.append(f"  Dominant type: {analysis['dominant_type']}")

        if 'message' in analysis or 'warning' in analysis:
            output.append("")
            output.append(analysis.get('message', analysis.get('warning', '')))

        output.append("")

        # Products
        products = recommendations.get('products', {})

        if 'cleanser' in products and products['cleanser']:
            output.append("🧼 RECOMMENDED CLEANSER:")
            for p in products['cleanser']:
                output.append(f"  • {p['name']}")
                output.append(f"    Ingredient: {p.get('key_ingredient', 'N/A')}")
                output.append(f"    Price: {p.get('price_range', 'N/A')}")
                output.append(f"    Why: {p.get('why_it_works', 'N/A')}")
                output.append("")

        if 'treatment' in products and products['treatment']:
            output.append("💊 RECOMMENDED TREATMENTS:")
            for p in products['treatment']:
                output.append(f"  • {p['name']}")
                output.append(f"    Ingredient: {p.get('key_ingredient', 'N/A')}")
                output.append(f"    Price: {p.get('price_range', 'N/A')}")
                output.append(f"    Why: {p.get('why_it_works', 'N/A')}")
                output.append(f"    How to use: {p.get('how_to_use', 'N/A')}")
                output.append("")

        if 'moisturizer' in products and products['moisturizer']:
            output.append("💧 RECOMMENDED MOISTURIZER:")
            for p in products['moisturizer']:
                output.append(f"  • {p['name']}")
                output.append(f"    Ingredient: {p.get('key_ingredient', 'N/A')}")
                output.append(f"    Price: {p.get('price_range', 'N/A')}")
                output.append(f"    Why: {p.get('why_it_works', 'N/A')}")
                output.append("")

        # Routine
        routine = recommendations.get('routine', {})
        if routine:
            output.append("📅 DAILY ROUTINE:")
            output.append(f"  {routine.get('summary', '')}")
            output.append("")

        # Timeline
        if 'timeline' in recommendations:
            output.append(f"⏱️  Expected results: {recommendations['timeline']}")
            output.append("")

        # Pro tip
        if 'pro_tip' in recommendations:
            output.append(f"💡 Pro Tip: {recommendations['pro_tip']}")
            output.append("")

        # When to see doctor
        if 'when_to_see_doctor' in recommendations:
            output.append(f"⚕️  See a doctor: {recommendations['when_to_see_doctor']}")
            output.append("")

        output.append("="*70)

        return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Get product recommendations based on acne detection'
    )
    parser.add_argument(
        '--lesions',
        type=str,
        help='JSON string with lesion counts, e.g. \'{"papules": 5, "pustules": 2}\''
    )
    parser.add_argument(
        '--severity',
        type=str,
        choices=['mild', 'moderate', 'severe'],
        help='Severity level'
    )
    parser.add_argument(
        '--dominant-type',
        type=str,
        choices=['comedones', 'papules', 'pustules', 'nodules'],
        help='Dominant lesion type'
    )
    parser.add_argument(
        '--budget',
        type=str,
        choices=['budget', 'moderate', 'premium'],
        default='moderate',
        help='Budget level (default: moderate)'
    )

    args = parser.parse_args()

    # Initialize system
    recommender = ProductRecommendationSystem()

    # Build detection results
    if args.lesions:
        lesion_counts = json.loads(args.lesions)
        detection_results = {
            'comedones_count': lesion_counts.get('comedones', 0),
            'papules_count': lesion_counts.get('papules', 0),
            'pustules_count': lesion_counts.get('pustules', 0),
            'nodules_count': lesion_counts.get('nodules', 0),
            'total_detections': sum(lesion_counts.values())
        }
    else:
        # Example detection
        detection_results = {
            'comedones_count': 3,
            'papules_count': 7,
            'pustules_count': 2,
            'nodules_count': 0,
            'total_detections': 12
        }

    # Get recommendations
    recommendations = recommender.recommend(detection_results, budget=args.budget)

    # Print formatted output
    print(recommender.format_recommendations(recommendations))


if __name__ == '__main__':
    main()
