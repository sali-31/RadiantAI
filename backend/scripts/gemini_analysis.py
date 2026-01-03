"""
Gemini Vision API integration for acne analysis

This script uses Google's Gemini 1.5 Flash model to analyze skin images
and provide natural language insights about acne severity, types, and recommendations.

Usage:
    # Set API key
    export GEMINI_API_KEY="your_api_key_here"

    # Analyze single image
    python scripts/gemini_analysis.py --image path/to/image.jpg

    # Analyze with custom prompt
    python scripts/gemini_analysis.py --image path/to/image.jpg --detailed

    # Batch analysis
    python scripts/gemini_analysis.py --folder data/test_images/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from PIL import Image
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiAcneAnalyzer:
    """Gemini Vision API wrapper for acne detection and analysis"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini analyzer

        Args:
            api_key: Google API key (if None, reads from GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter.\n"
                "Get your free API key at: https://makersuite.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Use Gemini 1.5 Flash (fast, free tier: 15 RPM, 1M tokens/day)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        logger.info("✓ Gemini Vision API initialized (model: gemini-1.5-flash)")

    def analyze_acne(
        self,
        image_path: str,
        detailed: bool = False,
        temperature: float = 0.4
    ) -> Dict[str, Any]:
        """
        Analyze acne in image using Gemini Vision

        Args:
            image_path: Path to skin image
            detailed: If True, request more detailed analysis
            temperature: Model temperature (0.0-1.0, lower = more focused)

        Returns:
            Dictionary with analysis results:
            {
                'severity': 'mild' | 'moderate' | 'severe',
                'lesion_types': ['papules', 'pustules', ...],
                'estimated_count': int,
                'concerns': ['inflammation', 'scarring', ...],
                'recommendations': ['treatment1', 'treatment2', ...],
                'skin_type': 'oily' | 'dry' | 'combination' | 'normal',
                'confidence': float,
                'raw_response': str
            }
        """
        # Load image
        image = Image.open(image_path)

        # Construct prompt
        if detailed:
            prompt = self._get_detailed_prompt()
        else:
            prompt = self._get_standard_prompt()

        logger.info(f"Analyzing image: {image_path}")

        try:
            # Generate analysis
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=2048,
                )
            )

            # Parse response
            result = self._parse_response(response.text)
            result['raw_response'] = response.text

            logger.info(f"✓ Analysis complete - Severity: {result.get('severity', 'unknown')}")

            return result

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise

    def _get_standard_prompt(self) -> str:
        """Get standard analysis prompt"""
        return """
You are an expert dermatology AI assistant. Analyze this facial skin image for acne.

Provide your analysis in JSON format with these fields:

{
  "severity": "mild" | "moderate" | "severe",
  "lesion_types": ["list of detected types: comedones, papules, pustules, nodules, cysts"],
  "estimated_count": <total number of visible lesions>,
  "concerns": ["list of concerns: inflammation, redness, scarring, hyperpigmentation"],
  "skin_type": "oily" | "dry" | "combination" | "normal",
  "recommendations": [
    "specific treatment recommendation 1",
    "specific treatment recommendation 2",
    "specific treatment recommendation 3"
  ],
  "summary": "Brief 2-3 sentence summary of the analysis"
}

Guidelines:
- Be precise and objective
- Base severity on lesion count and type (mild: <10, moderate: 10-30, severe: >30)
- Only mention lesion types you actually see
- Provide actionable, specific recommendations
- Consider both over-the-counter and prescription options where appropriate
"""

    def _get_detailed_prompt(self) -> str:
        """Get detailed analysis prompt"""
        return """
You are an expert dermatology AI assistant. Provide a comprehensive analysis of this skin image.

Analyze and return JSON with:

{
  "severity": "mild" | "moderate" | "severe",
  "severity_explanation": "detailed explanation of severity assessment",

  "lesion_analysis": {
    "comedones": {"count": <int>, "locations": ["area1", "area2"]},
    "papules": {"count": <int>, "locations": ["area1", "area2"]},
    "pustules": {"count": <int>, "locations": ["area1", "area2"]},
    "nodules": {"count": <int>, "locations": ["area1", "area2"]},
    "cysts": {"count": <int>, "locations": ["area1", "area2"]}
  },

  "skin_characteristics": {
    "type": "oily" | "dry" | "combination" | "normal",
    "tone": "fair" | "medium" | "olive" | "dark",
    "texture": "smooth" | "rough" | "uneven",
    "concerns": ["redness", "inflammation", "scarring", "hyperpigmentation"]
  },

  "treatment_plan": {
    "immediate": ["action 1", "action 2"],
    "short_term": ["1-4 weeks recommendations"],
    "long_term": ["maintenance recommendations"],
    "otc_products": ["product type 1", "product type 2"],
    "prescription_options": ["option 1 (if severe)", "option 2"]
  },

  "lifestyle_recommendations": [
    "diet suggestion",
    "skincare routine",
    "habits to avoid"
  ],

  "follow_up": "When to see dermatologist / reassess",

  "confidence": <0.0-1.0 confidence score>,

  "summary": "Comprehensive 3-5 sentence summary"
}

Be thorough, evidence-based, and provide specific, actionable guidance.
"""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini response into structured format

        Args:
            response_text: Raw text response from Gemini

        Returns:
            Parsed dictionary
        """
        try:
            # Remove markdown code blocks if present
            clean_text = response_text.strip()
            if clean_text.startswith('```json'):
                clean_text = clean_text[7:]
            if clean_text.startswith('```'):
                clean_text = clean_text[3:]
            if clean_text.endswith('```'):
                clean_text = clean_text[:-3]

            clean_text = clean_text.strip()

            # Parse JSON
            result = json.loads(clean_text)

            # Validate required fields
            if 'severity' not in result:
                result['severity'] = 'unknown'

            if 'estimated_count' not in result and 'lesion_analysis' in result:
                # Calculate from detailed analysis
                total = sum(
                    lesion.get('count', 0)
                    for lesion in result['lesion_analysis'].values()
                    if isinstance(lesion, dict)
                )
                result['estimated_count'] = total

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Raw response: {response_text[:200]}...")

            # Fallback: extract key information manually
            return {
                'severity': self._extract_severity(response_text),
                'summary': response_text[:500],
                'parse_error': str(e)
            }

    def _extract_severity(self, text: str) -> str:
        """Extract severity from unstructured text"""
        text_lower = text.lower()
        if 'severe' in text_lower:
            return 'severe'
        elif 'moderate' in text_lower:
            return 'moderate'
        elif 'mild' in text_lower:
            return 'mild'
        else:
            return 'unknown'

    def compare_with_yolo(
        self,
        gemini_result: Dict[str, Any],
        yolo_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare Gemini analysis with YOLO detection results

        Args:
            gemini_result: Result from analyze_acne()
            yolo_result: Result from YOLO detection

        Returns:
            Comparison analysis
        """
        comparison = {
            'lesion_count': {
                'gemini': gemini_result.get('estimated_count', 0),
                'yolo': yolo_result.get('total_detections', 0),
                'difference': abs(
                    gemini_result.get('estimated_count', 0) -
                    yolo_result.get('total_detections', 0)
                )
            },
            'severity_assessment': {
                'gemini': gemini_result.get('severity', 'unknown'),
                'yolo_based': self._yolo_severity(yolo_result)
            },
            'agreement': None
        }

        # Calculate agreement
        count_diff = comparison['lesion_count']['difference']
        gemini_count = comparison['lesion_count']['gemini']

        if gemini_count > 0:
            agreement_pct = 100 * (1 - min(count_diff / gemini_count, 1.0))
            comparison['agreement'] = f"{agreement_pct:.1f}%"

        return comparison

    def _yolo_severity(self, yolo_result: Dict[str, Any]) -> str:
        """Derive severity from YOLO detection counts"""
        total = yolo_result.get('total_detections', 0)

        if total >= 30:
            return 'severe'
        elif total >= 10:
            return 'moderate'
        elif total > 0:
            return 'mild'
        else:
            return 'clear'


def main():
    parser = argparse.ArgumentParser(
        description='Analyze acne in skin images using Gemini Vision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single image
  python scripts/gemini_analysis.py --image test_acne.jpg

  # Detailed analysis
  python scripts/gemini_analysis.py --image test.jpg --detailed

  # Save results to JSON
  python scripts/gemini_analysis.py --image test.jpg --output results.json

  # Analyze folder of images
  python scripts/gemini_analysis.py --folder data/test_images/

Get your free API key at: https://makersuite.google.com/app/apikey
        """
    )

    parser.add_argument(
        '--image',
        type=str,
        help='Path to skin image to analyze'
    )
    parser.add_argument(
        '--folder',
        type=str,
        help='Analyze all images in folder'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Request detailed analysis'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Google API key (or set GEMINI_API_KEY env var)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.folder:
        parser.error("Must specify --image or --folder")

    # Initialize analyzer
    try:
        analyzer = GeminiAcneAnalyzer(api_key=args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Analyze
    results = []

    if args.image:
        # Single image
        result = analyzer.analyze_acne(args.image, detailed=args.detailed)
        results.append({'image': args.image, 'analysis': result})

        # Print results
        print("\n" + "="*70)
        print("GEMINI ACNE ANALYSIS")
        print("="*70 + "\n")
        print(f"Image: {args.image}")
        print(f"Severity: {result.get('severity', 'unknown').upper()}")
        print(f"Estimated lesions: {result.get('estimated_count', 'N/A')}")

        if 'lesion_types' in result:
            print(f"Types detected: {', '.join(result['lesion_types'])}")

        if 'summary' in result:
            print(f"\nSummary:\n{result['summary']}")

        if 'recommendations' in result:
            print("\nRecommendations:")
            for i, rec in enumerate(result['recommendations'], 1):
                print(f"  {i}. {rec}")

    elif args.folder:
        # Batch analysis
        folder_path = Path(args.folder)
        image_files = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))

        logger.info(f"Found {len(image_files)} images in {args.folder}")

        for img_path in image_files:
            try:
                result = analyzer.analyze_acne(str(img_path), detailed=args.detailed)
                results.append({'image': str(img_path), 'analysis': result})
                print(f"✓ {img_path.name}: {result.get('severity', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to analyze {img_path.name}: {e}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Results saved to {output_path}")

    print("\n" + "="*70)
    print(f"✓ Analysis complete - {len(results)} image(s) processed")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
