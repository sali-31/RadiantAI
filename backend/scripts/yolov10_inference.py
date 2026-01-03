"""
YOLOv10 Inference Script for Acne Detection

Run trained YOLOv10 model on images or videos for acne detection.

Usage:
    # Single image
    python scripts/yolov10_inference.py --model runs/detect/acne_yolov10/weights/best.pt --source image.jpg

    # Multiple images
    python scripts/yolov10_inference.py --model best.pt --source data/samples/

    # Video
    python scripts/yolov10_inference.py --model best.pt --source video.mp4

    # Webcam
    python scripts/yolov10_inference.py --model best.pt --source 0

    # With custom confidence threshold
    python scripts/yolov10_inference.py --model best.pt --source image.jpg --conf 0.5
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, List, Tuple
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AcneYOLOv10Detector:
    """YOLOv10 Acne Detector with severity assessment"""

    # Acne severity thresholds
    SEVERITY_THRESHOLDS = {
        'mild': 20,      # <= 20 lesions
        'moderate': 50,  # 21-50 lesions
        'severe': float('inf')  # > 50 lesions
    }

    # Acne type severity mapping
    LESION_SEVERITY = {
        'comedone': 1,    # Mild (blackheads, whiteheads)
        'papule': 2,      # Moderate (red bumps)
        'pustule': 2,     # Moderate (pus-filled)
        'nodule': 3       # Severe (deep, painful)
    }

    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        """
        Initialize YOLOv10 detector

        Args:
            model_path: Path to trained YOLOv10 model
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLOv10 model"""
        try:
            from ultralytics import YOLOv10
            logger.info(f"Loading YOLOv10 model from {self.model_path}")
            self.model = YOLOv10(self.model_path)
            logger.info("✓ Model loaded successfully")
        except ImportError:
            logger.warning("YOLOv10 not available, falling back to YOLOv8")
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            logger.info("✓ Model loaded successfully (using YOLOv8 API)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, source: str, save: bool = True, save_txt: bool = False,
               save_conf: bool = True, save_crop: bool = False,
               show: bool = False, visualize: bool = False) -> Dict:
        """
        Run detection on image/video/directory

        Args:
            source: Path to image, video, directory, or webcam index
            save: Save annotated images
            save_txt: Save detection results as text
            save_conf: Include confidence in text files
            save_crop: Save cropped detection images
            show: Display results
            visualize: Visualize feature maps

        Returns:
            Dictionary with detection results and analysis
        """
        logger.info(f"Running detection on: {source}")

        results = self.model.predict(
            source=source,
            conf=self.conf_threshold,
            save=save,
            save_txt=save_txt,
            save_conf=save_conf,
            save_crop=save_crop,
            show=show,
            visualize=visualize,
            stream=False,
            verbose=True
        )

        # Process results
        analysis = self._analyze_results(results)

        return analysis

    def _analyze_results(self, results) -> Dict:
        """
        Analyze detection results and assess severity

        Args:
            results: YOLO detection results

        Returns:
            Dictionary with analysis
        """
        all_detections = []
        total_count = 0
        type_counts = {}

        for idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                logger.info(f"Image {idx + 1}: No acne detected")
                continue

            image_detections = []
            for box in boxes:
                # Get box details
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = result.names[cls]

                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class': class_name,
                    'class_id': cls
                }
                image_detections.append(detection)

                # Count by type
                type_counts[class_name] = type_counts.get(class_name, 0) + 1
                total_count += 1

            all_detections.append({
                'image_index': idx,
                'detections': image_detections,
                'count': len(image_detections)
            })

            logger.info(f"Image {idx + 1}: {len(image_detections)} lesions detected")

        # Assess severity
        severity = self._assess_severity(total_count, type_counts)

        analysis = {
            'total_images': len(results),
            'total_lesions': total_count,
            'lesion_types': type_counts,
            'severity': severity,
            'detections': all_detections
        }

        return analysis

    def _assess_severity(self, total_count: int, type_counts: Dict) -> Dict:
        """
        Assess acne severity based on count and types

        Args:
            total_count: Total number of lesions
            type_counts: Dictionary of lesion type counts

        Returns:
            Severity assessment dictionary
        """
        # Determine severity level based on count
        if total_count == 0:
            level = 'clear'
        elif total_count <= self.SEVERITY_THRESHOLDS['mild']:
            level = 'mild'
        elif total_count <= self.SEVERITY_THRESHOLDS['moderate']:
            level = 'moderate'
        else:
            level = 'severe'

        # Check for severe lesion types
        has_nodules = 'nodule' in type_counts and type_counts['nodule'] > 0
        if has_nodules and level != 'severe':
            level = 'moderate'  # Upgrade severity if nodules present

        # Calculate severity score (weighted by lesion type)
        severity_score = 0
        for lesion_type, count in type_counts.items():
            weight = self.LESION_SEVERITY.get(lesion_type.lower(), 1)
            severity_score += count * weight

        return {
            'level': level,
            'score': severity_score,
            'total_count': total_count,
            'has_severe_lesions': has_nodules,
            'breakdown': type_counts
        }

    def print_summary(self, analysis: Dict):
        """Print detection summary"""
        print("\n" + "="*70)
        print("🔍 ACNE DETECTION RESULTS")
        print("="*70)

        print(f"\n📊 Summary:")
        print(f"  Total Images: {analysis['total_images']}")
        print(f"  Total Lesions: {analysis['total_lesions']}")

        if analysis['total_lesions'] > 0:
            print(f"\n📈 Lesion Breakdown:")
            for lesion_type, count in analysis['lesion_types'].items():
                percentage = (count / analysis['total_lesions']) * 100
                print(f"  {lesion_type.capitalize()}: {count} ({percentage:.1f}%)")

            severity = analysis['severity']
            print(f"\n⚕️  Severity Assessment:")
            print(f"  Level: {severity['level'].upper()}")
            print(f"  Score: {severity['score']}")

            # Severity interpretation
            level_emojis = {
                'clear': '✅',
                'mild': '🟢',
                'moderate': '🟡',
                'severe': '🔴'
            }
            emoji = level_emojis.get(severity['level'], '❓')
            print(f"  {emoji} {self._get_severity_message(severity['level'])}")

            if severity['has_severe_lesions']:
                print(f"  ⚠️  Contains severe lesions (nodules/cysts)")

        else:
            print("\n✅ No acne detected - Clear skin!")

        print("\n" + "="*70)

    def _get_severity_message(self, level: str) -> str:
        """Get human-readable severity message"""
        messages = {
            'clear': 'No acne detected',
            'mild': 'Mild acne - Over-the-counter treatments may help',
            'moderate': 'Moderate acne - Consider consulting a dermatologist',
            'severe': 'Severe acne - Dermatologist consultation recommended'
        }
        return messages.get(level, 'Unknown severity')

    def save_results(self, analysis: Dict, output_path: str):
        """Save results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv10 Acne Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect acne in a single image
  python scripts/yolov10_inference.py --model best.pt --source image.jpg

  # Process all images in a directory
  python scripts/yolov10_inference.py --model best.pt --source data/samples/

  # Save detection results as JSON
  python scripts/yolov10_inference.py --model best.pt --source image.jpg --save-json results.json

  # Show results in real-time
  python scripts/yolov10_inference.py --model best.pt --source image.jpg --show

  # Use higher confidence threshold
  python scripts/yolov10_inference.py --model best.pt --source image.jpg --conf 0.6
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLOv10 model (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to image, video, directory, or webcam index (0)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.4,
        help='Confidence threshold (default: 0.4)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='Save annotated images (default: True)'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='Save detection results as text files'
    )
    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='Save cropped detection images'
    )
    parser.add_argument(
        '--save-json',
        type=str,
        default=None,
        help='Save analysis results to JSON file'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results in real-time'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize model feature maps'
    )

    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        logger.info("\nPlease train a model first:")
        logger.info("  python scripts/train_yolov10.py")
        return

    # Check if source exists (if not webcam)
    if not args.source.isdigit():
        if not Path(args.source).exists():
            logger.error(f"Source not found: {args.source}")
            return

    # Initialize detector
    detector = AcneYOLOv10Detector(
        model_path=args.model,
        conf_threshold=args.conf
    )

    # Run detection
    analysis = detector.detect(
        source=args.source,
        save=args.save,
        save_txt=args.save_txt,
        save_crop=args.save_crop,
        show=args.show,
        visualize=args.visualize
    )

    # Print summary
    detector.print_summary(analysis)

    # Save JSON results if requested
    if args.save_json:
        detector.save_results(analysis, args.save_json)

    logger.info("\n✅ Detection complete!")


if __name__ == '__main__':
    main()
