"""
Generate labels for acne images using Roboflow Ensemble Detector

This script processes all images in your dataset and creates a CSV file
with acne detection features like the one in data/skin_features.csv

Usage:
    python scripts/generate_labels.py --api-key YOUR_ROBOFLOW_API_KEY --input data/raw/acne --output data/labels/acne_labels.csv

    # Or process all datasets:
    python scripts/generate_labels.py --api-key YOUR_ROBOFLOW_API_KEY --all
"""

import argparse
import csv
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ensemble_detector import AcneEnsembleDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_redness_features(image: np.ndarray, detections: list) -> dict:
    """
    Calculate redness metrics for detected acne regions

    Args:
        image: BGR image
        detections: List of Detection objects with bounding boxes

    Returns:
        dict with avg_redness and global_redness
    """
    # Convert BGR to RGB
    if len(image.shape) == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Calculate global redness (entire image)
    r_channel = rgb[:, :, 0].astype(float)
    g_channel = rgb[:, :, 1].astype(float)
    b_channel = rgb[:, :, 2].astype(float)

    # Redness metric: R / (R + G + B)
    total = r_channel + g_channel + b_channel + 1e-6  # Avoid division by zero
    global_redness = float(np.mean(r_channel / total))

    # Calculate average redness for detected regions
    avg_redness = 0.0
    if detections:
        redness_values = []
        height, width = image.shape[:2]

        for det in detections:
            x, y, w, h = det.bbox
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(width, int(x + w))
            y2 = min(height, int(y + h))

            if x2 > x1 and y2 > y1:
                roi = rgb[y1:y2, x1:x2]
                roi_r = roi[:, :, 0].astype(float)
                roi_g = roi[:, :, 1].astype(float)
                roi_b = roi[:, :, 2].astype(float)

                roi_total = roi_r + roi_g + roi_b + 1e-6
                roi_redness = np.mean(roi_r / roi_total)
                redness_values.append(roi_redness)

        if redness_values:
            avg_redness = float(np.mean(redness_values))

    return {
        'avg_redness': avg_redness,
        'global_redness': global_redness
    }


def calculate_detection_features(detections: list) -> dict:
    """
    Calculate geometric features from detections

    Args:
        detections: List of Detection objects

    Returns:
        dict with count, average dimensions, and area
    """
    if not detections:
        return {
            'acne_count': 0,
            'avg_acne_width': 0,
            'avg_acne_height': 0,
            'avg_acne_area': 0
        }

    widths = [det.bbox[2] for det in detections]
    heights = [det.bbox[3] for det in detections]
    areas = [det.bbox[2] * det.bbox[3] for det in detections]

    return {
        'acne_count': len(detections),
        'avg_acne_width': float(np.mean(widths)),
        'avg_acne_height': float(np.mean(heights)),
        'avg_acne_area': float(np.mean(areas))
    }


def count_acne_types(detections: list) -> dict:
    """
    Count each type of acne lesion

    Args:
        detections: List of Detection objects

    Returns:
        dict with counts for each type
    """
    # Normalize class names
    type_mapping = {
        'papule': 'papules',
        'papules': 'papules',
        'pustule': 'pustules',
        'pustules': 'pustules',
        'comedone': 'comedone',
        'comedones': 'comedone',
        'whitehead': 'comedone',
        'blackhead': 'comedone',
        'nodule': 'nodules',
        'nodules': 'nodules',
        'cyst': 'nodules'
    }

    counts = {
        'papules_count': 0,
        'pustules_count': 0,
        'comedone_count': 0,
        'nodules_count': 0
    }

    for det in detections:
        class_lower = det.class_name.lower()
        acne_type = type_mapping.get(class_lower)

        if acne_type:
            counts[f'{acne_type}_count'] += 1

    return counts


def process_image(detector: AcneEnsembleDetector, image_path: Path) -> dict:
    """
    Process a single image and extract all features

    Args:
        detector: Initialized AcneEnsembleDetector
        image_path: Path to image

    Returns:
        dict with all features for CSV row
    """
    logger.info(f"Processing: {image_path.name}")

    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        # Run detection
        result = detector.detect(image)

        # Extract features
        detection_features = calculate_detection_features(result.detections)
        type_counts = count_acne_types(result.detections)
        redness_features = calculate_redness_features(image, result.detections)

        # Combine all features
        row = {
            'filename': image_path.name,
            **detection_features,
            **type_counts,
            **redness_features,
            'skin_disease_label': result.classification_label if result.classification_label else 'NULL',
            'skin_disease_confidence': result.classification_confidence if result.classification_confidence else 'NULL',
            'skin_classification_labels': 'NULL',  # Can be extended
            'acne_detected': 1 if result.count > 0 else 0,
            'result': 'NaN'  # Placeholder for future use
        }

        return row

    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_dataset(
    api_key: str,
    input_dir: Path,
    output_csv: Path,
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
):
    """
    Process all images in a directory and generate labels CSV

    Args:
        api_key: Roboflow API key
        input_dir: Directory containing images
        output_csv: Output CSV file path
        image_extensions: Valid image file extensions
    """
    # Initialize detector
    logger.info("Initializing Roboflow Ensemble Detector...")
    detector = AcneEnsembleDetector(api_key=api_key)

    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    # Process images
    rows = []
    for image_path in tqdm(image_files, desc="Processing images"):
        row = process_image(detector, image_path)
        if row:
            rows.append(row)

    # Write CSV
    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'ID', 'filename', 'acne_count', 'avg_acne_width', 'avg_acne_height',
            'avg_acne_area', 'papules_count', 'pustules_count', 'comedone_count',
            'nodules_count', 'avg_redness', 'global_redness', 'skin_disease_label',
            'skin_disease_confidence', 'skin_classification_labels', 'acne_detected',
            'result'
        ]

        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, row in enumerate(rows, start=1):
                row['ID'] = idx
                writer.writerow(row)

        logger.info(f"✓ Successfully generated labels: {output_csv}")
        logger.info(f"  Total images processed: {len(rows)}")
        logger.info(f"  Images with acne detected: {sum(1 for r in rows if r['acne_detected'] == 1)}")
    else:
        logger.error("No images were successfully processed")


def main():
    parser = argparse.ArgumentParser(
        description='Generate acne detection labels using Roboflow ensemble detector'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='Roboflow API key'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all datasets (acne, rosacea) in data/raw/'
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.all:
        # Process all datasets
        datasets = [
            ('data/raw/acne', 'data/labels/acne_labels.csv'),
            ('data/raw/rosacea', 'data/labels/rosacea_labels.csv')
        ]

        for input_dir, output_csv in datasets:
            input_path = project_root / input_dir
            output_path = project_root / output_csv

            if input_path.exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing dataset: {input_dir}")
                logger.info(f"{'='*60}\n")
                process_dataset(args.api_key, input_path, output_path)
            else:
                logger.warning(f"Directory not found: {input_path}")

    elif args.input and args.output:
        # Process single dataset
        input_path = Path(args.input)
        output_path = Path(args.output)

        if not input_path.exists():
            logger.error(f"Input directory not found: {input_path}")
            sys.exit(1)

        process_dataset(args.api_key, input_path, output_path)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
