"""
YOLOv8 Inference for Acne Detection

Use trained YOLOv8 model to:
1. Detect acne in new images
2. Generate labels for unlabeled dataset
3. Create visualizations with bounding boxes

Usage:
    # Detect acne in a single image
    python scripts/yolo_inference.py --model runs/detect/acne_detector/weights/best.pt --source test_image.jpg

    # Generate labels for entire dataset
    python scripts/yolo_inference.py --model runs/detect/acne_detector/weights/best.pt --source data/raw/acne --save-labels

    # Batch inference with visualizations
    python scripts/yolo_inference.py --model runs/detect/acne_detector/weights/best.pt --source data/raw/acne --save-vis
"""

import argparse
from pathlib import Path
import cv2
import csv
from typing import List, Dict
from tqdm import tqdm
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_acne(
    model: YOLO,
    image_path: Path,
    conf_threshold: float = 0.25
) -> Dict:
    """
    Detect acne in a single image

    Args:
        model: Trained YOLO model
        image_path: Path to image
        conf_threshold: Confidence threshold

    Returns:
        Dictionary with detection results
    """
    results = model(str(image_path), conf=conf_threshold, verbose=False)

    detections = []
    if len(results) > 0:
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': conf,
                    'class_id': cls,
                    'class_name': model.names[cls]
                })

    return {
        'image': image_path.name,
        'detections': detections,
        'count': len(detections)
    }


def calculate_features(result: Dict, image_path: Path) -> Dict:
    """
    Calculate features similar to skin_features.csv

    Args:
        result: Detection result dictionary
        image_path: Path to image

    Returns:
        Dictionary with calculated features
    """
    detections = result['detections']

    if len(detections) == 0:
        return {
            'filename': result['image'],
            'acne_count': 0,
            'avg_acne_width': 0,
            'avg_acne_height': 0,
            'avg_acne_area': 0,
            'papules_count': 0,
            'pustules_count': 0,
            'comedone_count': 0,
            'nodules_count': 0,
            'acne_detected': 0
        }

    # Calculate geometric features
    widths = []
    heights = []
    areas = []

    type_counts = {
        'comedone': 0,
        'papule': 0,
        'pustule': 0,
        'nodule': 0
    }

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = x2 - x1
        h = y2 - y1
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

        # Count by type
        class_name = det['class_name'].lower()
        if class_name in type_counts:
            type_counts[class_name] += 1

    return {
        'filename': result['image'],
        'acne_count': len(detections),
        'avg_acne_width': sum(widths) / len(widths),
        'avg_acne_height': sum(heights) / len(heights),
        'avg_acne_area': sum(areas) / len(areas),
        'papules_count': type_counts['papule'],
        'pustules_count': type_counts['pustule'],
        'comedone_count': type_counts['comedone'],
        'nodules_count': type_counts['nodule'],
        'acne_detected': 1
    }


def save_yolo_labels(result: Dict, output_dir: Path, img_width: int, img_height: int):
    """
    Save detections in YOLO format

    Args:
        result: Detection result
        output_dir: Output directory for labels
        img_width: Image width
        img_height: Image height
    """
    label_path = output_dir / f"{Path(result['image']).stem}.txt"

    with open(label_path, 'w') as f:
        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']

            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            class_id = det['class_id']

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def save_visualization(result: Dict, image_path: Path, output_dir: Path, model: YOLO):
    """
    Save image with bounding boxes drawn

    Args:
        result: Detection result
        image_path: Path to input image
        output_dir: Output directory
        model: YOLO model (for class names and colors)
    """
    img = cv2.imread(str(image_path))

    if img is None:
        logger.warning(f"Could not load image: {image_path}")
        return

    # Draw bounding boxes
    for det in result['detections']:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_name = det['class_name']

        # Color by class
        colors = {
            'comedone': (0, 255, 0),      # Green
            'papule': (0, 165, 255),      # Orange
            'pustule': (0, 0, 255),       # Red
            'nodule': (255, 0, 255)       # Magenta
        }
        color = colors.get(class_name.lower(), (255, 255, 255))

        # Draw box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (int(x1), int(y1) - label_h - 10), (int(x1) + label_w, int(y1)), color, -1)
        cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    output_path = output_dir / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(output_path), img)


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 inference for acne detection')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained YOLOv8 model (.pt file)'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Image file or directory containing images'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--save-labels',
        action='store_true',
        help='Save labels in YOLO format'
    )
    parser.add_argument(
        '--save-csv',
        type=str,
        help='Save results to CSV file (e.g., data/labels/predictions.csv)'
    )
    parser.add_argument(
        '--save-vis',
        action='store_true',
        help='Save visualizations with bounding boxes'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='runs/inference',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Get class names
    logger.info(f"Classes: {model.names}")

    # Find images
    source_path = Path(args.source)

    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(source_path.glob(f'*{ext}'))
            image_files.extend(source_path.glob(f'*{ext.upper()}'))
    else:
        logger.error(f"Source not found: {source_path}")
        return

    if not image_files:
        logger.error(f"No images found in {source_path}")
        return

    logger.info(f"Found {len(image_files)} images to process")

    # Create output directories
    output_dir = Path(args.output)
    if args.save_labels:
        labels_dir = output_dir / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)

    if args.save_vis:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    all_results = []
    all_features = []

    for img_path in tqdm(image_files, desc="Processing images"):
        # Detect
        result = detect_acne(model, img_path, conf_threshold=args.conf)
        all_results.append(result)

        # Calculate features
        features = calculate_features(result, img_path)
        all_features.append(features)

        # Save labels
        if args.save_labels:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                save_yolo_labels(result, labels_dir, w, h)

        # Save visualization
        if args.save_vis:
            save_visualization(result, img_path, vis_dir, model)

    # Save CSV
    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'ID', 'filename', 'acne_count', 'avg_acne_width', 'avg_acne_height',
            'avg_acne_area', 'papules_count', 'pustules_count', 'comedone_count',
            'nodules_count', 'acne_detected'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx, features in enumerate(all_features, start=1):
                features['ID'] = idx
                writer.writerow(features)

        logger.info(f"✓ Saved CSV to: {csv_path}")

    # Print summary
    total_detections = sum(r['count'] for r in all_results)
    images_with_acne = sum(1 for r in all_results if r['count'] > 0)

    logger.info(f"\n{'='*60}")
    logger.info("Inference Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"\nSummary:")
    logger.info(f"  Total images: {len(image_files)}")
    logger.info(f"  Images with acne: {images_with_acne} ({images_with_acne/len(image_files)*100:.1f}%)")
    logger.info(f"  Total detections: {total_detections}")
    logger.info(f"  Avg detections per image: {total_detections/len(image_files):.2f}")

    if args.save_labels:
        logger.info(f"\n✓ Labels saved to: {labels_dir}")

    if args.save_vis:
        logger.info(f"✓ Visualizations saved to: {vis_dir}")

    logger.info(f"\n{'='*60}")


if __name__ == '__main__':
    main()
