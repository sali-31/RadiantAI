"""
Prepare dataset for YOLOv8 training

This script:
1. Creates YOLO dataset structure (images/ and labels/ folders)
2. Splits data into train/val/test sets
3. Uses pre-trained YOLOv8 to generate initial labels (pseudo-labeling)
4. Organizes everything for YOLOv8 training

Usage:
    # Generate labels using pre-trained YOLO
    python scripts/prepare_yolo_dataset.py --source data/raw/acne --method pretrained

    # Or if you have existing annotations (CSV/JSON)
    python scripts/prepare_yolo_dataset.py --source data/raw/acne --method from_csv --csv data/skin_features.csv
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yolo_structure(output_dir: Path):
    """Create YOLO dataset directory structure"""
    splits = ['train', 'val', 'test']

    for split in splits:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    logger.info(f"✓ Created YOLO dataset structure in {output_dir}")


def split_dataset(
    image_files: List[Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split dataset into train/val/test sets"""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    random.shuffle(image_files)

    n = len(image_files)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    logger.info(f"Dataset split:")
    logger.info(f"  Train: {len(train_files)} images ({train_ratio*100:.1f}%)")
    logger.info(f"  Val:   {len(val_files)} images ({val_ratio*100:.1f}%)")
    logger.info(f"  Test:  {len(test_files)} images ({test_ratio*100:.1f}%)")

    return train_files, val_files, test_files


def generate_labels_with_pretrained_yolo(
    image_path: Path,
    model,
    conf_threshold: float = 0.25
) -> List[str]:
    """
    Generate YOLO format labels using pre-trained model

    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized to [0, 1]

    Args:
        image_path: Path to image
        model: Pre-trained YOLO model
        conf_threshold: Confidence threshold for detections

    Returns:
        List of label strings in YOLO format
    """
    # Run inference
    results = model(str(image_path), conf=conf_threshold, verbose=False)

    labels = []

    if len(results) > 0:
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            # Get image dimensions
            img = cv2.imread(str(image_path))
            img_h, img_w = img.shape[:2]

            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                # Get class ID
                class_id = int(box.cls[0].cpu().numpy())

                # YOLO format: class x_center y_center width height
                label = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                labels.append(label)

    return labels


def copy_and_label_dataset(
    source_dir: Path,
    output_dir: Path,
    train_files: List[Path],
    val_files: List[Path],
    test_files: List[Path],
    labeling_method: str = 'pretrained',
    model=None
):
    """
    Copy images and generate/copy labels to YOLO structure

    Args:
        source_dir: Source directory with images
        output_dir: Output YOLO dataset directory
        train_files, val_files, test_files: Split file lists
        labeling_method: 'pretrained' or 'empty'
        model: Pre-trained YOLO model (if method='pretrained')
    """

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        logger.info(f"\nProcessing {split_name} set...")

        for img_path in tqdm(files, desc=f"  {split_name}"):
            # Copy image
            dest_img = output_dir / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dest_img)

            # Generate label file
            label_path = output_dir / 'labels' / split_name / f"{img_path.stem}.txt"

            if labeling_method == 'pretrained' and model is not None:
                # Generate labels using pre-trained model
                labels = generate_labels_with_pretrained_yolo(img_path, model)

                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))

            elif labeling_method == 'empty':
                # Create empty label file (for manual annotation later)
                label_path.touch()

    logger.info(f"\n✓ Dataset prepared successfully in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare acne dataset for YOLOv8 training'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Source directory containing images (e.g., data/raw/acne)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/yolo_dataset',
        help='Output directory for YOLO dataset'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['pretrained', 'empty'],
        default='pretrained',
        help='Labeling method: pretrained (use YOLOv8 pre-trained model) or empty (create empty labels for manual annotation)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.25,
        help='Confidence threshold for pre-trained model (default: 0.25)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        choices=['n', 's', 'm', 'l', 'x'],
        default='n',
        help='YOLOv8 model size for labeling (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return

    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(source_dir.glob(f'*{ext}'))

    if not image_files:
        logger.error(f"No images found in {source_dir}")
        return

    logger.info(f"Found {len(image_files)} images in {source_dir}")

    # Create YOLO structure
    create_yolo_structure(output_dir)

    # Split dataset
    train_files, val_files, test_files = split_dataset(
        image_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    # Load pre-trained model if needed
    model = None
    if args.method == 'pretrained':
        try:
            from ultralytics import YOLO

            model_name = f"yolov8{args.model_size}.pt"
            logger.info(f"\nLoading pre-trained YOLOv8 model: {model_name}")
            logger.info("(This will download the model on first use)")

            model = YOLO(model_name)
            logger.info("✓ Model loaded successfully")

            logger.info("\nGenerating pseudo-labels using pre-trained YOLOv8...")
            logger.info("Note: These are initial labels from a general object detector.")
            logger.info("You should review and refine them for acne-specific detection.")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Falling back to empty labels...")
            args.method = 'empty'

    # Copy images and generate labels
    copy_and_label_dataset(
        source_dir,
        output_dir,
        train_files,
        val_files,
        test_files,
        labeling_method=args.method,
        model=model
    )

    # Print summary
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"\nNext steps:")

    if args.method == 'pretrained':
        print("\n1. Review the generated labels (they may need refinement)")
        print("   - Use a tool like LabelImg or Roboflow to review/edit")
        print("   - Labels are in: data/yolo_dataset/labels/")
        print("\n2. Train YOLOv8 on this dataset:")
        print("   python scripts/train_yolo.py")
    else:
        print("\n1. Annotate your images using:")
        print("   - LabelImg: https://github.com/heartexlabs/labelImg")
        print("   - Roboflow: https://roboflow.com")
        print("   - CVAT: https://cvat.org")
        print("\n2. After annotation, train YOLOv8:")
        print("   python scripts/train_yolo.py")

    print("\n3. Use the trained model for inference:")
    print("   python scripts/yolo_inference.py --image path/to/image.jpg")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
