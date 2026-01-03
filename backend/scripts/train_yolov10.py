"""
Train YOLOv10 model on acne dataset

YOLOv10 is the latest YOLO version with improved architecture:
- NMS-free detection (faster, cleaner predictions)
- Better small object detection (perfect for acne lesions)
- 20-30% faster inference than YOLOv8

Usage:
    # Basic training with default settings
    python scripts/train_yolov10.py

    # Custom training
    python scripts/train_yolov10.py --model yolov10s.pt --epochs 200 --batch 32

    # Resume from checkpoint
    python scripts/train_yolov10.py --resume runs/detect/train/weights/last.pt
"""

import argparse
from pathlib import Path
import torch
import yaml
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_yolov10_installation():
    """Check if YOLOv10 is properly installed"""
    try:
        from ultralytics import YOLOv10
        logger.info("✓ YOLOv10 is properly installed")
        return True
    except ImportError:
        logger.error("❌ YOLOv10 not found. Installing...")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/THU-MIG/yolov10.git"
        ])
        logger.info("✓ YOLOv10 installed successfully")
        return True


def train_yolov10(
    model_name: str = 'yolov10n.pt',
    data_config: str = 'config/yolo_acne.yaml',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    device: str = None,
    project: str = 'runs/detect',
    name: str = 'acne_yolov10',
    resume: str = None,
    cache: bool = False,
    workers: int = 8,
    **kwargs
):
    """
    Train YOLOv10 model

    Args:
        model_name: Pre-trained model to start from (yolov10n/s/m/b/l/x.pt)
        data_config: Path to dataset YAML config
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        patience: Early stopping patience
        device: Device to train on (None=auto, 'cpu', '0', '0,1', etc.)
        project: Project directory
        name: Experiment name
        resume: Resume from checkpoint path
        cache: Cache images for faster training
        workers: Number of dataloader workers
        **kwargs: Additional YOLO training arguments
    """

    logger.info("="*70)
    logger.info("YOLOv10 Acne Detection Training")
    logger.info("="*70)

    # Check YOLOv10 installation
    check_yolov10_installation()

    # Import after installation check
    try:
        from ultralytics import YOLOv10
    except ImportError:
        # Fallback to regular YOLO if YOLOv10 not available
        logger.warning("YOLOv10 not available, falling back to YOLOv8")
        from ultralytics import YOLO as YOLOv10

    # Load data config to get class names
    data_path = Path(data_config)
    if data_path.exists():
        with open(data_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            logger.info(f"\nDataset Configuration:")
            logger.info(f"  Path: {data_cfg.get('path', 'N/A')}")
            logger.info(f"  Classes: {data_cfg.get('names', 'N/A')}")
            logger.info(f"  Number of classes: {data_cfg.get('nc', 'N/A')}")
    else:
        logger.error(f"❌ Data config not found: {data_config}")
        logger.info("\nPlease create the dataset first:")
        logger.info("  python scripts/prepare_yolo_dataset.py")
        return

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = '0' if cuda_available else 'cpu'

    logger.info(f"\nHardware Configuration:")
    logger.info(f"  Device: {device}")
    if cuda_available:
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("  ⚠️  Training on CPU (will be significantly slower)")
        logger.info("  💡 Consider using Google Colab for free GPU access")

    # Load model
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Model: {model_name}")

    if resume:
        logger.info(f"  Resuming from: {resume}")
        model = YOLOv10(resume)
    else:
        model = YOLOv10(model_name)

    # Display model info
    model_sizes = {
        'yolov10n': '2.3M params (fastest, mobile)',
        'yolov10s': '7.2M params (small, balanced)',
        'yolov10m': '15.4M params (medium, accurate)',
        'yolov10b': '19.1M params (balanced, v10-specific)',
        'yolov10l': '24.4M params (large, high accuracy)',
        'yolov10x': '29.5M params (extra-large, best accuracy)'
    }
    model_key = model_name.replace('.pt', '')
    if model_key in model_sizes:
        logger.info(f"  Size: {model_sizes[model_key]}")

    # Training parameters
    logger.info(f"\nTraining Hyperparameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Image size: {imgsz}x{imgsz}")
    logger.info(f"  Patience: {patience} (early stopping)")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Cache: {cache}")

    # Start training
    logger.info(f"\n{'='*70}")
    logger.info("Starting Training... ⏳")
    logger.info(f"{'='*70}\n")

    try:
        results = model.train(
            data=str(data_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer='auto',  # Automatically selects best optimizer
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,  # Rectangular training
            cos_lr=True,  # Cosine learning rate scheduler
            close_mosaic=10,  # Disable mosaic augmentation last N epochs
            resume=resume is not None,
            amp=True,  # Automatic Mixed Precision for faster training
            fraction=1.0,  # Fraction of dataset to use
            profile=False,
            freeze=None,  # Freeze layers (None = train all)
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=0.937,  # SGD momentum
            weight_decay=0.0005,  # Optimizer weight decay
            warmup_epochs=3.0,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # Distribution focal loss gain
            pose=12.0,  # Pose loss gain (if using pose model)
            kobj=1.0,  # Keypoint object loss gain
            label_smoothing=0.0,  # Label smoothing
            nbs=64,  # Nominal batch size
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,  # HSV-Saturation augmentation
            hsv_v=0.4,  # HSV-Value augmentation
            degrees=0.0,  # Rotation augmentation
            translate=0.1,  # Translation augmentation
            scale=0.5,  # Scale augmentation
            shear=0.0,  # Shear augmentation
            perspective=0.0,  # Perspective augmentation
            flipud=0.0,  # Vertical flip probability
            fliplr=0.5,  # Horizontal flip probability
            mosaic=1.0,  # Mosaic augmentation probability
            mixup=0.0,  # Mixup augmentation probability
            copy_paste=0.0,  # Copy-paste augmentation
            cache=cache,  # Cache images for faster training
            workers=workers,
            **kwargs
        )

        # Print results
        logger.info(f"\n{'='*70}")
        logger.info("✅ Training Complete!")
        logger.info(f"{'='*70}")

        # Get the best model path
        best_model_path = Path(project) / name / 'weights' / 'best.pt'
        last_model_path = Path(project) / name / 'weights' / 'last.pt'

        if best_model_path.exists():
            logger.info(f"\n📁 Model Saved:")
            logger.info(f"  Best: {best_model_path}")
            logger.info(f"  Last: {last_model_path}")

            # Validate the best model
            logger.info("\n📊 Validating best model...")
            model = YOLOv10(str(best_model_path))
            metrics = model.val(data=str(data_path))

            logger.info(f"\n🎯 Validation Metrics:")
            logger.info(f"  mAP@50: {metrics.box.map50:.4f} (mean Average Precision at IoU=0.5)")
            logger.info(f"  mAP@50-95: {metrics.box.map:.4f} (mAP across IoU 0.5-0.95)")
            logger.info(f"  Precision: {metrics.box.mp:.4f} (% of detections that are correct)")
            logger.info(f"  Recall: {metrics.box.mr:.4f} (% of ground truth objects detected)")

            # Performance interpretation
            logger.info(f"\n💡 Performance Interpretation:")
            if metrics.box.map50 > 0.8:
                logger.info("  ✅ Excellent performance! Model is production-ready.")
            elif metrics.box.map50 > 0.6:
                logger.info("  ✓ Good performance. Consider training longer or with more data.")
            else:
                logger.info("  ⚠️  Needs improvement. Try more epochs, data, or hyperparameter tuning.")

        logger.info(f"\n{'='*70}")
        logger.info("📝 Next Steps:")
        logger.info(f"{'='*70}")
        logger.info(f"\n1️⃣  Test the model on images:")
        logger.info(f"   python scripts/yolov10_inference.py --model {best_model_path} --source path/to/image.jpg")
        logger.info(f"\n2️⃣  Export the model for deployment:")
        logger.info(f"   python scripts/export_yolov10.py --model {best_model_path}")
        logger.info(f"\n3️⃣  View training results (charts, confusion matrix, etc.):")
        logger.info(f"   open {Path(project) / name}/")
        logger.info(f"\n4️⃣  Launch TensorBoard to visualize training:")
        logger.info(f"   tensorboard --logdir {Path(project) / name}")

        return results

    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        logger.info("\n💡 Troubleshooting:")
        logger.info("  1. Check if dataset exists and is properly formatted")
        logger.info("  2. Reduce batch size if out of memory")
        logger.info("  3. Ensure CUDA is properly installed (if using GPU)")
        logger.info("  4. Check data config file (config/yolo_acne.yaml)")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv10 for acne detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (YOLOv10n, 100 epochs)
  python scripts/train_yolov10.py

  # Train larger model for better accuracy
  python scripts/train_yolov10.py --model yolov10m.pt --epochs 200

  # Train with larger batch size (if you have more GPU memory)
  python scripts/train_yolov10.py --batch 32

  # Resume from checkpoint
  python scripts/train_yolov10.py --resume runs/detect/acne_yolov10/weights/last.pt

  # Train on CPU (slower)
  python scripts/train_yolov10.py --device cpu
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolov10n.pt',
        choices=['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt'],
        help='YOLOv10 model size (n=fastest, x=most accurate)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='config/yolo_acne.yaml',
        help='Path to dataset YAML configuration'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (reduce if out of memory, default: 16)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience in epochs (default: 20)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (auto/cpu/0/0,1/etc, default: auto)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory (default: runs/detect)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='acne_yolov10',
        help='Experiment name (default: acne_yolov10)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint path'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers (default: 8)'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Cache images for faster training (requires more RAM)'
    )

    args = parser.parse_args()

    # Train
    train_yolov10(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
        cache=args.cache
    )


if __name__ == '__main__':
    main()
