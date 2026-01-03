"""
Train YOLOv8 model on acne dataset

This script trains a YOLOv8 model for acne detection using your prepared dataset.

Usage:
    # Basic training with default settings
    python scripts/train_yolo.py

    # Custom training
    python scripts/train_yolo.py --model yolov8s.pt --epochs 200 --batch 32

    # Resume from checkpoint
    python scripts/train_yolo.py --resume runs/detect/train/weights/last.pt
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_yolo(
    model_name: str = 'yolov8n.pt',
    data_config: str = 'config/yolo_acne.yaml',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    device: str = None,
    project: str = 'runs/detect',
    name: str = 'acne_detector',
    resume: str = None,
    **kwargs
):
    """
    Train YOLOv8 model

    Args:
        model_name: Pre-trained model to start from (yolov8n/s/m/l/x.pt)
        data_config: Path to dataset YAML config
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        patience: Early stopping patience
        device: Device to train on (None=auto, 'cpu', '0', '0,1', etc.)
        project: Project directory
        name: Experiment name
        resume: Resume from checkpoint path
        **kwargs: Additional YOLO training arguments
    """

    logger.info("="*60)
    logger.info("YOLOv8 Acne Detection Training")
    logger.info("="*60)

    # Load data config to get class names
    data_path = Path(data_config)
    if data_path.exists():
        with open(data_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
            logger.info(f"\nDataset: {data_cfg.get('path', 'N/A')}")
            logger.info(f"Classes: {data_cfg.get('names', 'N/A')}")
            logger.info(f"Number of classes: {data_cfg.get('nc', 'N/A')}")
    else:
        logger.error(f"Data config not found: {data_config}")
        return

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if device is None:
        device = '0' if cuda_available else 'cpu'

    logger.info(f"\nDevice: {device}")
    if cuda_available:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("⚠️  Training on CPU (will be slower)")

    # Load model
    logger.info(f"\nLoading model: {model_name}")

    if resume:
        logger.info(f"Resuming from: {resume}")
        model = YOLO(resume)
    else:
        model = YOLO(model_name)

    # Training parameters
    logger.info(f"\nTraining Parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Patience: {patience}")

    # Start training
    logger.info(f"\n{'='*60}")
    logger.info("Starting training...")
    logger.info(f"{'='*60}\n")

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
        optimizer='auto',
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        close_mosaic=10,
        resume=resume is not None,
        amp=True,  # Automatic Mixed Precision
        fraction=1.0,
        profile=False,
        freeze=None,
        **kwargs
    )

    # Print results
    logger.info(f"\n{'='*60}")
    logger.info("Training Complete!")
    logger.info(f"{'='*60}")

    # Get the best model path
    best_model_path = Path(project) / name / 'weights' / 'best.pt'
    last_model_path = Path(project) / name / 'weights' / 'last.pt'

    if best_model_path.exists():
        logger.info(f"\n✓ Best model saved to: {best_model_path}")
        logger.info(f"✓ Last model saved to: {last_model_path}")

        # Validate the best model
        logger.info("\nValidating best model...")
        model = YOLO(str(best_model_path))
        metrics = model.val(data=str(data_path))

        logger.info(f"\nValidation Metrics:")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall: {metrics.box.mr:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info("Next Steps:")
    logger.info(f"{'='*60}")
    logger.info(f"\n1. Test the model:")
    logger.info(f"   python scripts/yolo_inference.py --model {best_model_path} --source path/to/image.jpg")
    logger.info(f"\n2. Export the model:")
    logger.info(f"   python scripts/export_yolo.py --model {best_model_path}")
    logger.info(f"\n3. View training results:")
    logger.info(f"   tensorboard --logdir {Path(project) / name}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for acne detection')

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLOv8 model size (n=fastest, x=most accurate)'
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
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (reduce if out of memory)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (epochs without improvement)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (None=auto, cpu, 0, 0,1, etc.)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='runs/detect',
        help='Project directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='acne_detector',
        help='Experiment name'
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
        help='Number of dataloader workers'
    )

    args = parser.parse_args()

    # Train
    train_yolo(
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
        workers=args.workers
    )


if __name__ == '__main__':
    main()
