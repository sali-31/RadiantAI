"""
Model Comparison and Benchmarking Script

Compare performance of different models (YOLOv8 vs YOLOv10 vs others)
on acne detection task.

Usage:
    # Compare two models
    python scripts/compare_models.py \
        --model1 runs/detect/yolov8/weights/best.pt \
        --model2 runs/detect/yolov10/weights/best.pt \
        --test-dir data/yolo_dataset/images/test

    # Full benchmark
    python scripts/compare_models.py \
        --model1 yolov8m.pt \
        --model2 yolov10m.pt \
        --test-dir data/samples/ \
        --save-report benchmark_report.json
"""

import argparse
import time
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_name: str
    model_size_mb: float
    avg_inference_time_ms: float
    fps: float
    total_detections: int
    avg_confidence: float
    memory_usage_mb: float

    # Accuracy metrics (if ground truth available)
    precision: float = None
    recall: float = None
    mAP50: float = None
    mAP50_95: float = None


class ModelBenchmark:
    """Benchmark and compare models"""

    def __init__(self):
        self.results = {}

    def load_model(self, model_path: str, model_name: str = None):
        """Load model and return inference function"""
        try:
            # Try YOLOv10 first
            from ultralytics import YOLOv10
            model = YOLOv10(model_path)
            logger.info(f"✓ Loaded {model_name or model_path} as YOLOv10")
        except (ImportError, Exception):
            # Fallback to regular YOLO
            from ultralytics import YOLO
            model = YOLO(model_path)
            logger.info(f"✓ Loaded {model_name or model_path} as YOLO")

        return model

    def get_model_size(self, model_path: str) -> float:
        """Get model file size in MB"""
        path = Path(model_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            return round(size_mb, 2)
        return 0.0

    def benchmark_inference_speed(
        self,
        model,
        test_images: List[str],
        warmup_runs: int = 5,
        benchmark_runs: int = 20
    ) -> Tuple[float, float]:
        """
        Benchmark model inference speed

        Returns:
            avg_time_ms: Average inference time in milliseconds
            fps: Frames per second
        """
        logger.info(f"Warming up with {warmup_runs} runs...")

        # Warmup
        for _ in range(warmup_runs):
            img = test_images[0]
            _ = model(img, verbose=False)

        # Benchmark
        logger.info(f"Benchmarking with {benchmark_runs} runs...")
        times = []

        for i in range(benchmark_runs):
            # Use different images if available
            img = test_images[i % len(test_images)]

            start = time.perf_counter()
            _ = model(img, verbose=False)
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        fps = 1000 / avg_time_ms

        logger.info(f"  Avg: {avg_time_ms:.2f}ms (±{std_time_ms:.2f}ms)")
        logger.info(f"  FPS: {fps:.2f}")

        return avg_time_ms, fps

    def count_detections(
        self,
        model,
        test_images: List[str],
        conf_threshold: float = 0.4
    ) -> Tuple[int, float]:
        """
        Count total detections and average confidence

        Returns:
            total_detections: Total number of detections
            avg_confidence: Average confidence score
        """
        total = 0
        confidences = []

        for img_path in test_images:
            results = model(img_path, conf=conf_threshold, verbose=False)

            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    total += len(boxes)
                    confidences.extend(boxes.conf.cpu().numpy().tolist())

        avg_conf = np.mean(confidences) if confidences else 0.0

        return total, avg_conf

    def benchmark_model(
        self,
        model_path: str,
        model_name: str,
        test_images: List[str],
        conf_threshold: float = 0.4
    ) -> ModelMetrics:
        """
        Complete benchmark of a model

        Args:
            model_path: Path to model weights
            model_name: Name for identification
            test_images: List of test image paths
            conf_threshold: Confidence threshold for detections

        Returns:
            ModelMetrics object with all metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {model_name}")
        logger.info(f"{'='*60}")

        # Load model
        model = self.load_model(model_path, model_name)

        # Get model size
        model_size = self.get_model_size(model_path)
        logger.info(f"Model size: {model_size:.2f} MB")

        # Benchmark inference speed
        avg_time, fps = self.benchmark_inference_speed(
            model, test_images, warmup_runs=5, benchmark_runs=20
        )

        # Count detections
        logger.info(f"Counting detections on {len(test_images)} images...")
        total_det, avg_conf = self.count_detections(
            model, test_images, conf_threshold
        )
        logger.info(f"  Total detections: {total_det}")
        logger.info(f"  Avg confidence: {avg_conf:.3f}")

        # Memory usage (approximate)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)

        metrics = ModelMetrics(
            model_name=model_name,
            model_size_mb=model_size,
            avg_inference_time_ms=round(avg_time, 2),
            fps=round(fps, 2),
            total_detections=total_det,
            avg_confidence=round(avg_conf, 3),
            memory_usage_mb=round(memory_mb, 2)
        )

        self.results[model_name] = metrics

        logger.info(f"✓ Benchmark complete for {model_name}")

        return metrics

    def compare_models(self, metrics_list: List[ModelMetrics]) -> Dict:
        """
        Compare multiple models and generate comparison

        Args:
            metrics_list: List of ModelMetrics

        Returns:
            Comparison dictionary
        """
        if len(metrics_list) < 2:
            logger.warning("Need at least 2 models to compare")
            return {}

        comparison = {
            'models': [m.model_name for m in metrics_list],
            'speed_winner': None,
            'size_winner': None,
            'accuracy_winner': None,
            'overall_winner': None,
            'details': {}
        }

        # Find winners
        fastest = min(metrics_list, key=lambda m: m.avg_inference_time_ms)
        smallest = min(metrics_list, key=lambda m: m.model_size_mb)
        most_detections = max(metrics_list, key=lambda m: m.total_detections)

        comparison['speed_winner'] = fastest.model_name
        comparison['size_winner'] = smallest.model_name
        comparison['accuracy_winner'] = most_detections.model_name

        # Calculate speed improvements
        for m in metrics_list:
            if m.model_name != fastest.model_name:
                speedup = (m.avg_inference_time_ms / fastest.avg_inference_time_ms - 1) * 100
                comparison['details'][f'{m.model_name}_vs_{fastest.model_name}_speed'] = \
                    f"{fastest.model_name} is {speedup:.1f}% faster"

        return comparison

    def print_comparison_table(self, metrics_list: List[ModelMetrics]):
        """Print formatted comparison table"""
        print("\n" + "="*100)
        print("MODEL COMPARISON RESULTS")
        print("="*100)

        # Table header
        header = f"{'Model':<20} {'Size (MB)':<12} {'Speed (ms)':<12} {'FPS':<8} {'Detections':<12} {'Confidence':<12}"
        print(header)
        print("-"*100)

        # Table rows
        for m in metrics_list:
            row = f"{m.model_name:<20} {m.model_size_mb:<12.2f} {m.avg_inference_time_ms:<12.2f} {m.fps:<8.2f} {m.total_detections:<12} {m.avg_confidence:<12.3f}"
            print(row)

        print("="*100)

        # Winners
        fastest = min(metrics_list, key=lambda m: m.avg_inference_time_ms)
        smallest = min(metrics_list, key=lambda m: m.model_size_mb)
        most_detections = max(metrics_list, key=lambda m: m.total_detections)

        print("\n🏆 WINNERS:")
        print(f"  🚀 Fastest: {fastest.model_name} ({fastest.avg_inference_time_ms:.2f}ms, {fastest.fps:.2f} FPS)")
        print(f"  📦 Smallest: {smallest.model_name} ({smallest.model_size_mb:.2f} MB)")
        print(f"  🎯 Most Detections: {most_detections.model_name} ({most_detections.total_detections} detections)")

        # Speed comparison
        if len(metrics_list) == 2:
            m1, m2 = metrics_list
            speedup = ((m2.avg_inference_time_ms / m1.avg_inference_time_ms) - 1) * 100
            if speedup > 0:
                print(f"\n⚡ Speed Comparison:")
                print(f"  {m1.model_name} is {abs(speedup):.1f}% faster than {m2.model_name}")
            else:
                print(f"\n⚡ Speed Comparison:")
                print(f"  {m2.model_name} is {abs(speedup):.1f}% faster than {m1.model_name}")

    def plot_comparison(self, metrics_list: List[ModelMetrics], save_path: str = None):
        """Create comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')

        models = [m.model_name for m in metrics_list]

        # 1. Inference Speed
        ax = axes[0, 0]
        speeds = [m.avg_inference_time_ms for m in metrics_list]
        bars = ax.bar(models, speeds, color=['#3498db', '#e74c3c'][:len(models)])
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Inference Speed (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom')

        # 2. FPS
        ax = axes[0, 1]
        fps_values = [m.fps for m in metrics_list]
        bars = ax.bar(models, fps_values, color=['#2ecc71', '#f39c12'][:len(models)])
        ax.set_ylabel('FPS')
        ax.set_title('Throughput (Higher is Better)')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')

        # 3. Model Size
        ax = axes[1, 0]
        sizes = [m.model_size_mb for m in metrics_list]
        bars = ax.bar(models, sizes, color=['#9b59b6', '#1abc9c'][:len(models)])
        ax.set_ylabel('Model Size (MB)')
        ax.set_title('Model Size (Lower is Better)')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}MB',
                   ha='center', va='bottom')

        # 4. Detections
        ax = axes[1, 1]
        detections = [m.total_detections for m in metrics_list]
        bars = ax.bar(models, detections, color=['#e67e22', '#34495e'][:len(models)])
        ax.set_ylabel('Total Detections')
        ax.set_title('Detection Count')
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved comparison plot to {save_path}")
        else:
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved comparison plot to model_comparison.png")

        plt.close()

    def save_report(self, metrics_list: List[ModelMetrics], output_path: str):
        """Save benchmark report as JSON"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models': [asdict(m) for m in metrics_list],
            'comparison': self.compare_models(metrics_list)
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Saved benchmark report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare and benchmark acne detection models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare YOLOv8 vs YOLOv10
  python scripts/compare_models.py \\
    --model1 runs/detect/yolov8/weights/best.pt \\
    --model2 runs/detect/yolov10/weights/best.pt \\
    --test-dir data/samples/

  # Save detailed report
  python scripts/compare_models.py \\
    --model1 yolov8m.pt \\
    --model2 yolov10m.pt \\
    --test-dir data/samples/ \\
    --save-report benchmark.json \\
    --save-plot comparison.png
        """
    )

    parser.add_argument(
        '--model1',
        type=str,
        required=True,
        help='Path to first model (.pt file)'
    )
    parser.add_argument(
        '--model1-name',
        type=str,
        default=None,
        help='Name for first model (default: auto from path)'
    )
    parser.add_argument(
        '--model2',
        type=str,
        required=True,
        help='Path to second model (.pt file)'
    )
    parser.add_argument(
        '--model2-name',
        type=str,
        default=None,
        help='Name for second model (default: auto from path)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Directory with test images'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.4,
        help='Confidence threshold (default: 0.4)'
    )
    parser.add_argument(
        '--save-report',
        type=str,
        default=None,
        help='Save benchmark report to JSON file'
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default=None,
        help='Save comparison plot to file'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.model1).exists():
        logger.error(f"Model 1 not found: {args.model1}")
        return

    if not Path(args.model2).exists():
        logger.error(f"Model 2 not found: {args.model2}")
        return

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {args.test_dir}")
        return

    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(test_dir.glob(ext)))

    if not test_images:
        logger.error(f"No images found in {args.test_dir}")
        return

    test_images = [str(p) for p in test_images]
    logger.info(f"Found {len(test_images)} test images")

    # Auto-generate model names if not provided
    model1_name = args.model1_name or Path(args.model1).stem
    model2_name = args.model2_name or Path(args.model2).stem

    # Create benchmark
    benchmark = ModelBenchmark()

    # Benchmark both models
    metrics1 = benchmark.benchmark_model(
        args.model1, model1_name, test_images, args.conf
    )

    metrics2 = benchmark.benchmark_model(
        args.model2, model2_name, test_images, args.conf
    )

    # Print comparison
    benchmark.print_comparison_table([metrics1, metrics2])

    # Plot comparison
    benchmark.plot_comparison(
        [metrics1, metrics2],
        save_path=args.save_plot
    )

    # Save report
    if args.save_report:
        benchmark.save_report([metrics1, metrics2], args.save_report)

    logger.info("\n✅ Benchmark complete!")


if __name__ == '__main__':
    main()
