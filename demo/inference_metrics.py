# Copyright (c) OpenMMLab. All rights reserved.
import time
import psutil
try:
    import GPUtil
    GPU_UTIL_AVAILABLE = True
except ImportError:
    GPU_UTIL_AVAILABLE = False
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Union

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot


class InferenceMetrics:
    """Class to measure and track inference performance metrics."""

    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.start_time = None
        self.end_time = None

    def start_measurement(self):
        """Start measuring inference time."""
        self.start_time = time.time()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    def end_measurement(self):
        """End measuring inference time."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.end_time = time.time()
        if self.start_time is not None:
            inference_time = self.end_time - self.start_time
            self.inference_times.append(inference_time)

        # Record memory usage
        self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB

        # Record GPU memory usage if available
        if torch.cuda.is_available() and GPU_UTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_memory_usage.append(gpus[0].memoryUsed)
            except Exception:
                pass  # Silently handle GPU monitoring errors

    def get_metrics_summary(self, num_frames: int = 1) -> dict:
        """Get comprehensive metrics summary."""
        if not self.inference_times:
            return {}

        inference_times = np.array(self.inference_times)
        memory_usage = np.array(self.memory_usage) if self.memory_usage else None
        gpu_memory_usage = np.array(self.gpu_memory_usage) if self.gpu_memory_usage else None

        metrics = {
            'total_frames': len(self.inference_times),
            'total_time_seconds': np.sum(inference_times),
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'fps': num_frames / np.sum(inference_times) if np.sum(inference_times) > 0 else 0,
        }

        if memory_usage is not None:
            metrics.update({
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage),
                'memory_usage_std_mb': np.std(memory_usage),
            })

        if gpu_memory_usage is not None:
            metrics.update({
                'avg_gpu_memory_usage_mb': np.mean(gpu_memory_usage),
                'max_gpu_memory_usage_mb': np.max(gpu_memory_usage),
                'gpu_memory_usage_std_mb': np.std(gpu_memory_usage),
            })

        return metrics

    def print_metrics_summary(self, num_frames: int = 1):
        """Print formatted metrics summary."""
        metrics = self.get_metrics_summary(num_frames)

        if not metrics:
            print("No metrics available.")
            return

        print("\n" + "="*60)
        print("📊 INFERENCE PERFORMANCE METRICS")
        print("="*60)

        print("⏱️  TIMING METRICS:")
        print(f"  Total Frames: {metrics['total_frames']}")
        print(f"  Total Time: {metrics['total_time_seconds']:.2f}s")
        print(f"  Average Inference Time: {metrics['avg_inference_time_ms']:.2f}ms")
        print(f"  Min Inference Time: {metrics['min_inference_time_ms']:.2f}ms")
        print(f"  Max Inference Time: {metrics['max_inference_time_ms']:.2f}ms")
        print(f"  Std Inference Time: {metrics['std_inference_time_ms']:.2f}ms")
        print(f"  Frames Per Second: {metrics['fps']:.1f}")

        if 'avg_memory_usage_mb' in metrics:
            print("\n💾 MEMORY METRICS:")
            print(f"  Average Memory Usage: {metrics['avg_memory_usage_mb']:.1f}MB")
            print(f"  Max Memory Usage: {metrics['max_memory_usage_mb']:.1f}MB")
            print(f"  Memory Usage Std: {metrics['memory_usage_std_mb']:.1f}MB")

        if 'avg_gpu_memory_usage_mb' in metrics:
            print("\n🎮 GPU MEMORY METRICS:")
            print(f"  Average GPU Memory: {metrics['avg_gpu_memory_usage_mb']:.1f}MB")
            print(f"  Max GPU Memory: {metrics['max_gpu_memory_usage_mb']:.1f}MB")
            print(f"  GPU Memory Std: {metrics['gpu_memory_usage_std_mb']:.1f}MB")

        print("\n" + "="*60)


def benchmark_image_inference(model, img_path: str, num_runs: int = 10) -> InferenceMetrics:
    """Benchmark image inference performance."""
    metrics = InferenceMetrics()

    print(f"🔬 Benchmarking image inference with {num_runs} runs...")

    # Warm-up run
    print("Warm-up run...")
    _ = inference_model(model, img_path)

    # Benchmark runs
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}...")
        metrics.start_measurement()
        result = inference_model(model, img_path)
        metrics.end_measurement()

    return metrics


def benchmark_video_inference(model, video_path: str, max_frames: int = 100) -> InferenceMetrics:
    """Benchmark video inference performance."""
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV required for video benchmarking. Install with: pip install opencv-python")
        return InferenceMetrics()

    metrics = InferenceMetrics()

    print(f"🎬 Benchmarking video inference (max {max_frames} frames)...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video file: {video_path}")
        return metrics

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video has {total_frames} frames. Processing up to {max_frames} frames...")

    # Warm-up
    ret, frame = cap.read()
    if ret:
        _ = inference_model(model, frame)

    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        metrics.start_measurement()
        result = inference_model(model, frame)
        metrics.end_measurement()

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{min(max_frames, total_frames)} frames...")

    cap.release()
    return metrics


def main():
    parser = ArgumentParser(description='Measure inference performance metrics for MMSegmentation models')
    parser.add_argument('input', help='Input image file or video file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--num-runs', type=int, default=10,
        help='Number of benchmark runs for images (default: 10)')
    parser.add_argument(
        '--max-frames', type=int, default=100,
        help='Maximum number of frames to process for videos (default: 100)')
    parser.add_argument(
        '--output-dir', default='benchmark_results',
        help='Directory to save benchmark results')
    parser.add_argument(
        '--save-sample-result', action='store_true',
        help='Save a sample segmentation result')

    args = parser.parse_args()

    # Check if input is image or video
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        return

    is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']

    print("🚀 Starting MMSegmentation Inference Benchmark")
    print(f"Input: {args.input}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Type: {'Video' if is_video else 'Image'}")

    # Initialize model
    print("\n📦 Loading model...")
    try:
        model = init_model(args.config, args.checkpoint, device=args.device)
        if args.device == 'cpu':
            model = revert_sync_batchnorm(model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run benchmark
    if is_video:
        metrics = benchmark_video_inference(model, str(input_path), args.max_frames)
        num_frames = len(metrics.inference_times)
    else:
        metrics = benchmark_image_inference(model, str(input_path), args.num_runs)
        num_frames = args.num_runs

    # Print results
    metrics.print_metrics_summary(num_frames)

    # Save sample result if requested
    if args.save_sample_result and not is_video:
        print("\n💾 Saving sample segmentation result...")
        try:
            result = inference_model(model, str(input_path))
            show_result_pyplot(
                model, str(input_path), result,
                title='benchmark_sample',
                opacity=0.5,
                with_labels=True,
                draw_gt=False,
                show=False,
                out_file=str(output_dir / 'sample_result.png')
            )
            print(f"✅ Sample result saved to: {output_dir / 'sample_result.png'}")
        except Exception as e:
            print(f"❌ Failed to save sample result: {e}")

    # Save metrics to file
    metrics_file = output_dir / 'inference_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("MMSegmentation Inference Benchmark Results\n")
        f.write("="*50 + "\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Type: {'Video' if is_video else 'Image'}\n\n")

        metrics_dict = metrics.get_metrics_summary(num_frames)
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")

    print(f"\n📄 Detailed metrics saved to: {metrics_file}")
    print("🎉 Benchmark completed!")


if __name__ == '__main__':
    main()
