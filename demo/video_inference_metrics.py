#!/usr/bin/env python3
"""
Video Inference Demo for MMSegmentation
Measures performance metrics for video segmentation processing
"""

import argparse
import time
import cv2
import numpy as np
import torch
import psutil
import GPUtil
from pathlib import Path
from mmseg.apis import init_model, inference_model
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import warnings
warnings.filterwarnings("ignore")


class VideoInferenceMetrics:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.gpu_available = torch.cuda.is_available()
        self.metrics = {
            'total_frames': 0,
            'processed_frames': 0,
            'inference_times': [],
            'fps_values': [],
            'gpu_memory_usage': [],
            'cpu_memory_usage': [],
            'gpu_utilization': []
        }

    def measure_system_resources(self):
        """Measure current system resource usage"""
        resources = {}

        # CPU memory
        process = psutil.Process()
        resources['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024

        # GPU metrics if available
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    resources['gpu_memory_mb'] = gpu.memoryUsed
                    resources['gpu_utilization'] = gpu.load * 100
                else:
                    resources['gpu_memory_mb'] = 0
                    resources['gpu_utilization'] = 0
            except:
                resources['gpu_memory_mb'] = 0
                resources['gpu_utilization'] = 0
        else:
            resources['gpu_memory_mb'] = 0
            resources['gpu_utilization'] = 0

        return resources

    def process_video(self, video_path, model, max_frames=None, skip_frames=1):
        """Process video and collect metrics"""
        print(f"Opening video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} total frames")

        self.metrics['total_frames'] = total_frames
        self.metrics['video_fps'] = fps
        self.metrics['video_resolution'] = f"{width}x{height}"

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if requested
            if frame_count % skip_frames != 0:
                continue

            # Limit frames if specified
            if max_frames and processed_count >= max_frames:
                break

            # Measure resources before inference
            pre_resources = self.measure_system_resources()

            # Perform inference
            start_time = time.time()

            # Convert BGR to RGB and prepare for model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = inference_model(model, rgb_frame)

            inference_time = time.time() - start_time

            # Measure resources after inference
            post_resources = self.measure_system_resources()

            # Store metrics
            self.metrics['inference_times'].append(inference_time * 1000)  # Convert to ms
            self.metrics['gpu_memory_usage'].append(post_resources['gpu_memory_mb'])
            self.metrics['cpu_memory_usage'].append(post_resources['cpu_memory_mb'])
            self.metrics['gpu_utilization'].append(post_resources['gpu_utilization'])

            processed_count += 1

            if processed_count % 10 == 0:
                print(f"Processed {processed_count} frames...")

        cap.release()

        self.metrics['processed_frames'] = processed_count
        self.calculate_derived_metrics()

    def calculate_derived_metrics(self):
        """Calculate derived performance metrics"""
        if not self.metrics['inference_times']:
            return

        times = np.array(self.metrics['inference_times'])

        self.metrics['avg_inference_time_ms'] = np.mean(times)
        self.metrics['min_inference_time_ms'] = np.min(times)
        self.metrics['max_inference_time_ms'] = np.max(times)
        self.metrics['std_inference_time_ms'] = np.std(times)

        # Calculate FPS (frames per second for inference)
        avg_time_per_frame_sec = self.metrics['avg_inference_time_ms'] / 1000
        self.metrics['avg_fps'] = 1.0 / avg_time_per_frame_sec if avg_time_per_frame_sec > 0 else 0

        # Memory statistics
        if self.metrics['gpu_memory_usage']:
            gpu_mem = np.array(self.metrics['gpu_memory_usage'])
            self.metrics['avg_gpu_memory_mb'] = np.mean(gpu_mem)
            self.metrics['max_gpu_memory_mb'] = np.max(gpu_mem)

        if self.metrics['cpu_memory_usage']:
            cpu_mem = np.array(self.metrics['cpu_memory_usage'])
            self.metrics['avg_cpu_memory_mb'] = np.mean(cpu_mem)
            self.metrics['max_cpu_memory_mb'] = np.max(cpu_mem)

        if self.metrics['gpu_utilization']:
            gpu_util = np.array(self.metrics['gpu_utilization'])
            self.metrics['avg_gpu_utilization'] = np.mean(gpu_util)
            self.metrics['max_gpu_utilization'] = np.max(gpu_util)

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("VIDEO INFERENCE PERFORMANCE SUMMARY")
        print("="*60)

        print(f"Video Resolution: {self.metrics.get('video_resolution', 'N/A')}")
        print(f"Video FPS: {self.metrics.get('video_fps', 'N/A'):.1f}")
        print(f"Total Frames: {self.metrics['total_frames']}")
        print(f"Processed Frames: {self.metrics['processed_frames']}")

        print("\nInference Performance:")
        print(f"  Average Inference Time: {self.metrics['avg_inference_time_ms']:.2f} ms")
        print(f"  Min Inference Time: {self.metrics['min_inference_time_ms']:.2f} ms")
        print(f"  Max Inference Time: {self.metrics['max_inference_time_ms']:.2f} ms")
        print(f"  Std Inference Time: {self.metrics['std_inference_time_ms']:.2f} ms")
        print(f"  Average FPS: {self.metrics['avg_fps']:.2f}")

        print("\nMemory Usage:")
        if self.gpu_available and self.metrics.get('avg_gpu_memory_mb', 0) > 0:
            print(f"  Average GPU Memory: {self.metrics['avg_gpu_memory_mb']:.1f} MB")
            print(f"  Max GPU Memory: {self.metrics['max_gpu_memory_mb']:.1f} MB")
            print(f"  Average GPU Utilization: {self.metrics['avg_gpu_utilization']:.1f}%")
            print(f"  Max GPU Utilization: {self.metrics['max_gpu_utilization']:.1f}%")

        print(f"  Average CPU Memory: {self.metrics['avg_cpu_memory_mb']:.1f} MB")
        print(f"  Max CPU Memory: {self.metrics['max_cpu_memory_mb']:.1f} MB")

        print("\nReal-time Capability:")
        video_fps = self.metrics.get('video_fps', 30)
        if self.metrics['avg_fps'] >= video_fps:
            print(f"  ✅ Real-time capable ({self.metrics['avg_fps']:.1f} >= {video_fps:.1f} FPS)")
        else:
            print(f"  ❌ Not real-time ({self.metrics['avg_fps']:.1f} < {video_fps:.1f} FPS)")
            speedup_needed = video_fps / self.metrics['avg_fps']
            print(f"     Speedup needed: {speedup_needed:.1f}x")

        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Video Inference Performance Metrics')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference on')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--skip-frames', type=int, default=1,
                       help='Process every Nth frame (for faster testing)')

    args = parser.parse_args()

    # Initialize model
    print("Loading model...")
    model = init_model(args.config, args.checkpoint, device=args.device)

    # Initialize metrics collector
    metrics = VideoInferenceMetrics(device=args.device)

    # Process video
    try:
        metrics.process_video(
            video_path=args.video_path,
            model=model,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )

        # Print results
        metrics.print_summary()

    except Exception as e:
        print(f"Error processing video: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
