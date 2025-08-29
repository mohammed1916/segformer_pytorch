#!/usr/bin/env python3
"""
Quick Start Model Optimization
Demonstrates the most common optimization workflow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.model_optimizer import ModelOptimizer

def quick_optimize():
    """Quick optimization demonstration"""

    # Configuration - adjust these paths for your setup
    config_path = "local_configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py"
    checkpoint_path = "segformer.b0.1024x1024.city.160k.pth"
    device = "cuda:0"

    print("🚀 MMSegmentation Model Optimization Quick Start")
    print("=" * 60)

    # Initialize optimizer
    try:
        optimizer = ModelOptimizer(config_path, checkpoint_path, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the config and checkpoint paths are correct")
        return

    # Apply most effective optimizations
    print("\n📊 Applying optimizations...")

    # FP16 for immediate speedup
    print("1. Converting to FP16...")
    optimizer.optimize_fp16()

    # INT8 for maximum compression
    print("2. Applying INT8 quantization...")
    optimizer.optimize_quantization_aware()

    # Batch processing optimization
    print("3. Optimizing for batch processing...")
    optimizer.optimize_batch_processing(batch_size=4)

    # Benchmark all versions
    print("\n⚡ Benchmarking optimized models...")
    results = optimizer.compare_optimizations(num_runs=5)

    # Show recommendations
    print("\n💡 Recommendations:")
    if 'fp16' in results:
        fp16_speedup = results['original']['avg_inference_time_ms'] / results['fp16']['avg_inference_time_ms']
        print(f"   - FP16: {fp16_speedup:.1f}x speedup")
        print("   - Best for: Real-time applications with GPU")

    if 'int8' in results:
        int8_speedup = results['original']['avg_inference_time_ms'] / results['int8']['avg_inference_time_ms']
        print(f"   - INT8: {int8_speedup:.1f}x speedup")
        print("   - Best for: Edge devices, mobile deployment")

    print("\n✅ Optimization complete!")
    print("Use the optimized models in your inference pipeline for better performance.")

if __name__ == "__main__":
    quick_optimize()
