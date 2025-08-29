#!/usr/bin/env python3
"""
Batch Benchmarking Script for MMSegmentation Models
Benchmarks all .pth files in the root directory
"""

import sys
import os
import glob
import argparse
from pathlib import Path
import json
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.model_optimizer import ModelOptimizer

def find_pth_files(root_dir):
    """Find all .pth files in the root directory"""
    pth_files = []
    for file in glob.glob(os.path.join(root_dir, "*.pth")):
        pth_files.append(os.path.basename(file))
    return sorted(pth_files)

def find_matching_config(checkpoint_name, config_dir="local_configs"):
    """Find matching config file for a checkpoint"""
    # Common config patterns for SegFormer models
    config_patterns = {
        "segformer.b0": "segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b1": "segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b2": "segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b3": "segformer/segformer_mit-b3_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b4": "segformer/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py",
        "segformer.b5": "segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py"
    }

    for key, config_path in config_patterns.items():
        if key in checkpoint_name:
            full_config_path = os.path.join(config_dir, config_path)
            if os.path.exists(full_config_path):
                return full_config_path

    # Fallback to default config
    default_config = os.path.join(config_dir, "segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py")
    if os.path.exists(default_config):
        return default_config

    return None

def benchmark_single_model(checkpoint_path, config_path, device='cuda:0', output_dir='optimized_models'):
    """Benchmark a single model"""
    print(f"\n🔍 Benchmarking: {os.path.basename(checkpoint_path)}")
    print("-" * 60)

    try:
        # Initialize optimizer
        optimizer = ModelOptimizer(config_path, checkpoint_path, device, output_dir)

        # Benchmark original model only (no optimization)
        print("Benchmarking original model...")
        results = optimizer.benchmark_model(optimizer.original_model, num_runs=5)

        # Save results with checkpoint filename
        checkpoint_name = Path(checkpoint_path).stem
        optimizer._save_benchmark_results(results, f"benchmark_{checkpoint_name}_{optimizer.session_id}")

        print(".2f")
        print(".1f")
        print(f"✅ Benchmark completed for {os.path.basename(checkpoint_path)}")

        return results

    except Exception as e:
        print(f"❌ Failed to benchmark {os.path.basename(checkpoint_path)}: {e}")
        return None

def create_batch_summary(all_results, output_dir):
    """Create a summary of all benchmark results"""
    benchmarks_dir = Path(output_dir) / "benchmarks"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save comprehensive results
    summary_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create CSV summary
    csv_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.csv"
    if all_results:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['checkpoint', 'model_size_mb', 'avg_inference_time_ms', 'fps'])
            writer.writeheader()
            for checkpoint, results in all_results.items():
                if results:
                    writer.writerow({
                        'checkpoint': checkpoint,
                        'model_size_mb': results.get('model_size_mb', 0),
                        'avg_inference_time_ms': results.get('avg_inference_time_ms', 0),
                        'fps': results.get('fps', 0)
                    })

    # Create human-readable summary
    txt_file = benchmarks_dir / f"batch_benchmark_summary_{timestamp}.txt"
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BATCH BENCHMARK SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total models benchmarked: {len(all_results)}\n\n")

        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<40} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<8}\n")
        f.write("-"*80 + "\n")

        for checkpoint, results in all_results.items():
            if results:
                f.write(f"{checkpoint:<40} {results['model_size_mb']:<10.1f} "
                       f"{results['avg_inference_time_ms']:<12.2f} {results['fps']:<8.1f}\n")

        f.write("-"*80 + "\n")
        f.write(f"\nDetailed results saved to: {benchmarks_dir}\n")

    return summary_file

def main():
    parser = argparse.ArgumentParser(description='Batch Benchmark MMSegmentation Models')
    parser.add_argument('--root-dir', default='.', help='Root directory to scan for .pth files')
    parser.add_argument('--config-dir', default='local_configs', help='Directory containing config files')
    parser.add_argument('--device', default='cuda:0', help='Device for benchmarking')
    parser.add_argument('--output-dir', default='optimized_models', help='Output directory for results')
    parser.add_argument('--num-runs', type=int, default=5, help='Number of runs for benchmarking')
    parser.add_argument('--checkpoint', help='Specific checkpoint file to benchmark (optional)')

    args = parser.parse_args()

    print("🚀 MMSegmentation Batch Benchmarking")
    print("=" * 60)

    # Find .pth files
    if args.checkpoint:
        pth_files = [args.checkpoint]
        print(f"📁 Benchmarking specific file: {args.checkpoint}")
    else:
        pth_files = find_pth_files(args.root_dir)
        print(f"📁 Found {len(pth_files)} .pth files in {args.root_dir}")

    if not pth_files:
        print("❌ No .pth files found!")
        return

    # Benchmark all models
    all_results = {}

    for pth_file in pth_files:
        checkpoint_path = os.path.join(args.root_dir, pth_file)

        # Find matching config
        config_path = find_matching_config(pth_file, args.config_dir)

        if not config_path:
            print(f"⚠️  No matching config found for {pth_file}, skipping...")
            continue

        print(f"📋 Using config: {config_path}")

        # Benchmark the model
        results = benchmark_single_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=args.device,
            output_dir=args.output_dir
        )

        if results:
            all_results[pth_file] = results

    # Create batch summary
    if all_results:
        summary_file = create_batch_summary(all_results, args.output_dir)
        print("\n📊 Batch benchmarking completed!")
        print(f"📄 Summary saved to: {summary_file}")

        # Print quick summary
        print("\n🎯 QUICK RESULTS SUMMARY:")
        print("-" * 50)
        for checkpoint, results in sorted(all_results.items()):
            if results:
                print(f"{checkpoint:<40} {results['model_size_mb']:<10.1f} "
                      f"{results['avg_inference_time_ms']:<12.2f} {results['fps']:<8.1f}")
    else:
        print("❌ No models were successfully benchmarked!")

if __name__ == "__main__":
    main()
