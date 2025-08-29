#!/usr/bin/env python3
"""
Model Optimization Toolkit for MMSegmentation
Optimizes models for inference without retraining using various techniques
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import warnings
import json
import csv
from datetime import datetime
import hashlib
warnings.filterwarnings("ignore")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False


class ModelOptimizer:
    def __init__(self, config_path, checkpoint_path, device='cuda:0', output_dir='optimized_models'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_filename = Path(checkpoint_path).name  # Store filename for logging
        self.device = device
        self.output_dir = Path(output_dir)
        self.benchmarks_dir = self.output_dir / "benchmarks"
        self.original_model = None
        self.optimized_models = {}
        self.session_id = self._generate_session_id()

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.benchmarks_dir.mkdir(exist_ok=True)

        # Load original model
        self._load_original_model()

    def _generate_session_id(self):
        """Generate unique session ID based on timestamp and config"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = hashlib.md5(str(self.config_path).encode()).hexdigest()[:8]
        return f"{timestamp}_{config_hash}"

    def _get_model_filename(self, model_type, extension):
        """Generate unique filename for model"""
        return f"{model_type}_{self.session_id}.{extension}"

    def _save_benchmark_results(self, results, filename):
        """Save benchmark results to JSON and CSV"""
        # Add checkpoint filename to results
        if isinstance(results, dict):
            results = dict(results)  # Create a copy
            results['checkpoint_filename'] = self.checkpoint_filename

        # Save JSON
        json_path = self.benchmarks_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV
        csv_path = self.benchmarks_dir / f"{filename}.csv"
        if isinstance(results, dict):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
                writer.writerow(results)

    def _log_optimization_summary(self, results):
        """Log optimization summary to file"""
        summary_path = self.benchmarks_dir / f"optimization_summary_{self.session_id}.txt"

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL OPTIMIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Config: {self.config_path}\n")
            f.write(f"Checkpoint: {self.checkpoint_filename}\n")
            f.write(f"Device: {self.device}\n\n")

            f.write("PERFORMANCE COMPARISON:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<15} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<8} {'Speedup':<10}\n")
            f.write("-"*80 + "\n")

            if 'original' in results:
                original_time = results['original']['avg_inference_time_ms']

                for name, metrics in results.items():
                    speedup = original_time / metrics['avg_inference_time_ms']
                    f.write(f"{name:<15} {metrics['model_size_mb']:<10.1f} "
                           f"{metrics['avg_inference_time_ms']:<12.2f} {metrics['fps']:<8.1f} "
                           f"{speedup:<10.2f}x\n")

            f.write("-"*80 + "\n\n")

            f.write("GENERATED FILES:\n")
            for name in self.optimized_models.keys():
                if name in ['fp16', 'int8']:
                    f.write(f"  - {self._get_model_filename(name, 'pth')}\n")
                elif name == 'onnx':
                    f.write(f"  - {self._get_model_filename('onnx', 'onnx')}\n")
                    f.write(f"  - {self._get_model_filename('onnx_optimized', 'onnx')}\n")

            f.write(f"\nBenchmark results saved to: {self.benchmarks_dir}\n")

    def _load_original_model(self):
        """Load the original model"""
        from mmseg.apis import init_model
        print("Loading original model...")
        self.original_model = init_model(self.config_path, self.checkpoint_path, device=self.device)
        if self.original_model is None:
            raise ValueError("Failed to load model")
        print("Original model loaded successfully")

    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

    def benchmark_model(self, model, input_shape=(1, 3, 1024, 1024), num_runs=10):
        """Benchmark model inference speed"""
        model.eval()

        # Create dummy input with appropriate dtype
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Convert input dtype to match model precision
        if next(model.parameters()).dtype == torch.float16:
            dummy_input = dummy_input.half()
        elif next(model.parameters()).dtype == torch.float32:
            dummy_input = dummy_input.float()

        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model(dummy_input)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time

        return {
            'avg_inference_time_ms': avg_time,
            'fps': fps,
            'model_size_mb': self.get_model_size(model)
        }

    def optimize_fp16(self):
        """Convert model to FP16 precision"""
        print("Optimizing model to FP16...")
        assert self.original_model is not None, "Model not loaded"

        import copy
        model_fp16 = copy.deepcopy(self.original_model).half()

        # Convert input to FP16 for consistency
        def fp16_forward_hook(module, input, output):
            if isinstance(input[0], torch.Tensor) and input[0].dtype == torch.float32:
                input[0] = input[0].half()
            return output

        # Register hook to convert inputs to FP16
        for module in model_fp16.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(fp16_forward_hook)

        # Save FP16 model
        fp16_path = self.output_dir / self._get_model_filename('fp16', 'pth')
        torch.save(model_fp16.state_dict(), fp16_path)
        print(f"FP16 model saved to: {fp16_path}")

        self.optimized_models['fp16'] = model_fp16
        print("FP16 optimization completed")
        return model_fp16

    def optimize_quantization_aware(self, calibration_data=None):
        """Apply quantization-aware training simulation"""
        print("Applying quantization-aware optimization...")

        import copy
        # Use torch.quantization for dynamic quantization
        model_int8 = torch.quantization.quantize_dynamic(
            copy.deepcopy(self.original_model),
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

        # Save INT8 model
        int8_path = self.output_dir / self._get_model_filename('int8', 'pth')
        torch.save(model_int8.state_dict(), int8_path)
        print(f"INT8 model saved to: {int8_path}")

        self.optimized_models['int8'] = model_int8
        print("INT8 quantization completed")
        return model_int8

    def convert_to_onnx(self, input_shape=(1, 3, 1024, 1024)):
        """Convert model to ONNX format"""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available. Install with: pip install onnxruntime")
            return None

        onnx_filename = self._get_model_filename('onnx', 'onnx')
        onnx_path = self.output_dir / onnx_filename
        print(f"Converting model to ONNX format: {onnx_path}")
        assert self.original_model is not None, "Model not loaded"

        # Create a fresh copy of the original model for ONNX export
        # to avoid FP16 conversion issues
        from mmseg.apis import init_model
        onnx_model = init_model(self.config_path, self.checkpoint_path, device='cpu')

        # Create dummy input
        dummy_input = torch.randn(input_shape)

        # Export to ONNX
        torch.onnx.export(
            onnx_model,  # type: ignore
            (dummy_input,),
            onnx_path,
            export_params=True,
            opset_version=13,  # Updated from 11 to 13 for unflatten support
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        print(f"ONNX model saved to: {onnx_path}")
        return onnx_path

    def optimize_onnx_model(self, onnx_path):
        """Optimize ONNX model using ONNX Runtime"""
        if not ONNX_AVAILABLE:
            print("ONNX Runtime not available")
            return None

        assert ONNX_AVAILABLE, "ONNX Runtime not available"
        import onnxruntime as ort  # type: ignore

        optimized_filename = self._get_model_filename('onnx_optimized', 'onnx')
        optimized_path = self.output_dir / optimized_filename

        print("Optimizing ONNX model...")

        # Create ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(optimized_path)

        # Create session to trigger optimization
        ort.InferenceSession(onnx_path, sess_options)

        print(f"Optimized ONNX model saved to: {optimized_path}")
        return optimized_path

    def create_tensorrt_engine(self, onnx_path, engine_path, input_shape=(1, 3, 1024, 1024)):
        """Convert ONNX model to TensorRT engine"""
        if not TENSORRT_AVAILABLE:
            print("TensorRT not available. Install TensorRT for maximum performance")
            return None

        assert TENSORRT_AVAILABLE, "TensorRT not available"
        import tensorrt as trt  # type: ignore

        print("Creating TensorRT engine...")

        # This is a simplified version - in practice you'd use trtexec or similar
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        # Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # Set input shape
        profile = builder.create_optimization_profile()
        profile.set_shape("input", input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path

    def optimize_batch_processing(self, batch_size=4):
        """Optimize model for batch processing"""
        print(f"Optimizing model for batch processing (batch_size={batch_size})")

        # Enable cuDNN benchmark mode for optimized convolution algorithms
        torch.backends.cudnn.benchmark = True

        # Set optimal cuDNN settings
        torch.backends.cudnn.enabled = True

        # For batch processing, we can use the original model with optimizations
        model_batch = self.original_model

        # Add batch processing optimizations
        self.optimized_models['batch'] = {
            'model': model_batch,
            'batch_size': batch_size,
            'optimizations': ['cudnn_benchmark', 'batch_processing']
        }

        print("Batch processing optimization completed")
        return model_batch

    def compare_optimizations(self, input_shape=(1, 3, 1024, 1024), num_runs=10):
        """Compare performance of all optimized models"""
        print("\n" + "="*80)
        print("MODEL OPTIMIZATION COMPARISON")
        print("="*80)

        results = {}

        # Benchmark original model
        print("\nBenchmarking original model...")
        results['original'] = self.benchmark_model(self.original_model, input_shape, num_runs)

        # Benchmark optimized models
        for name, model in self.optimized_models.items():
            if isinstance(model, dict) and 'model' in model:
                # Handle batch processing case
                print(f"\nBenchmarking {name} optimized model...")
                results[name] = self.benchmark_model(model['model'], input_shape, num_runs)
                results[name]['batch_size'] = model['batch_size']
            else:
                print(f"\nBenchmarking {name} optimized model...")
                results[name] = self.benchmark_model(model, input_shape, num_runs)

        # Save individual benchmark results
        for name, metrics in results.items():
            self._save_benchmark_results(metrics, f"benchmark_{name}_{self.session_id}")

        # Save comparison results
        self._save_benchmark_results(results, f"comparison_{self.session_id}")

        # Log optimization summary
        self._log_optimization_summary(results)

        # Print comparison table
        print("\n" + "-"*80)
        print(f"{'Model':<15} {'Size (MB)':<10} {'Time (ms)':<12} {'FPS':<8} {'Speedup':<10}")
        print("-"*80)

        original_time = results['original']['avg_inference_time_ms']

        for name, metrics in results.items():
            speedup = original_time / metrics['avg_inference_time_ms']
            print(f"{name:<15} {metrics['model_size_mb']:<10.1f} "
                  f"{metrics['avg_inference_time_ms']:<12.2f} {metrics['fps']:<8.1f} "
                  f"{speedup:<10.2f}x")

        print("-"*80)
        return results


def main():
    parser = argparse.ArgumentParser(description='Model Optimization Toolkit')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint file')
    parser.add_argument('--device', default='cuda:0', help='Device for optimization')
    parser.add_argument('--input-shape', default='1,3,1024,1024',
                       help='Input shape for benchmarking (batch_size,channels,height,width)')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of runs for benchmarking')
    parser.add_argument('--output-dir', default='optimized_models',
                       help='Output directory for optimized models')
    parser.add_argument('--optimize', nargs='+',
                       choices=['fp16', 'int8', 'onnx', 'tensorrt', 'batch', 'all'],
                       default=['all'], help='Optimization techniques to apply')

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize optimizer
    optimizer = ModelOptimizer(args.config, args.checkpoint, args.device)

    # Apply optimizations
    if 'fp16' in args.optimize or 'all' in args.optimize:
        optimizer.optimize_fp16()

    if 'int8' in args.optimize or 'all' in args.optimize:
        optimizer.optimize_quantization_aware()

    if 'batch' in args.optimize or 'all' in args.optimize:
        optimizer.optimize_batch_processing()

    if 'onnx' in args.optimize or 'all' in args.optimize:
        onnx_path = optimizer.convert_to_onnx(input_shape)

        if onnx_path:
            # Optimize ONNX model
            optimizer.optimize_onnx_model(onnx_path)

    if 'tensorrt' in args.optimize or 'all' in args.optimize:
        onnx_filename = optimizer._get_model_filename('onnx', 'onnx')
        onnx_path = optimizer.output_dir / onnx_filename
        if onnx_path.exists():
            engine_filename = optimizer._get_model_filename('tensorrt', 'engine')
            engine_path = optimizer.output_dir / engine_filename
            optimizer.create_tensorrt_engine(onnx_path, engine_path, input_shape)

    # Compare all models
    optimizer.compare_optimizations(input_shape, args.num_runs)

    print(f"\nOptimized models saved to: {optimizer.output_dir}")
    print(f"Benchmark results saved to: {optimizer.benchmarks_dir}")
    print(f"Session ID: {optimizer.session_id}")
    print("\nOptimization complete! 🚀")


if __name__ == '__main__':
    main()
