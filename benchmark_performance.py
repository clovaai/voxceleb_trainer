#!/usr/bin/env python3
"""
Performance Benchmark Script - benchmark_performance.py

This script compares the original vs optimized versions of the VoxCeleb trainer.
Run this to measure the actual speedup achieved with optimizations.

Usage:
    python benchmark_performance.py --config configs/experiment_01.yaml --mode original
    python benchmark_performance.py --config configs/experiment_01_performance_updated.yaml --mode optimized
"""

import time
import torch
import argparse
import sys
import os
import yaml
import numpy as np
from collections import defaultdict

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING = True
except:
    GPU_MONITORING = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")


class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        
    def start(self, name):
        """Start timing a metric"""
        self.start_times[name] = time.time()
        
    def stop(self, name):
        """Stop timing and record metric"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.metrics[name].append(elapsed)
            return elapsed
        return None
    
    def record(self, name, value):
        """Record a metric value"""
        self.metrics[name].append(value)
    
    def get_summary(self):
        """Get summary statistics"""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
        return summary
    
    def print_summary(self):
        """Print formatted summary"""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        summary = self.get_summary()
        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")
            print(f"  Count: {stats['count']}")


def get_gpu_utilization():
    """Get current GPU utilization"""
    if not GPU_MONITORING:
        return None
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'gpu_util': util.gpu,
            'mem_util': util.memory,
            'mem_used_gb': mem_info.used / 1024**3,
            'mem_total_gb': mem_info.total / 1024**3
        }
    except:
        return None


def benchmark_dataloader(config_path, num_batches=100):
    """Benchmark data loading performance"""
    print("\n" + "=" * 80)
    print("BENCHMARKING DATA LOADER")
    print("=" * 80)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    monitor = PerformanceMonitor()
    
    # Import appropriate modules based on config
    if 'performance' in config_path:
        from DatasetLoader_performance_updated import train_dataset_loader, train_dataset_sampler, worker_init_fn
        version = "OPTIMIZED"
    else:
        from DatasetLoader import train_dataset_loader, train_dataset_sampler, worker_init_fn
        version = "ORIGINAL"
    
    print(f"Version: {version}")
    print(f"Config: {config_path}")
    print(f"Batch size: {config.get('batch_size', 100)}")
    print(f"Workers: {config.get('nDataLoaderThread', 5)}")
    
    # Add missing parameters with defaults
    dataset_config = config.copy()
    if 'max_frames' not in dataset_config:
        dataset_config['max_frames'] = 200  # Default value
    if 'max_seg_per_spk' not in dataset_config:
        dataset_config['max_seg_per_spk'] = 500  # Default value
    if 'seed' not in dataset_config:
        dataset_config['seed'] = 10  # Default value
    if 'nPerSpeaker' not in dataset_config:
        dataset_config['nPerSpeaker'] = 1  # Default value
    if 'distributed' not in dataset_config:
        dataset_config['distributed'] = False  # Default value
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = train_dataset_loader(**dataset_config)
    
    # Create sampler
    sampler_config = dataset_config.copy()
    sampler = train_dataset_sampler(dataset, **sampler_config)
    
    # Create dataloader with appropriate settings
    dataloader_kwargs = {
        'batch_size': config.get('batch_size', 100),
        'num_workers': config.get('nDataLoaderThread', 5),
        'sampler': sampler,
        'worker_init_fn': worker_init_fn,
        'drop_last': True,
    }
    
    # Add optimized settings if available
    if 'prefetch_factor' in config:
        dataloader_kwargs['prefetch_factor'] = config['prefetch_factor']
    if 'persistent_workers' in config:
        dataloader_kwargs['persistent_workers'] = config['persistent_workers']
    if config.get('pin_memory', False):
        dataloader_kwargs['pin_memory'] = True
    
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    
    print(f"\nBenchmarking {num_batches} batches...")
    
    batch_times = []
    gpu_utils = []
    
    # Warmup
    print("Warming up...")
    for i, (data, labels) in enumerate(loader):
        if i >= 5:
            break
    
    # Actual benchmark
    print("Running benchmark...")
    start_total = time.time()
    
    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Simulate minimal processing
        data = data.cuda(non_blocking=True)
        torch.cuda.synchronize()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Get GPU utilization
        gpu_util = get_gpu_utilization()
        if gpu_util:
            gpu_utils.append(gpu_util['gpu_util'])
        
        if (i + 1) % 20 == 0:
            avg_time = np.mean(batch_times[-20:])
            throughput = config.get('batch_size', 100) / avg_time
            print(f"  Batch {i+1}/{num_batches}: {avg_time:.4f}s ({throughput:.1f} samples/s)")
    
    total_time = time.time() - start_total
    
    # Results
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    print(f"Total batches: {len(batch_times)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average batch time: {np.mean(batch_times):.4f}s ± {np.std(batch_times):.4f}s")
    print(f"Min batch time: {np.min(batch_times):.4f}s")
    print(f"Max batch time: {np.max(batch_times):.4f}s")
    print(f"Throughput: {config.get('batch_size', 100) * len(batch_times) / total_time:.1f} samples/s")
    
    if gpu_utils:
        print(f"\nGPU Utilization:")
        print(f"  Average: {np.mean(gpu_utils):.1f}%")
        print(f"  Max: {np.max(gpu_utils):.1f}%")
    
    return {
        'version': version,
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'throughput': config.get('batch_size', 100) * len(batch_times) / total_time,
        'gpu_util': np.mean(gpu_utils) if gpu_utils else None
    }


def compare_results(original_results, optimized_results):
    """Compare original vs optimized results"""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = original_results['total_time'] / optimized_results['total_time']
    throughput_increase = (optimized_results['throughput'] / original_results['throughput'] - 1) * 100
    
    print(f"\nOriginal:")
    print(f"  Total time: {original_results['total_time']:.2f}s")
    print(f"  Avg batch time: {original_results['avg_batch_time']:.4f}s")
    print(f"  Throughput: {original_results['throughput']:.1f} samples/s")
    if original_results['gpu_util']:
        print(f"  GPU util: {original_results['gpu_util']:.1f}%")
    
    print(f"\nOptimized:")
    print(f"  Total time: {optimized_results['total_time']:.2f}s")
    print(f"  Avg batch time: {optimized_results['avg_batch_time']:.4f}s")
    print(f"  Throughput: {optimized_results['throughput']:.1f} samples/s")
    if optimized_results['gpu_util']:
        print(f"  GPU util: {optimized_results['gpu_util']:.1f}%")
    
    print(f"\n{'🚀 IMPROVEMENT:':}")
    print(f"  Speedup: {speedup:.2f}x faster")
    print(f"  Throughput increase: +{throughput_increase:.1f}%")
    
    if original_results['gpu_util'] and optimized_results['gpu_util']:
        gpu_util_increase = optimized_results['gpu_util'] - original_results['gpu_util']
        print(f"  GPU utilization increase: +{gpu_util_increase:.1f}%")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Benchmark VoxCeleb Trainer Performance')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to benchmark')
    parser.add_argument('--compare', action='store_true', help='Compare with original')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VOXCELEB TRAINER PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = benchmark_dataloader(args.config, args.num_batches)
    
    if args.compare:
        print("\nNow benchmarking original version for comparison...")
        original_config = args.config.replace('_performance_updated', '')
        if os.path.exists(original_config):
            original_results = benchmark_dataloader(original_config, args.num_batches)
            compare_results(original_results, results)
        else:
            print(f"Warning: Original config not found at {original_config}")
    
    print("\n✓ Benchmark complete!")


if __name__ == '__main__':
    main()
