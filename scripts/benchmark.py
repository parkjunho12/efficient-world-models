#!/usr/bin/env python3
"""
Benchmark world model performance.

Measures latency, throughput, and memory usage.
"""

import sys
import torch
import argparse
from pathlib import Path
import time
import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.world_model import build_world_model
from evaluation.metrics.efficiency import LatencyBenchmark


def benchmark_latency(model, device, batch_sizes=[1, 2, 4, 8]):
    """Benchmark inference latency for different batch sizes."""
    print("\n" + "=" * 80)
    print("Latency Benchmark")
    print("=" * 80)
    
    results = []
    
    for batch_size in batch_sizes:
        benchmark = LatencyBenchmark(model, device)
        metrics = benchmark.measure(batch_size=batch_size, num_iterations=100)
        
        results.append([
            batch_size,
            f"{metrics['mean']:.2f}",
            f"{metrics['std']:.2f}",
            f"{metrics['min']:.2f}",
            f"{metrics['max']:.2f}",
            f"{metrics['p95']:.2f}"
        ])
    
    headers = ['Batch Size', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)', 'P95 (ms)']
    print(tabulate(results, headers=headers, tablefmt='grid'))


def benchmark_throughput(model, device, batch_sizes=[1, 2, 4, 8, 16]):
    """Benchmark throughput (FPS) for different batch sizes."""
    print("\n" + "=" * 80)
    print("Throughput Benchmark")
    print("=" * 80)
    
    model.eval()
    results = []
    
    for batch_size in batch_sizes:
        dummy_image = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model.encode(dummy_image)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        num_iterations = 100
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model.encode(dummy_image)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        fps = (num_iterations * batch_size) / elapsed
        
        results.append([
            batch_size,
            f"{fps:.2f}",
            f"{elapsed / num_iterations * 1000:.2f}"
        ])
    
    headers = ['Batch Size', 'Throughput (FPS)', 'Latency (ms)']
    print(tabulate(results, headers=headers, tablefmt='grid'))


def benchmark_memory(model, device, batch_sizes=[1, 2, 4, 8, 16]):
    """Benchmark GPU memory usage."""
    if device.type != 'cuda':
        print("\n⚠️  Memory benchmark only available on CUDA devices")
        return
    
    print("\n" + "=" * 80)
    print("Memory Benchmark")
    print("=" * 80)
    
    model.eval()
    results = []
    
    for batch_size in batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_image = torch.randn(batch_size, 3, 256, 256, device=device)
        dummy_action = torch.randn(batch_size, 4, device=device)
        
        with torch.no_grad():
            _ = model.encode(dummy_image)
            latent = model.encode(dummy_image)
            _ = model.dynamics(latent, dummy_action)
        
        torch.cuda.synchronize()
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        current_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        results.append([
            batch_size,
            f"{current_memory:.2f}",
            f"{peak_memory:.2f}"
        ])
    
    headers = ['Batch Size', 'Current (MB)', 'Peak (MB)']
    print(tabulate(results, headers=headers, tablefmt='grid'))


def benchmark_model_size(model):
    """Benchmark model size and parameter count."""
    print("\n" + "=" * 80)
    print("Model Size Benchmark")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get component sizes
    component_params = model.get_num_params()
    
    results = []
    for name, count in component_params.items():
        results.append([
            name.capitalize(),
            f"{count / 1e6:.2f}M",
            f"{count / total_params * 100:.1f}%"
        ])
    
    headers = ['Component', 'Parameters', 'Percentage']
    print(tabulate(results, headers=headers, tablefmt='grid'))
    
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Estimate model size on disk
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Estimated size on disk: {param_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Benchmark world model performance')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional, will use random weights if not provided)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    parser.add_argument(
        '--benchmarks',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'latency', 'throughput', 'memory', 'size'],
        help='Benchmarks to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create model
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        config = checkpoint['config']['model']
        model = build_world_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from epoch {checkpoint['epoch']}")
    else:
        print("Using random weights (no checkpoint provided)")
        config = {
            'latent_dim': 256,
            'action_dim': 4,
            'hidden_dim': 512,
            'num_layers': 4
        }
        model = build_world_model(config)
    
    model = model.to(device)
    model.eval()
    
    # Run benchmarks
    benchmarks = args.benchmarks
    if 'all' in benchmarks:
        benchmarks = ['latency', 'throughput', 'memory', 'size']
    
    if 'size' in benchmarks:
        benchmark_model_size(model)
    
    if 'latency' in benchmarks:
        benchmark_latency(model, device)
    
    if 'throughput' in benchmarks:
        benchmark_throughput(model, device)
    
    if 'memory' in benchmarks:
        benchmark_memory(model, device)
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)