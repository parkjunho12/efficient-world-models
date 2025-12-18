#!/usr/bin/env python3
"""
Evaluation script for World Model.

Comprehensive evaluation including:
- Reconstruction metrics (PSNR, SSIM, LPIPS)
- Prediction metrics (temporal consistency)
- Latency benchmarking
- Sample visualizations
- Quantitative analysis

Usage:
    # Basic evaluation
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    
    # With specific dataset
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --data-root data/processed/nuscenes \
        --split val
    
    # Save visualizations
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --output-dir results/eval \
        --save-samples
    
    # Benchmark latency
    python scripts/evaluate.py \
        --checkpoint checkpoints/best.pt \
        --benchmark
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.world_model import build_world_model
from src.data.datasets.nuscenes import NuScenesDataset
from src.data.transforms.spatial import MinimalTransform
from src.data.loaders.collate import create_dataloader
from src.evaluation.metrics.image_quality import calculate_psnr, calculate_ssim
from src.utils.checkpointing.loader import CheckpointLoader


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    
    Evaluates model on multiple metrics and generates detailed reports.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path
    ):
        """
        Args:
            model: Model to evaluate
            device: Device to run on
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        
        # Try to load LPIPS
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        except:
            print("Warning: LPIPS not available")
            self.lpips_fn = None
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Data batch
        
        Returns:
            Metrics dictionary
        """
        images = batch['images'].to(self.device)  # (B, T, C, H, W)
        actions = batch['actions'].to(self.device)  # (B, T-1, A)
        
        # Forward pass
        outputs = self.model(images, actions)
        
        reconstructed = outputs['reconstructed']  # (B, T, C, H, W)
        predicted = outputs['predicted']  # (B, T-1, C, H, W)
        
        # Calculate metrics
        metrics = {}
        
        # Reconstruction metrics (all frames)
        recon_psnr = calculate_psnr(reconstructed, images)
        recon_ssim = calculate_ssim(reconstructed, images)
        metrics['recon_psnr'] = recon_psnr.item()
        metrics['recon_ssim'] = recon_ssim.item()
        
        if self.lpips_fn:
            # LPIPS expects (B*T, C, H, W)
            B, T = images.shape[:2]
            recon_flat = reconstructed.reshape(B * T, *reconstructed.shape[2:])
            images_flat = images.reshape(B * T, *images.shape[2:])
            recon_lpips = self.lpips_fn(recon_flat, images_flat).mean()
            metrics['recon_lpips'] = recon_lpips.item()
        
        # Prediction metrics (future frames)
        pred_psnr = calculate_psnr(predicted, images[:, 1:])
        pred_ssim = calculate_ssim(predicted, images[:, 1:])
        metrics['pred_psnr'] = pred_psnr.item()
        metrics['pred_ssim'] = pred_ssim.item()
        
        if self.lpips_fn:
            B, T = predicted.shape[:2]
            pred_flat = predicted.reshape(B * T, *predicted.shape[2:])
            target_flat = images[:, 1:].reshape(B * T, *images.shape[2:])
            pred_lpips = self.lpips_fn(pred_flat, target_flat).mean()
            metrics['pred_lpips'] = pred_lpips.item()
        
        # Temporal consistency
        if T > 1:
            temporal_diff = torch.mean(
                torch.abs(predicted[:, 1:] - predicted[:, :-1])
            )
            metrics['temporal_consistency'] = temporal_diff.item()
        
        return metrics
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        max_batches: int = None
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset.
        
        Args:
            dataloader: Data loader
            max_batches: Maximum batches to evaluate (None = all)
        
        Returns:
            Aggregated metrics
        """
        self.model.eval()
        
        all_metrics = []
        
        pbar = tqdm(dataloader, desc="Evaluating")
        for i, batch in enumerate(pbar):
            if max_batches and i >= max_batches:
                break
            
            try:
                metrics = self.evaluate_batch(batch)
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'PSNR': f"{metrics['recon_psnr']:.2f}",
                    'SSIM': f"{metrics['recon_ssim']:.3f}"
                })
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
        
        if not all_metrics:
            print("No metrics collected!")
            return {}
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    @torch.no_grad()
    def generate_samples(
        self,
        dataloader: DataLoader,
        num_samples: int = 8
    ):
        """
        Generate visualization samples.
        
        Args:
            dataloader: Data loader
            num_samples: Number of samples to generate
        """
        self.model.eval()
        
        # Get first batch
        batch = next(iter(dataloader))
        images = batch['images'][:num_samples].to(self.device)
        actions = batch['actions'][:num_samples].to(self.device)
        
        # Forward pass
        outputs = self.model(images, actions)
        reconstructed = outputs['reconstructed']
        predicted = outputs['predicted']
        
        # Move to CPU for visualization
        images = images.cpu()
        reconstructed = reconstructed.cpu()
        predicted = predicted.cpu()
        
        # Save individual samples
        for i in range(num_samples):
            self._save_sample_comparison(
                ground_truth=images[i],
                reconstructed=reconstructed[i],
                predicted=predicted[i],
                sample_idx=i
            )
        
        # Create summary visualization
        self._create_summary_figure(
            images, reconstructed, predicted
        )
        
        print(f"✓ Samples saved to {self.output_dir / 'samples'}")
    
    def _save_sample_comparison(
        self,
        ground_truth: torch.Tensor,
        reconstructed: torch.Tensor,
        predicted: torch.Tensor,
        sample_idx: int
    ):
        """Save comparison visualization for a single sample."""
        T = ground_truth.shape[0]
        num_frames = min(T, 5)
        
        fig, axes = plt.subplots(3, num_frames, figsize=(num_frames * 3, 9))
        if num_frames == 1:
            axes = axes.reshape(-1, 1)
        
        for t in range(num_frames):
            # Ground truth
            axes[0, t].imshow(self._tensor_to_image(ground_truth[t]))
            axes[0, t].set_title(f'GT t={t}')
            axes[0, t].axis('off')
            
            # Reconstruction
            axes[1, t].imshow(self._tensor_to_image(reconstructed[t]))
            psnr = calculate_psnr(
                reconstructed[t].unsqueeze(0),
                ground_truth[t].unsqueeze(0)
            ).item()
            axes[1, t].set_title(f'Recon t={t}\nPSNR: {psnr:.1f}dB')
            axes[1, t].axis('off')
            
            # Prediction (if available)
            if t > 0 and t <= predicted.shape[0]:
                axes[2, t].imshow(self._tensor_to_image(predicted[t-1]))
                pred_psnr = calculate_psnr(
                    predicted[t-1].unsqueeze(0),
                    ground_truth[t].unsqueeze(0)
                ).item()
                axes[2, t].set_title(f'Pred t={t}\nPSNR: {pred_psnr:.1f}dB')
            axes[2, t].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'samples' / f'sample_{sample_idx:03d}.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
    
    def _create_summary_figure(
        self,
        images: torch.Tensor,
        reconstructed: torch.Tensor,
        predicted: torch.Tensor
    ):
        """Create summary comparison figure."""
        N = min(4, images.shape[0])
        T = min(3, images.shape[1])
        
        fig, axes = plt.subplots(N, T * 3, figsize=(T * 9, N * 3))
        if N == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(N):
            for t in range(T):
                # Ground truth
                col = t * 3
                axes[i, col].imshow(self._tensor_to_image(images[i, t]))
                if i == 0:
                    axes[i, col].set_title(f'GT t={t}')
                axes[i, col].axis('off')
                
                # Reconstruction
                col = t * 3 + 1
                axes[i, col].imshow(self._tensor_to_image(reconstructed[i, t]))
                if i == 0:
                    axes[i, col].set_title(f'Recon t={t}')
                axes[i, col].axis('off')
                
                # Prediction
                col = t * 3 + 2
                if t > 0 and t <= predicted.shape[1]:
                    axes[i, col].imshow(self._tensor_to_image(predicted[i, t-1]))
                    if i == 0:
                        axes[i, col].set_title(f'Pred t={t}')
                axes[i, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            self.output_dir / 'visualizations' / 'summary.png',
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image."""
        # Denormalize from [-1, 1] to [0, 1]
        image = (tensor + 1) / 2
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy
        image = image.permute(1, 2, 0).numpy()
        
        return image
    
    def benchmark_latency(
        self,
        batch_size: int = 1,
        sequence_length: int = 10,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model latency.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            num_iterations: Number of iterations
            warmup_iterations: Warmup iterations
        
        Returns:
            Latency statistics
        """
        self.model.eval()
        
        # Create dummy data
        images = torch.randn(
            batch_size, sequence_length, 3, 256, 256,
            device=self.device
        )
        actions = torch.randn(
            batch_size, sequence_length - 1, 4,
            device=self.device
        )
        
        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(images, actions)
        
        # Benchmark
        print(f"Benchmarking ({num_iterations} iterations)...")
        latencies = []
        
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.time()
            
            with torch.no_grad():
                _ = self.model(images, actions)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'throughput_fps': float(1000 / np.mean(latencies))
        }
        
        return stats
    
    def create_report(self, metrics: Dict, latency_stats: Dict = None):
        """Create evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("WORLD MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        if metrics:
            # Reconstruction metrics
            report.append("📊 RECONSTRUCTION METRICS")
            report.append("-" * 80)
            report.append(f"PSNR:  {metrics.get('recon_psnr_mean', 0):.2f} ± {metrics.get('recon_psnr_std', 0):.2f} dB")
            report.append(f"       Range: [{metrics.get('recon_psnr_min', 0):.2f}, {metrics.get('recon_psnr_max', 0):.2f}]")
            report.append(f"SSIM:  {metrics.get('recon_ssim_mean', 0):.4f} ± {metrics.get('recon_ssim_std', 0):.4f}")
            report.append(f"       Range: [{metrics.get('recon_ssim_min', 0):.4f}, {metrics.get('recon_ssim_max', 0):.4f}]")
            
            if 'recon_lpips_mean' in metrics:
                report.append(f"LPIPS: {metrics['recon_lpips_mean']:.4f} ± {metrics['recon_lpips_std']:.4f}")
            report.append("")
            
            # Prediction metrics
            report.append("🔮 PREDICTION METRICS")
            report.append("-" * 80)
            report.append(f"PSNR:  {metrics.get('pred_psnr_mean', 0):.2f} ± {metrics.get('pred_psnr_std', 0):.2f} dB")
            report.append(f"SSIM:  {metrics.get('pred_ssim_mean', 0):.4f} ± {metrics.get('pred_ssim_std', 0):.4f}")
            if 'temporal_consistency_mean' in metrics:
                report.append(f"Temporal Consistency: {metrics['temporal_consistency_mean']:.4f}")
            report.append("")
        
        # Latency metrics
        if latency_stats:
            report.append("⚡ LATENCY BENCHMARKS")
            report.append("-" * 80)
            report.append(f"Mean:       {latency_stats['mean_ms']:.2f} ms")
            report.append(f"Std:        {latency_stats['std_ms']:.2f} ms")
            report.append(f"P50:        {latency_stats['p50_ms']:.2f} ms")
            report.append(f"P95:        {latency_stats['p95_ms']:.2f} ms")
            report.append(f"P99:        {latency_stats['p99_ms']:.2f} ms")
            report.append(f"Throughput: {latency_stats['throughput_fps']:.2f} FPS")
            report.append("")
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        report.append("🏗️  MODEL INFORMATION")
        report.append("-" * 80)
        report.append(f"Parameters: {total_params / 1e6:.2f}M")
        report.append("")
        
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Print to console
        print(report_text)
        
        # Save to file
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text
    
    def save_results(self, metrics: Dict, latency_stats: Dict = None):
        """Save results as JSON."""
        results = {
            'metrics': metrics,
            'latency': latency_stats,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {self.output_dir / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model')
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/nuscenes',
        help='Path to dataset'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum batches to evaluate (None = all)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory'
    )
    parser.add_argument(
        '--save-samples',
        action='store_true',
        help='Save sample visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='Number of samples to visualize'
    )
    
    # Benchmark arguments
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run latency benchmark'
    )
    parser.add_argument(
        '--benchmark-batch-size',
        type=int,
        default=1,
        help='Batch size for benchmarking'
    )
    parser.add_argument(
        '--benchmark-iterations',
        type=int,
        default=100,
        help='Number of benchmark iterations'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting evaluation with checkpoint: {args.checkpoint}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("model")
        or checkpoint.get("state_dict")
    )

    if state_dict is None:
        raise KeyError(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    # Build model
    print("Building model...")
    model_config = checkpoint.get('config', {}).get('model', {
        'latent_dim': 256,
        'action_dim': 4,
        'hidden_dim': 512
    })
    model = build_world_model(model_config)
    
    # Load weights
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"✓ Model loaded (epoch {epoch})")
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device, output_dir)
    
    # Evaluate on dataset
    metrics = {}
    if Path(args.data_root).exists():
        print(f"\nLoading dataset from {args.data_root}...")
        
        try:
            # Create dataset
            transform = MinimalTransform(image_size=(256, 256))
            dataset = NuScenesDataset(
                data_root=args.data_root,
                split=args.split,
                transform=transform
            )
            
            # Create dataloader
            dataloader = create_dataloader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            print(f"✓ Dataset loaded ({len(dataset)} samples)")
            
            # Run evaluation
            print("\nEvaluating on dataset...")
            metrics = evaluator.evaluate_dataset(dataloader, args.max_batches)
            
            # Generate samples
            if args.save_samples:
                print("\nGenerating sample visualizations...")
                evaluator.generate_samples(dataloader, args.num_samples)
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Continuing with latency benchmark only...")
    else:
        print(f"Warning: Data root not found: {args.data_root}")
        print("Skipping dataset evaluation...")
    
    # Benchmark latency
    latency_stats = None
    if args.benchmark:
        print("\nRunning latency benchmark...")
        latency_stats = evaluator.benchmark_latency(
            batch_size=args.benchmark_batch_size,
            num_iterations=args.benchmark_iterations
        )
        print(f"✓ Mean latency: {latency_stats['mean_ms']:.2f} ms")
        print(f"✓ Throughput: {latency_stats['throughput_fps']:.2f} FPS")
    
    # Create report
    print("\nGenerating report...")
    evaluator.create_report(metrics, latency_stats)
    evaluator.save_results(metrics, latency_stats)
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()