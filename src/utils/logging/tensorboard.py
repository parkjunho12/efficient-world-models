"""
TensorBoard logging utilities for training monitoring.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    TensorBoard logger with convenience methods for common logging tasks.
    
    Features:
        - Scalar logging (loss, metrics)
        - Image logging (reconstructions, predictions)
        - Histogram logging (weights, gradients)
        - Text logging (hyperparameters)
    """
    
    def __init__(self, log_dir: str = './runs', experiment_name: str = 'world_model'):
        """
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        print(f"TensorBoard logging to: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the scalar (e.g., 'train/loss')
            value: Scalar value
            step: Global step
        """
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """
        Log multiple scalars in one plot.
        
        Args:
            tag: Group name (e.g., 'losses')
            values: Dict of {name: value}
            step: Global step
        """
        self.writer.add_scalars(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """
        Log a single image.
        
        Args:
            tag: Name of the image
            image: (C, H, W) tensor in [0, 1] or [-1, 1]
            step: Global step
        """
        # Normalize to [0, 1] if needed
        if image.min() < 0:
            image = (image + 1) / 2
        
        self.writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int, max_images: int = 8):
        """
        Log a batch of images as a grid.
        
        Args:
            tag: Name for the image grid
            images: (B, C, H, W) tensor
            step: Global step
            max_images: Maximum number of images to show
        """
        import torchvision
        
        # Take first max_images
        images = images[:max_images]
        
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Create grid
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=False)
        
        self.writer.add_image(tag, grid, step)
    
    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """
        Log a video sequence.
        
        Args:
            tag: Name of the video
            video: (T, C, H, W) or (B, T, C, H, W) tensor
            step: Global step
            fps: Frames per second
        """
        # Add batch dimension if needed
        if video.ndim == 4:
            video = video.unsqueeze(0)
        
        # Normalize to [0, 1]
        if video.min() < 0:
            video = (video + 1) / 2
        
        # TensorBoard expects (B, T, C, H, W) in [0, 255]
        video = (video * 255).byte()
        
        self.writer.add_video(tag, video, step, fps=fps)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """
        Log a histogram of values.
        
        Args:
            tag: Name of the histogram
            values: Tensor of values
            step: Global step
        """
        self.writer.add_histogram(tag, values, step)
    
    def log_model_weights(self, model: torch.nn.Module, step: int):
        """
        Log histograms of all model weights.
        
        Args:
            model: PyTorch model
            step: Global step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'weights/{name}', param, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_comparison(
        self,
        tag: str,
        ground_truth: torch.Tensor,
        reconstruction: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        step: int = 0
    ):
        """
        Log side-by-side comparison of images.
        
        Args:
            tag: Name for the comparison
            ground_truth: (B, C, H, W) ground truth images
            reconstruction: (B, C, H, W) reconstructed images
            prediction: (B, C, H, W) predicted images (optional)
            step: Global step
        """
        import torchvision
        
        # Take first 4 samples
        gt = ground_truth[:4]
        recon = reconstruction[:4]
        
        # Normalize
        if gt.min() < 0:
            gt = (gt + 1) / 2
            recon = (recon + 1) / 2
        
        # Create comparison
        if prediction is not None:
            pred = prediction[:4]
            if pred.min() < 0:
                pred = (pred + 1) / 2
            
            # Interleave: GT, Recon, Pred
            comparison = torch.stack([gt, recon, pred], dim=1).flatten(0, 1)
        else:
            # Interleave: GT, Recon
            comparison = torch.stack([gt, recon], dim=1).flatten(0, 1)
        
        grid = torchvision.utils.make_grid(comparison, nrow=4, padding=2)
        self.writer.add_image(f'{tag}/comparison', grid, step)
    
    def log_text(self, tag: str, text: str, step: int = 0):
        """
        Log text.
        
        Args:
            tag: Name of the text
            text: Text string
            step: Global step
        """
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and metrics for comparison.
        
        Args:
            hparams: Dict of hyperparameters
            metrics: Dict of metrics
        """
        self.writer.add_hparams(hparams, metrics)
    
    def log_embedding(
        self,
        tag: str,
        embeddings: torch.Tensor,
        metadata: Optional[list] = None,
        step: int = 0
    ):
        """
        Log embeddings for visualization in TensorBoard.
        
        Args:
            tag: Name for the embeddings
            embeddings: (N, D) tensor of embeddings
            metadata: List of labels for each embedding
            step: Global step
        """
        self.writer.add_embedding(embeddings, metadata=metadata, tag=tag, global_step=step)
    
    def close(self):
        """Close the writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_logger(log_dir: str = './runs', experiment_name: str = 'world_model') -> TensorBoardLogger:
    """
    Convenience function to create a TensorBoard logger.
    
    Args:
        log_dir: Base directory for logs
        experiment_name: Name of the experiment
    
    Returns:
        TensorBoardLogger instance
    """
    return TensorBoardLogger(log_dir, experiment_name)


if __name__ == '__main__':
    # Test logger
    logger = TensorBoardLogger(experiment_name='test_run')
    
    # Test scalar logging
    for step in range(100):
        logger.log_scalar('train/loss', 1.0 / (step + 1), step)
        logger.log_scalar('train/accuracy', step / 100.0, step)
    
    # Test image logging
    dummy_images = torch.randn(8, 3, 64, 64)
    logger.log_images('test/images', dummy_images, step=0)
    
    # Test comparison
    gt = torch.randn(4, 3, 64, 64)
    recon = torch.randn(4, 3, 64, 64)
    pred = torch.randn(4, 3, 64, 64)
    logger.log_comparison('test/comparison', gt, recon, pred, step=0)
    
    # Test hyperparameters
    hparams = {'lr': 0.001, 'batch_size': 16}
    metrics = {'accuracy': 0.95, 'loss': 0.05}
    logger.log_hyperparameters(hparams, metrics)
    
    logger.close()
    
    print("Test complete! Check logs with: tensorboard --logdir=./runs")