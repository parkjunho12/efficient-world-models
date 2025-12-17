"""
Weights & Biases (W&B) logging utilities.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class WandBLogger:
    """
    Weights & Biases logger for experiment tracking.
    
    Features:
        - Automatic experiment tracking
        - Model checkpointing
        - Hyperparameter logging
        - Rich media logging (images, videos)
    """
    
    def __init__(
        self,
        project: str = 'world-model',
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[list] = None
    ):
        """
        Args:
            project: W&B project name
            entity: W&B team/username
            name: Run name
            config: Configuration dict
            tags: List of tags for the run
        """
        try:
            import wandb
            self.wandb = wandb
            self.enabled = True
        except ImportError:
            print("⚠️  wandb not installed. Install with: pip install wandb")
            self.enabled = False
            return
        
        # Initialize run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            tags=tags,
            resume='allow'
        )
        
        print(f"W&B logging enabled: {self.run.url}")
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics and media.
        
        Args:
            data: Dict of metrics/media to log
            step: Global step (optional)
        """
        if not self.enabled:
            return
        
        self.wandb.log(data, step=step)
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value."""
        if not self.enabled:
            return
        
        self.wandb.log({tag: value}, step=step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """
        Log an image.
        
        Args:
            tag: Name of the image
            image: (C, H, W) tensor
            step: Global step
        """
        if not self.enabled:
            return
        
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Normalize to [0, 1]
        if image.min() < 0:
            image = (image + 1) / 2
        
        # Transpose to (H, W, C) for W&B
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
        
        self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        """
        Log multiple images.
        
        Args:
            tag: Name for the images
            images: (B, C, H, W) tensor
            step: Global step
        """
        if not self.enabled:
            return
        
        wandb_images = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Normalize
            if img.min() < 0:
                img = (img + 1) / 2
            
            # Transpose
            if img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            
            wandb_images.append(self.wandb.Image(img))
        
        self.wandb.log({tag: wandb_images}, step=step)
    
    def log_video(self, tag: str, video: torch.Tensor, step: Optional[int] = None, fps: int = 10):
        """
        Log a video.
        
        Args:
            tag: Name of the video
            video: (T, C, H, W) tensor
            step: Global step
            fps: Frames per second
        """
        if not self.enabled:
            return
        
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        
        # Normalize
        if video.min() < 0:
            video = (video + 1) / 2
        
        # Transpose to (T, H, W, C)
        if video.shape[1] in [1, 3]:
            video = np.transpose(video, (0, 2, 3, 1))
        
        # Convert to uint8
        video = (video * 255).astype(np.uint8)
        
        self.wandb.log({tag: self.wandb.Video(video, fps=fps)}, step=step)
    
    def log_comparison(
        self,
        tag: str,
        ground_truth: torch.Tensor,
        reconstruction: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ):
        """
        Log side-by-side comparison.
        
        Args:
            tag: Name for the comparison
            ground_truth: Ground truth images
            reconstruction: Reconstructed images
            prediction: Predicted images (optional)
            step: Global step
        """
        if not self.enabled:
            return
        
        comparisons = []
        
        for i in range(min(4, len(ground_truth))):
            gt = ground_truth[i].cpu().numpy()
            recon = reconstruction[i].cpu().numpy()
            
            # Normalize
            if gt.min() < 0:
                gt = (gt + 1) / 2
                recon = (recon + 1) / 2
            
            # Transpose
            if gt.shape[0] in [1, 3]:
                gt = np.transpose(gt, (1, 2, 0))
                recon = np.transpose(recon, (1, 2, 0))
            
            if prediction is not None:
                pred = prediction[i].cpu().numpy()
                if pred.min() < 0:
                    pred = (pred + 1) / 2
                if pred.shape[0] in [1, 3]:
                    pred = np.transpose(pred, (1, 2, 0))
                
                # Concatenate horizontally
                comparison = np.concatenate([gt, recon, pred], axis=1)
            else:
                comparison = np.concatenate([gt, recon], axis=1)
            
            comparisons.append(self.wandb.Image(comparison, caption=f"Sample {i}"))
        
        self.wandb.log({tag: comparisons}, step=step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """
        Log a histogram.
        
        Args:
            tag: Name of the histogram
            values: Tensor of values
            step: Global step
        """
        if not self.enabled:
            return
        
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)
    
    def log_model_checkpoint(self, model_path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint to W&B.
        
        Args:
            model_path: Path to model checkpoint
            metadata: Optional metadata dict
        """
        if not self.enabled:
            return
        
        artifact = self.wandb.Artifact(
            name='model',
            type='model',
            metadata=metadata
        )
        
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)
    
    def log_table(self, tag: str, data: list, columns: list, step: Optional[int] = None):
        """
        Log a table.
        
        Args:
            tag: Name of the table
            data: List of rows
            columns: Column names
            step: Global step
        """
        if not self.enabled:
            return
        
        table = self.wandb.Table(data=data, columns=columns)
        self.wandb.log({tag: table}, step=step)
    
    def finish(self):
        """Finish the run."""
        if not self.enabled:
            return
        
        self.run.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_logger(
    project: str = 'world-model',
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict] = None
) -> WandBLogger:
    """
    Convenience function to create W&B logger.
    
    Args:
        project: W&B project name
        entity: W&B team/username
        name: Run name
        config: Configuration dict
    
    Returns:
        WandBLogger instance
    """
    return WandBLogger(project, entity, name, config)


if __name__ == '__main__':
    # Test logger
    config = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 100
    }
    
    logger = WandBLogger(project='test-project', name='test-run', config=config)
    
    if logger.enabled:
        # Test scalar logging
        for step in range(10):
            logger.log_scalar('train/loss', 1.0 / (step + 1), step=step)
        
        # Test image logging
        dummy_image = torch.randn(3, 64, 64)
        logger.log_image('test/image', dummy_image, step=0)
        
        logger.finish()
        
        print("Test complete! Check your W&B dashboard")
    else:
        print("W&B not available - skipping test")