"""
Model checkpoint saving utilities.
"""

import torch
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
import json


class CheckpointSaver:
    """
    Utility for saving and managing model checkpoints.
    
    Features:
        - Save best checkpoints based on metrics
        - Save periodic checkpoints
        - Automatic cleanup of old checkpoints
        - Resume training from checkpoint
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        keep_top_k: int = 3,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_top_k: Keep only top K checkpoints
            monitor: Metric to monitor for best checkpoint
            mode: 'min' or 'max' for monitored metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_top_k = keep_top_k
        self.monitor = monitor
        self.mode = mode
        
        # Track best checkpoints
        self.best_checkpoints = []
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        checkpoint_name: Optional[str] = None
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            step: Global step
            metrics: Dict of metrics
            config: Configuration dict
            checkpoint_name: Custom checkpoint name (optional)
        
        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f'checkpoint_epoch{epoch:03d}.pt'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save checkpoint if it's one of the best.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            step: Global step
            metrics: Dict of metrics
            config: Configuration dict
        
        Returns:
            Path to saved checkpoint if saved, None otherwise
        """
        if self.monitor not in metrics:
            print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return None
        
        current_metric = metrics[self.monitor]
        
        # Check if this is a best checkpoint
        is_best = False
        if self.mode == 'min':
            is_best = current_metric < self.best_metric
        else:
            is_best = current_metric > self.best_metric
        
        if not is_best and len(self.best_checkpoints) >= self.keep_top_k:
            return None
        
        # Save checkpoint
        checkpoint_name = f'best_epoch{epoch:03d}_{self.monitor}{current_metric:.4f}.pt'
        checkpoint_path = self.save(
            model, optimizer, epoch, step, metrics, config, checkpoint_name
        )
        
        # Update best metric
        if is_best:
            self.best_metric = current_metric
            
            # Save as "best.pt" as well
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            shutil.copy(checkpoint_path, best_path)
            print(f"New best checkpoint! Metric: {current_metric:.4f}")
        
        # Track checkpoint
        self.best_checkpoints.append({
            'path': checkpoint_path,
            'metric': current_metric,
            'epoch': epoch
        })
        
        # Sort by metric
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x['metric'])
        else:
            self.best_checkpoints.sort(key=lambda x: -x['metric'])
        
        # Remove excess checkpoints
        if len(self.best_checkpoints) > self.keep_top_k:
            to_remove = self.best_checkpoints[self.keep_top_k:]
            for item in to_remove:
                path = Path(item['path'])
                if path.exists():
                    path.unlink()
                    print(f"Removed old checkpoint: {path.name}")
            
            self.best_checkpoints = self.best_checkpoints[:self.keep_top_k]
        
        return checkpoint_path
    
    def save_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None
    ) -> str:
        """
        Save latest checkpoint (overwrites previous).
        
        Useful for resuming training.
        """
        return self.save(
            model, optimizer, epoch, step, metrics, config,
            checkpoint_name='checkpoint_latest.pt'
        )
    
    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / 'checkpoint_best.pt'
        return str(best_path) if best_path.exists() else None
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        return str(latest_path) if latest_path.exists() else None
    
    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        return [str(p) for p in self.checkpoint_dir.glob('*.pt')]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    **kwargs
):
    """
    Simple checkpoint saving function.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Saved checkpoint: {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to load tensors to
    
    Returns:
        Dict with additional checkpoint data
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from: {filepath}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


if __name__ == '__main__':
    # Test checkpoint saver
    from torch import nn
    
    # Create dummy model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create saver
    saver = CheckpointSaver(checkpoint_dir='./test_checkpoints', keep_top_k=3)
    
    # Simulate training
    for epoch in range(10):
        metrics = {'val_loss': 1.0 / (epoch + 1), 'val_acc': epoch / 10}
        
        # Save latest
        saver.save_latest(model, optimizer, epoch, epoch * 100, metrics)
        
        # Try to save best
        saver.save_best(model, optimizer, epoch, epoch * 100, metrics)
    
    print("\nBest checkpoints:")
    for item in saver.best_checkpoints:
        print(f"  {item['path']} - metric: {item['metric']:.4f}")
    
    # Test loading
    best_path = saver.get_best_checkpoint_path()
    if best_path:
        checkpoint = load_checkpoint(best_path, model, optimizer)
        print(f"\nLoaded best checkpoint from epoch {checkpoint['epoch']}")
    
    # Cleanup
    import shutil
    shutil.rmtree('./test_checkpoints')
    print("\nTest complete!")