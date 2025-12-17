"""
Model checkpoint loading utilities.
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any


class CheckpointLoader:
    """
    Utility for loading model checkpoints with various options.
    
    Features:
        - Load full checkpoint or weights only
        - Handle missing keys gracefully
        - Support for partial loading
        - Device management
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: Device to load tensors to ('cpu', 'cuda', etc.)
        """
        self.device = device
    
    def load(
        self,
        filepath: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            strict: Whether to strictly enforce key matching
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Dict containing checkpoint metadata (epoch, metrics, etc.)
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(
                checkpoint['model_state_dict'],
                strict=strict
            )
            
            if missing:
                print(f"⚠️  Missing keys: {missing}")
            if unexpected:
                print(f"⚠️  Unexpected keys: {unexpected}")
        else:
            # Checkpoint is just state dict
            model.load_state_dict(checkpoint, strict=strict)
        
        # Load optimizer state
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️  Failed to load optimizer state: {e}")
        
        # Print info
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Loaded checkpoint: {filepath}")
        print(f"  Epoch: {epoch}")
        
        if 'metrics' in checkpoint:
            print(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def load_weights_only(
        self,
        filepath: str,
        model: torch.nn.Module,
        strict: bool = True
    ):
        """
        Load only model weights (no optimizer state).
        
        Useful for inference or transfer learning.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=strict)
        print(f"✓ Loaded weights from: {filepath}")
    
    def load_partial(
        self,
        filepath: str,
        model: torch.nn.Module,
        prefix: Optional[str] = None,
        exclude_keys: Optional[list] = None
    ):
        """
        Load partial checkpoint (e.g., only encoder).
        
        Args:
            filepath: Path to checkpoint
            model: Model to load into
            prefix: Load only keys starting with prefix (e.g., 'encoder.')
            exclude_keys: List of key patterns to exclude
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter keys
        filtered_dict = {}
        for key, value in state_dict.items():
            # Check prefix
            if prefix and not key.startswith(prefix):
                continue
            
            # Check exclusions
            if exclude_keys and any(excl in key for excl in exclude_keys):
                continue
            
            filtered_dict[key] = value
        
        # Load filtered state dict
        missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
        
        print(f"✓ Loaded partial checkpoint: {filepath}")
        print(f"  Loaded {len(filtered_dict)} keys")
        if missing:
            print(f"  Missing {len(missing)} keys")
        if unexpected:
            print(f"  Unexpected {len(unexpected)} keys")
    
    def transfer_weights(
        self,
        filepath: str,
        model: torch.nn.Module,
        layer_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Transfer weights with layer name mapping.
        
        Useful when loading from a model with different layer names.
        
        Args:
            filepath: Path to checkpoint
            model: Target model
            layer_mapping: Dict mapping source layer names to target layer names
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            source_dict = checkpoint['model_state_dict']
        else:
            source_dict = checkpoint
        
        # Get target model state dict
        target_dict = model.state_dict()
        
        # Transfer weights
        transferred = 0
        if layer_mapping:
            for source_key, target_key in layer_mapping.items():
                if source_key in source_dict and target_key in target_dict:
                    target_dict[target_key] = source_dict[source_key]
                    transferred += 1
        else:
            # Try direct matching
            for key in source_dict:
                if key in target_dict:
                    target_dict[key] = source_dict[key]
                    transferred += 1
        
        # Load transferred weights
        model.load_state_dict(target_dict)
        
        print(f"✓ Transferred {transferred} layers from: {filepath}")
    
    def find_latest_checkpoint(self, checkpoint_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in a directory.
        
        Args:
            checkpoint_dir: Directory to search
        
        Returns:
            Path to latest checkpoint or None
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Look for checkpoint_latest.pt first
        latest_path = checkpoint_dir / 'checkpoint_latest.pt'
        if latest_path.exists():
            return str(latest_path)
        
        # Otherwise find most recent checkpoint
        checkpoints = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            return None
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return str(checkpoints[0])
    
    def auto_resume(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically resume from latest checkpoint if available.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            model: Model to load into
            optimizer: Optimizer to load into (optional)
        
        Returns:
            Checkpoint dict if resumed, None otherwise
        """
        latest_checkpoint = self.find_latest_checkpoint(checkpoint_dir)
        
        if latest_checkpoint:
            print(f"Found checkpoint for resuming: {latest_checkpoint}")
            return self.load(latest_checkpoint, model, optimizer)
        else:
            print("No checkpoint found - starting from scratch")
            return None


def load_pretrained(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = 'cpu',
    strict: bool = False
):
    """
    Convenience function to load pretrained weights.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        device: Device to load to
        strict: Whether to strictly match keys
    """
    loader = CheckpointLoader(device)
    loader.load_weights_only(checkpoint_path, model, strict=strict)


def auto_resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str = './checkpoints',
    device: str = 'cpu'
) -> Optional[int]:
    """
    Convenience function to auto-resume training.
    
    Args:
        model: Model
        optimizer: Optimizer
        checkpoint_dir: Directory with checkpoints
        device: Device
    
    Returns:
        Starting epoch (0 if no checkpoint, checkpoint epoch + 1 if resumed)
    """
    loader = CheckpointLoader(device)
    checkpoint = loader.auto_resume(checkpoint_dir, model, optimizer)
    
    if checkpoint:
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resuming training from epoch {start_epoch}")
        return start_epoch
    else:
        return 0


if __name__ == '__main__':
    # Test checkpoint loader
    from torch import nn
    import tempfile
    import os
    
    # Create dummy model and checkpoint
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create temporary checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        
        # Save checkpoint
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': {'loss': 0.5, 'accuracy': 0.9}
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Test loading
        loader = CheckpointLoader(device='cpu')
        
        # Load full checkpoint
        loaded = loader.load(checkpoint_path, model, optimizer)
        print(f"Loaded epoch: {loaded['epoch']}")
        
        # Load weights only
        new_model = nn.Linear(10, 5)
        loader.load_weights_only(checkpoint_path, new_model)
        
        # Test auto-resume
        start_epoch = auto_resume_training(model, optimizer, tmpdir)
        print(f"Start epoch: {start_epoch}")
    
    print("\nTest complete!")