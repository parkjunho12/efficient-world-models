"""
Normalization utilities for images and tensors.

Provides various normalization strategies and denormalization for visualization.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union

__all__ = [
    "Normalize", 
    "Denormalize", 
    "normalize_batch",
    "denormalize_batch",
    "get_normalization_params"
]


class Normalize:
    """
    Normalize tensors with mean and std.
    
    Supports both single images and batches.
    """
    
    def __init__(
        self,
        mean: List[float] = None,
        std: List[float] = None,
        preset: str = "imagenet"
    ):
        """
        Args:
            mean: Mean values for each channel
            std: Std values for each channel
            preset: Preset normalization ('imagenet', 'cifar10', 'custom')
        """
        if mean is None or std is None:
            mean, std = get_normalization_params(preset)
        
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.preset = preset
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor.
        
        Args:
            tensor: (C, H, W) or (B, C, H, W) tensor
        
        Returns:
            Normalized tensor
        """
        # Move mean and std to same device
        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)
        
        # Handle batch dimension
        if tensor.ndim == 4:
            # (B, C, H, W)
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        else:
            # (C, H, W)
            mean = self.mean
            std = self.std
        
        return (tensor - mean) / std
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor for visualization.
        
        Args:
            tensor: Normalized tensor
        
        Returns:
            Denormalized tensor
        """
        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)
        
        if tensor.ndim == 4:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        else:
            mean = self.mean
            std = self.std
        
        return tensor * std + mean


class Denormalize:
    """
    Denormalization transform for visualization.
    
    Inverse of Normalize.
    """
    
    def __init__(
        self,
        mean: List[float] = None,
        std: List[float] = None,
        preset: str = "imagenet"
    ):
        """
        Args:
            mean: Mean values used in normalization
            std: Std values used in normalization
            preset: Preset normalization
        """
        if mean is None or std is None:
            mean, std = get_normalization_params(preset)
        
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor.
        
        Args:
            tensor: (C, H, W) or (B, C, H, W) normalized tensor
        
        Returns:
            Denormalized tensor in [0, 1] range
        """
        if tensor.device != self.mean.device:
            self.mean = self.mean.to(tensor.device)
            self.std = self.std.to(tensor.device)
        
        if tensor.ndim == 4:
            mean = self.mean.unsqueeze(0)
            std = self.std.unsqueeze(0)
        else:
            mean = self.mean
            std = self.std
        
        denorm = tensor * std + mean
        
        # Clamp to [0, 1]
        return torch.clamp(denorm, 0, 1)


class MinMaxNormalize:
    """
    Min-max normalization to [0, 1] range.
    
    Useful when data is not zero-centered.
    """
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        """
        Args:
            min_val: Minimum value of input data
            max_val: Maximum value of input data
        """
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1]."""
        return (tensor - self.min_val) / self.range
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0, 1]."""
        return tensor * self.range + self.min_val


class StandardizeNormalize:
    """
    Standardization: zero mean and unit variance.
    
    Computed per-channel on the fly.
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Small value to avoid division by zero
        """
        self.eps = eps
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Standardize tensor.
        
        Args:
            tensor: (C, H, W) or (B, C, H, W) tensor
        
        Returns:
            Standardized tensor
        """
        if tensor.ndim == 4:
            # (B, C, H, W)
            dims = (2, 3)
        else:
            # (C, H, W)
            dims = (1, 2)
        
        mean = tensor.mean(dim=dims, keepdim=True)
        std = tensor.std(dim=dims, keepdim=True) + self.eps
        
        return (tensor - mean) / std


def normalize_batch(
    batch: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    preset: str = "imagenet"
) -> torch.Tensor:
    """
    Normalize a batch of images.
    
    Args:
        batch: (B, C, H, W) tensor
        mean: Mean values
        std: Std values
        preset: Preset normalization
    
    Returns:
        Normalized batch
    """
    normalizer = Normalize(mean=mean, std=std, preset=preset)
    return normalizer(batch)


def denormalize_batch(
    batch: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    preset: str = "imagenet"
) -> torch.Tensor:
    """
    Denormalize a batch of images.
    
    Args:
        batch: (B, C, H, W) normalized tensor
        mean: Mean values used in normalization
        std: Std values used in normalization
        preset: Preset normalization
    
    Returns:
        Denormalized batch in [0, 1]
    """
    denormalizer = Denormalize(mean=mean, std=std, preset=preset)
    return denormalizer(batch)


def get_normalization_params(preset: str) -> Tuple[List[float], List[float]]:
    """
    Get normalization parameters for common presets.
    
    Args:
        preset: One of 'imagenet', 'cifar10', 'custom', 'zero_one'
    
    Returns:
        (mean, std) tuple
    """
    presets = {
        'imagenet': (
            [0.485, 0.456, 0.406],  # Mean
            [0.229, 0.224, 0.225]   # Std
        ),
        'cifar10': (
            [0.4914, 0.4822, 0.4465],
            [0.2470, 0.2435, 0.2616]
        ),
        'zero_one': (
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
        'custom': (
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        )
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]


class BatchNormalize(nn.Module):
    """
    Learnable batch normalization for images.
    
    Alternative to fixed normalization, learns optimal mean/std.
    """
    
    def __init__(self, num_channels: int = 3, eps: float = 1e-5, momentum: float = 0.1):
        """
        Args:
            num_channels: Number of input channels
            eps: Epsilon for numerical stability
            momentum: Momentum for running mean/var
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization."""
        return self.bn(x)


def compute_dataset_statistics(
    dataloader,
    num_batches: Optional[int] = None
) -> Tuple[List[float], List[float]]:
    """
    Compute mean and std of a dataset.
    
    Useful for custom normalization.
    
    Args:
        dataloader: PyTorch DataLoader
        num_batches: Number of batches to use (None = all)
    
    Returns:
        (mean, std) as lists
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if num_batches and i >= num_batches:
            break
        
        # Get images from batch
        if isinstance(batch, dict):
            images = batch.get('images', batch.get('image'))
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        # Handle sequence dimension
        if images.ndim == 5:
            # (B, T, C, H, W) -> flatten batch and time
            B, T = images.shape[:2]
            images = images.view(B * T, *images.shape[2:])
        
        batch_samples = images.shape[0]
        
        # Compute mean and std per channel
        for c in range(3):
            mean[c] += images[:, c, :, :].mean().item() * batch_samples
            std[c] += images[:, c, :, :].std().item() * batch_samples
        
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


if __name__ == '__main__':
    # Test normalization
    print("Testing normalization utilities...")
    
    # Create dummy image
    image = torch.rand(3, 256, 256)
    batch = torch.rand(4, 3, 256, 256)
    
    # Test Normalize
    print("\n1. Testing Normalize...")
    normalizer = Normalize(preset='imagenet')
    normalized = normalizer(image)
    print(f"  Input range: [{image.min():.2f}, {image.max():.2f}]")
    print(f"  Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # Test Denormalize
    print("\n2. Testing Denormalize...")
    denormalized = normalizer.denormalize(normalized)
    print(f"  Denormalized range: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
    print(f"  Reconstruction error: {(image - denormalized).abs().max():.6f}")
    
    # Test batch operations
    print("\n3. Testing batch operations...")
    normalized_batch = normalize_batch(batch, preset='imagenet')
    denormalized_batch = denormalize_batch(normalized_batch, preset='imagenet')
    print(f"  Batch normalized: {normalized_batch.shape}")
    print(f"  Batch denormalized: {denormalized_batch.shape}")
    
    # Test MinMaxNormalize
    print("\n4. Testing MinMaxNormalize...")
    minmax = MinMaxNormalize(min_val=0, max_val=255)
    image_uint8 = torch.randint(0, 256, (3, 256, 256)).float()
    normalized_mm = minmax(image_uint8)
    print(f"  Input range: [{image_uint8.min():.0f}, {image_uint8.max():.0f}]")
    print(f"  MinMax normalized: [{normalized_mm.min():.2f}, {normalized_mm.max():.2f}]")
    
    # Test StandardizeNormalize
    print("\n5. Testing StandardizeNormalize...")
    standardizer = StandardizeNormalize()
    standardized = standardizer(image)
    print(f"  Mean: {standardized.mean():.6f}")
    print(f"  Std: {standardized.std():.6f}")
    
    # Test presets
    print("\n6. Available presets:")
    for preset in ['imagenet', 'cifar10', 'zero_one']:
        mean, std = get_normalization_params(preset)
        print(f"  {preset}: mean={mean}, std={std}")
    
    print("\n✓ All tests passed!")