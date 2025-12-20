"""
Spatial image transforms for data augmentation.

Provides various augmentation strategies for autonomous driving images.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from typing import Tuple, Optional, List
from PIL import Image
import random

__all__ = ["SpatialTransform", "DrivingAugmentation", "MinimalTransform"]


class SpatialTransform:
    """
    Standard spatial augmentations for images.
    
    Includes:
        - Resizing
        - Random horizontal flip
        - Color jittering
        - Random cropping
        - Normalization
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        normalize: bool = True,
        mean: List[float] = None,
        std: List[float] = None
    ):
        """
        Args:
            image_size: Target image size (H, W)
            augment: Whether to apply augmentations
            normalize: Whether to normalize images
            mean: Normalization mean (default: ImageNet)
            std: Normalization std (default: ImageNet)
        """
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize
        
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet
        if std is None:
            std = [0.229, 0.224, 0.225]  # ImageNet
        
        self.mean = mean
        self.std = std
        
        # Build transform pipeline
        transforms = []
        
        # Resize
        transforms.append(T.Resize(image_size))
        
        # Augmentations
        if augment:
            transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                T.RandomCrop(image_size, padding=16, padding_mode='reflect'),
            ])
        
        # Convert to tensor
        transforms.append(T.ToTensor())
        
        # Normalize
        if normalize:
            transforms.append(T.Normalize(mean=mean, std=std))
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply transform to image.
        
        Args:
            image: PIL Image
        
        Returns:
            Transformed tensor (C, H, W)
        """
        return self.transform(image)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize tensor for visualization.
        
        Args:
            tensor: Normalized tensor (C, H, W)
        
        Returns:
            Denormalized tensor in [0, 1]
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).reshape(-1, 1, 1)
        std = torch.tensor(self.std).reshape(-1, 1, 1)
        
        return tensor * std + mean


class DrivingAugmentation:
    """
    Augmentations specifically designed for autonomous driving.
    
    Features:
        - Preserves spatial relationships important for driving
        - No vertical flips (would be unrealistic)
        - Moderate color changes
        - Optional random cropping (simulates camera movement)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        brightness: float = 0.2,
        contrast: float = 0.2
    ):
        """
        Args:
            image_size: Target size
            augment: Enable augmentations
            crop_scale: Random crop scale range
            brightness: Brightness jitter magnitude
            contrast: Contrast jitter magnitude
        """
        self.image_size = image_size
        self.augment = augment
        
        transforms = []
        
        if augment:
            # Random resized crop (simulates different viewing distances)
            transforms.append(
                T.RandomResizedCrop(
                    image_size,
                    scale=crop_scale,
                    ratio=(1.5, 2.0)  # Typical driving camera aspect
                )
            )
            
            # Color jittering (weather/lighting conditions)
            transforms.append(
                T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=0.1,
                    hue=0.05
                )
            )
            
            # Random horizontal flip (realistic for lane changes)
            # Note: We flip with low probability to maintain road structure
            transforms.append(T.RandomHorizontalFlip(p=0.3))
        else:
            transforms.append(T.Resize(image_size))
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class MinimalTransform:
    """
    Minimal transform - just resize and normalize.
    
    Use for validation/testing where augmentation is not desired.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        """
        Args:
            image_size: Target size
            normalize: Whether to normalize
        """
        transforms = [
            T.Resize(image_size),
            T.ToTensor()
        ]
        
        if normalize:
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class RandomOcclusion:
    """
    Random occlusion augmentation.
    
    Simulates occlusions (e.g., from windshield wipers, rain drops).
    """
    
    def __init__(self, p: float = 0.2, scale: Tuple[float, float] = (0.02, 0.1)):
        """
        Args:
            p: Probability of applying occlusion
            scale: Scale range for occlusion size
        """
        self.p = p
        self.scale = scale
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random occlusion to tensor image.
        
        Args:
            image: (C, H, W) tensor
        
        Returns:
            Image with random occlusion
        """
        if random.random() > self.p:
            return image
        
        C, H, W = image.shape
        
        # Random occlusion size
        scale = random.uniform(*self.scale)
        oh = int(H * scale)
        ow = int(W * scale)
        
        # Random position
        y = random.randint(0, H - oh)
        x = random.randint(0, W - ow)
        
        # Apply occlusion (fill with mean value)
        image[:, y:y+oh, x:x+ow] = 0
        
        return image


def get_transform(
    mode: str = 'train',
    image_size: Tuple[int, int] = (256, 256),
    augment: bool = True
):
    """
    Factory function to get appropriate transform.
    
    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size
        augment: Enable augmentations (only for train)
    
    Returns:
        Transform object
    """
    if mode == 'train':
        return DrivingAugmentation(
            image_size=image_size,
            augment=augment
        )
    else:
        return MinimalTransform(image_size=image_size)


if __name__ == '__main__':
    # Test transforms
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_image = Image.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    
    # Test SpatialTransform
    print("Testing SpatialTransform...")
    transform = SpatialTransform(image_size=(256, 256), augment=True)
    output = transform(dummy_image)
    print(f"  Input: PIL Image {dummy_image.size}")
    print(f"  Output: {output.shape}, range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test DrivingAugmentation
    print("\nTesting DrivingAugmentation...")
    transform = DrivingAugmentation(image_size=(256, 256), augment=True)
    output = transform(dummy_image)
    print(f"  Output: {output.shape}")
    
    # Test MinimalTransform
    print("\nTesting MinimalTransform...")
    transform = MinimalTransform(image_size=(256, 256))
    output = transform(dummy_image)
    print(f"  Output: {output.shape}")
    
    # Test factory function
    print("\nTesting factory function...")
    train_transform = get_transform('train', augment=True)
    val_transform = get_transform('val')
    print(f"  Train transform: {type(train_transform).__name__}")
    print(f"  Val transform: {type(val_transform).__name__}")
    
    print("\n✓ Transform tests complete!")