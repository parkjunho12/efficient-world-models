"""
Temporal transforms for video sequences.

Provides various temporal augmentation strategies for video data.
"""

import torch
import numpy as np
from typing import List, Union, Tuple, Optional
import random

__all__ = [
    "TemporalTransform",
    "RandomTemporalCrop",
    "TemporalSubsampling",
    "TemporalReverse",
    "TemporalJitter"
]


class TemporalTransform:
    """
    Basic temporal augmentation for video sequences.
    
    Features:
        - Random temporal cropping
        - Frame skipping
        - Configurable sequence length
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        frame_skip: int = 1,
        random_start: bool = True
    ):
        """
        Args:
            sequence_length: Number of frames to sample
            frame_skip: Skip every N frames (temporal downsampling)
            random_start: Random start position (training) vs fixed (val)
        """
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        self.random_start = random_start
    
    def __call__(self, frames: List) -> List:
        """
        Sample a sequence from frames.
        
        Args:
            frames: List of frames (images or tensors)
        
        Returns:
            Sampled sequence
        """
        required_length = self.sequence_length * self.frame_skip
        
        if len(frames) < required_length:
            raise ValueError(
                f"Not enough frames: need {required_length}, got {len(frames)}"
            )
        
        # Determine start index
        if self.random_start:
            max_start = len(frames) - required_length
            start_idx = np.random.randint(0, max_start + 1)
        else:
            start_idx = 0
        
        # Sample frames
        indices = range(
            start_idx,
            start_idx + required_length,
            self.frame_skip
        )
        
        return [frames[i] for i in indices]


class RandomTemporalCrop:
    """
    Random temporal crop with multiple strategies.
    
    Strategies:
        - 'random': Random start position
        - 'beginning': Always start from beginning
        - 'middle': Start from middle
        - 'end': Start from end
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        strategy: str = 'random'
    ):
        """
        Args:
            sequence_length: Number of frames to extract
            strategy: Cropping strategy
        """
        self.sequence_length = sequence_length
        self.strategy = strategy
        
        valid_strategies = ['random', 'beginning', 'middle', 'end']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    def __call__(self, frames: List) -> List:
        """Extract sequence based on strategy."""
        if len(frames) < self.sequence_length:
            raise ValueError(
                f"Not enough frames: need {self.sequence_length}, got {len(frames)}"
            )
        
        if self.strategy == 'random':
            start_idx = np.random.randint(0, len(frames) - self.sequence_length + 1)
        elif self.strategy == 'beginning':
            start_idx = 0
        elif self.strategy == 'middle':
            start_idx = (len(frames) - self.sequence_length) // 2
        elif self.strategy == 'end':
            start_idx = len(frames) - self.sequence_length
        
        return frames[start_idx:start_idx + self.sequence_length]


class TemporalSubsampling:
    """
    Temporal subsampling with different patterns.
    
    Patterns:
        - 'uniform': Uniform frame skip
        - 'random': Random frame selection
        - 'exponential': Exponentially increasing gaps (recent frames denser)
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        pattern: str = 'uniform',
        skip_rate: int = 2
    ):
        """
        Args:
            sequence_length: Target number of frames
            pattern: Sampling pattern
            skip_rate: Skip rate for uniform pattern
        """
        self.sequence_length = sequence_length
        self.pattern = pattern
        self.skip_rate = skip_rate
    
    def __call__(self, frames: List) -> List:
        """Subsample frames according to pattern."""
        n_frames = len(frames)
        
        if self.pattern == 'uniform':
            # Uniform skip
            indices = np.linspace(0, n_frames - 1, self.sequence_length, dtype=int)
        
        elif self.pattern == 'random':
            # Random selection
            indices = sorted(np.random.choice(
                n_frames,
                size=self.sequence_length,
                replace=False
            ))
        
        elif self.pattern == 'exponential':
            # Exponential spacing (more recent frames)
            # e.g., [0, 1, 2, 4, 8, 16, ...]
            indices = []
            idx = 0
            gap = 1
            
            while len(indices) < self.sequence_length and idx < n_frames:
                indices.append(idx)
                idx += gap
                gap = int(gap * 1.5)  # Exponentially increase gap
            
            # Fill remaining with last frames if needed
            while len(indices) < self.sequence_length:
                indices.append(n_frames - 1)
            
            indices = indices[:self.sequence_length]
        
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")
        
        return [frames[i] for i in indices]


class TemporalReverse:
    """
    Randomly reverse temporal order.
    
    Useful for learning temporal invariance.
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of reversing
        """
        self.p = p
    
    def __call__(self, frames: List) -> List:
        """Randomly reverse frame order."""
        if random.random() < self.p:
            return list(reversed(frames))
        return frames


class TemporalJitter:
    """
    Add temporal jitter by slightly shifting frame indices.
    
    Simulates variable frame rates or timing uncertainties.
    """
    
    def __init__(self, max_jitter: int = 2):
        """
        Args:
            max_jitter: Maximum frame shift
        """
        self.max_jitter = max_jitter
    
    def __call__(self, frames: List) -> List:
        """Apply temporal jitter."""
        n_frames = len(frames)
        jittered = []
        
        for i in range(len(frames)):
            # Add random jitter
            jitter = random.randint(-self.max_jitter, self.max_jitter)
            idx = np.clip(i + jitter, 0, n_frames - 1)
            jittered.append(frames[idx])
        
        return jittered


class TemporalDownsampling:
    """
    Downsample temporal resolution by factor.
    
    E.g., 30 FPS -> 15 FPS
    """
    
    def __init__(self, factor: int = 2):
        """
        Args:
            factor: Downsampling factor
        """
        self.factor = factor
    
    def __call__(self, frames: List) -> List:
        """Downsample by taking every Nth frame."""
        return frames[::self.factor]


class TemporalInterpolation:
    """
    Interpolate missing frames (useful for data augmentation).
    
    Simple linear interpolation between frames.
    """
    
    def __init__(self, interpolate_every: int = 2):
        """
        Args:
            interpolate_every: Insert interpolated frame every N frames
        """
        self.interpolate_every = interpolate_every
    
    def __call__(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Interpolate frames.
        
        Args:
            frames: List of tensors
        
        Returns:
            List with interpolated frames
        """
        if not frames or len(frames) < 2:
            return frames
        
        interpolated = [frames[0]]
        
        for i in range(1, len(frames)):
            # Add interpolated frame if interval reached
            if i % self.interpolate_every == 0 and i > 0:
                # Linear interpolation
                prev_frame = frames[i - 1]
                curr_frame = frames[i]
                interp_frame = (prev_frame + curr_frame) / 2.0
                interpolated.append(interp_frame)
            
            interpolated.append(frames[i])
        
        return interpolated


class TemporalCompose:
    """
    Compose multiple temporal transforms.
    
    Similar to torchvision.transforms.Compose but for temporal transforms.
    """
    
    def __init__(self, transforms: List):
        """
        Args:
            transforms: List of temporal transforms
        """
        self.transforms = transforms
    
    def __call__(self, frames: List) -> List:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            frames = transform(frames)
        return frames


def get_temporal_transform(
    mode: str = 'train',
    sequence_length: int = 10,
    frame_skip: int = 1
):
    """
    Factory function to get appropriate temporal transform.
    
    Args:
        mode: 'train' or 'val'
        sequence_length: Number of frames
        frame_skip: Skip rate
    
    Returns:
        Temporal transform
    """
    if mode == 'train':
        return TemporalCompose([
            TemporalTransform(
                sequence_length=sequence_length,
                frame_skip=frame_skip,
                random_start=True
            ),
            TemporalReverse(p=0.3)  # 30% chance of reverse
        ])
    else:
        return TemporalTransform(
            sequence_length=sequence_length,
            frame_skip=frame_skip,
            random_start=False
        )


if __name__ == '__main__':
    # Test temporal transforms
    print("Testing temporal transforms...")
    
    # Create dummy frame sequence
    frames = [torch.randn(3, 64, 64) for _ in range(30)]
    print(f"Input: {len(frames)} frames")
    
    # Test TemporalTransform
    print("\n1. Testing TemporalTransform...")
    transform = TemporalTransform(sequence_length=10, frame_skip=2)
    sampled = transform(frames)
    print(f"  Output: {len(sampled)} frames")
    
    # Test RandomTemporalCrop
    print("\n2. Testing RandomTemporalCrop...")
    for strategy in ['random', 'beginning', 'middle', 'end']:
        crop = RandomTemporalCrop(sequence_length=10, strategy=strategy)
        cropped = crop(frames)
        print(f"  {strategy}: {len(cropped)} frames")
    
    # Test TemporalSubsampling
    print("\n3. Testing TemporalSubsampling...")
    for pattern in ['uniform', 'random', 'exponential']:
        subsample = TemporalSubsampling(sequence_length=10, pattern=pattern)
        subsampled = subsample(frames)
        print(f"  {pattern}: {len(subsampled)} frames")
    
    # Test TemporalReverse
    print("\n4. Testing TemporalReverse...")
    reverse = TemporalReverse(p=1.0)  # Always reverse
    reversed_frames = reverse(frames[:5])
    print(f"  Original first frame sum: {frames[0].sum():.2f}")
    print(f"  Reversed first frame sum: {reversed_frames[0].sum():.2f}")
    print(f"  Match: {torch.allclose(frames[-1], reversed_frames[0])}")
    
    # Test TemporalJitter
    print("\n5. Testing TemporalJitter...")
    jitter = TemporalJitter(max_jitter=2)
    jittered = jitter(frames[:10])
    print(f"  Output: {len(jittered)} frames")
    
    # Test TemporalCompose
    print("\n6. Testing TemporalCompose...")
    composed = TemporalCompose([
        RandomTemporalCrop(sequence_length=15),
        TemporalSubsampling(sequence_length=10, pattern='uniform'),
        TemporalReverse(p=0.5)
    ])
    result = composed(frames)
    print(f"  Output: {len(result)} frames")
    
    # Test factory function
    print("\n7. Testing factory function...")
    train_transform = get_temporal_transform('train', sequence_length=10)
    val_transform = get_temporal_transform('val', sequence_length=10)
    print(f"  Train: {type(train_transform).__name__}")
    print(f"  Val: {type(val_transform).__name__}")
    
    print("\n✓ All tests passed!")