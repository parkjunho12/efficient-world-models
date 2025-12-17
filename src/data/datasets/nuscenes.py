"""
nuScenes dataset loader for world modeling.

Supports both preprocessed data and direct nuScenes API loading.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, List, Any
import json

from .base import BaseDataset

__all__ = ["NuScenesDataset", "NuScenesRawDataset"]


class NuScenesDataset(BaseDataset):
    """
    nuScenes dataset for world modeling.
    
    Loads preprocessed nuScenes data organized as:
        data_root/
            train/
                scene_001/
                    images/
                        000.jpg, 001.jpg, ...
                    actions.npy
                    metadata.json
                scene_002/
                    ...
            val/
                ...
    
    Features:
        - Efficient loading of preprocessed sequences
        - Configurable sequence length
        - Optional data augmentation
        - Metadata support
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 10,
        split: str = 'train',
        transform: Optional[Any] = None,
        frame_skip: int = 1,
        cache_images: bool = False
    ):
        """
        Args:
            data_root: Root directory of preprocessed data
            sequence_length: Number of frames per sequence
            split: 'train' or 'val'
            transform: Image transformation function
            frame_skip: Skip frames for temporal downsampling
            cache_images: Cache images in memory (faster but uses more RAM)
        """
        self.transform = transform
        self.frame_skip = frame_skip
        self.cache_images = cache_images
        self._image_cache = {} if cache_images else None
        
        super().__init__(data_root, sequence_length, split)
        
        print(f"NuScenesDataset initialized:")
        print(f"  Split: {split}")
        print(f"  Sequences: {len(self.sequences)}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Frame skip: {frame_skip}")
    
    def _load_sequences(self) -> List[Dict]:
        """
        Load all sequence metadata.
        
        Returns:
            List of sequence dicts with keys: 'path', 'scene_id', 'num_frames'
        """
        sequences = []
        scene_dir = self.data_root / self.split
        
        if not scene_dir.exists():
            raise ValueError(f"Scene directory not found: {scene_dir}")
        
        for seq_path in sorted(scene_dir.glob('*')):
            if not seq_path.is_dir():
                continue
            
            # Check if valid sequence
            images_dir = seq_path / 'images'
            actions_file = seq_path / 'actions.npy'
            
            if not images_dir.exists() or not actions_file.exists():
                print(f"Warning: Skipping invalid sequence: {seq_path.name}")
                continue
            
            # Count frames
            num_frames = len(list(images_dir.glob('*.jpg')))
            
            if num_frames < self.sequence_length:
                continue
            
            # Load metadata if available
            metadata_file = seq_path / 'metadata.json'
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            sequences.append({
                'path': seq_path,
                'scene_id': seq_path.name,
                'num_frames': num_frames,
                'metadata': metadata
            })
        
        return sequences
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """
        Load image with optional caching.
        
        Args:
            image_path: Path to image file
        
        Returns:
            PIL Image
        """
        if self.cache_images:
            cache_key = str(image_path)
            if cache_key not in self._image_cache:
                self._image_cache[cache_key] = Image.open(image_path).convert('RGB')
            return self._image_cache[cache_key].copy()
        else:
            return Image.open(image_path).convert('RGB')
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sequence sample.
        
        Args:
            idx: Sequence index
        
        Returns:
            Dict with keys:
                - 'images': (T, C, H, W) tensor
                - 'actions': (T-1, 4) tensor
                - 'scene_id': str
                - 'metadata': dict (optional)
        """
        seq = self.sequences[idx]
        seq_path = seq['path']
        images_dir = seq_path / 'images'
        
        # Get image paths
        all_image_paths = sorted(images_dir.glob('*.jpg'))
        
        # Apply frame skip
        if self.frame_skip > 1:
            all_image_paths = all_image_paths[::self.frame_skip]
        
        # Select sequence
        if len(all_image_paths) > self.sequence_length:
            # Random start position for training augmentation
            if self.split == 'train':
                max_start = len(all_image_paths) - self.sequence_length
                start_idx = np.random.randint(0, max_start + 1)
            else:
                start_idx = 0
            
            image_paths = all_image_paths[start_idx:start_idx + self.sequence_length]
        else:
            image_paths = all_image_paths[:self.sequence_length]
        
        # Load images
        images = []
        for img_path in image_paths:
            img = self._load_image(img_path)
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
        
        images = torch.stack(images)
        
        # Load actions
        actions = np.load(seq_path / 'actions.npy')
        actions = torch.from_numpy(actions).float()
        
        # Match actions to selected frames
        if self.frame_skip > 1 or len(image_paths) < len(actions):
            # Subsample actions to match frames
            action_indices = list(range(len(image_paths) - 1))
            actions = actions[action_indices]
        
        # Ensure correct shape
        if len(actions) != len(images) - 1:
            # Pad or truncate actions
            if len(actions) < len(images) - 1:
                padding = torch.zeros(len(images) - 1 - len(actions), actions.shape[-1])
                actions = torch.cat([actions, padding], dim=0)
            else:
                actions = actions[:len(images) - 1]
        
        result = {
            'images': images,
            'actions': actions,
            'scene_id': seq['scene_id']
        }
        
        # Add metadata if available
        if seq['metadata']:
            result['metadata'] = seq['metadata']
        
        return result
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def get_scene_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata for a sequence.
        
        Args:
            idx: Sequence index
        
        Returns:
            Dict with scene information
        """
        return self.sequences[idx]


class NuScenesRawDataset(BaseDataset):
    """
    Load nuScenes data directly from nuScenes API.
    
    Requires nuscenes-devkit:
        pip install nuscenes-devkit
    
    Usage:
        dataset = NuScenesRawDataset(
            data_root='/data/nuscenes',
            version='v1.0-mini',
            sequence_length=10
        )
    """
    
    def __init__(
        self,
        data_root: str,
        version: str = 'v1.0-mini',
        sequence_length: int = 10,
        split: str = 'train',
        transform: Optional[Any] = None,
        camera: str = 'CAM_FRONT'
    ):
        """
        Args:
            data_root: Path to nuScenes root directory
            version: nuScenes version ('v1.0-mini', 'v1.0-trainval')
            sequence_length: Number of frames per sequence
            split: 'train' or 'val'
            transform: Image transformation
            camera: Camera sensor to use
        """
        try:
            from nuscenes.nuscenes import NuScenes
            self.NuScenes = NuScenes
        except ImportError:
            raise ImportError(
                "nuscenes-devkit not installed. "
                "Install with: pip install nuscenes-devkit"
            )
        
        self.version = version
        self.transform = transform
        self.camera = camera
        
        # Load nuScenes
        print(f"Loading nuScenes {version}...")
        self.nusc = self.NuScenes(version=version, dataroot=data_root, verbose=False)
        print(f"✓ Loaded {len(self.nusc.scene)} scenes")
        
        super().__init__(data_root, sequence_length, split)
    
    def _load_sequences(self) -> List[Dict]:
        """
        Extract sequences from nuScenes scenes.
        
        Returns:
            List of sequence dicts
        """
        sequences = []
        
        # Split scenes into train/val (80/20)
        num_train = int(len(self.nusc.scene) * 0.8)
        
        if self.split == 'train':
            scenes = self.nusc.scene[:num_train]
        else:
            scenes = self.nusc.scene[num_train:]
        
        for scene in scenes:
            # Get samples for this scene
            sample_token = scene['first_sample_token']
            samples = []
            
            while sample_token:
                sample = self.nusc.get('sample', sample_token)
                samples.append(sample)
                sample_token = sample['next']
            
            # Create sequences with sliding window
            for i in range(len(samples) - self.sequence_length + 1):
                seq_samples = samples[i:i + self.sequence_length]
                
                sequences.append({
                    'scene_name': scene['name'],
                    'scene_token': scene['token'],
                    'samples': seq_samples,
                    'start_idx': i
                })
        
        return sequences
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sequence from nuScenes."""
        seq = self.sequences[idx]
        samples = seq['samples']
        
        images = []
        actions = []
        
        for sample in samples:
            # Get camera image
            cam_token = sample['data'][self.camera]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            # Load image
            img_path = self.data_root / cam_data['filename']
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            images.append(img)
            
            # Get vehicle state (simplified - would need ego vehicle data)
            # For now, create dummy actions
            action = np.random.randn(4).astype(np.float32)
            actions.append(action)
        
        images = torch.stack(images)
        actions = torch.from_numpy(np.array(actions[:-1])).float()
        
        return {
            'images': images,
            'actions': actions,
            'scene_id': seq['scene_name']
        }


if __name__ == '__main__':
    # Test dataset
    from pathlib import Path
    
    # Test preprocessed dataset
    if Path('data/processed/nuscenes/train').exists():
        print("Testing NuScenesDataset...")
        dataset = NuScenesDataset(
            data_root='data/processed/nuscenes',
            sequence_length=10,
            split='train'
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        sample = dataset[0]
        print(f"Sample shapes:")
        print(f"  Images: {sample['images'].shape}")
        print(f"  Actions: {sample['actions'].shape}")
        print(f"  Scene ID: {sample['scene_id']}")
    
    # Test raw dataset
    if Path('/data/nuscenes').exists():
        print("\nTesting NuScenesRawDataset...")
        dataset = NuScenesRawDataset(
            data_root='/data/nuscenes',
            version='v1.0-mini',
            sequence_length=10
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Sample shapes:")
        print(f"  Images: {sample['images'].shape}")
        print(f"  Actions: {sample['actions'].shape}")
    
    print("\n✓ Dataset tests complete!")