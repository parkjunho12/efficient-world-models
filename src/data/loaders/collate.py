"""
Custom collate functions for DataLoader.

Handles batching of sequences with variable lengths and proper padding.
"""

import torch
import numpy as np
from typing import List, Dict, Any


def world_model_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for world model data.
    
    Handles:
        - Stacking images into sequences
        - Padding variable-length sequences
        - Properly formatting actions
    
    Args:
        batch: List of samples, each containing:
            - 'images': (T, C, H, W) tensor or list of images
            - 'actions': (T-1, A) tensor
            - 'scene_id': string identifier (optional)
    
    Returns:
        dict with:
            - 'images': (B, T, C, H, W)
            - 'actions': (B, T-1, A)
            - 'scene_ids': List[str] (if present)
            - 'sequence_lengths': (B,) actual lengths before padding
    """
    
    # Extract components
    images_list = [item['images'] for item in batch]
    actions_list = [item['actions'] for item in batch]
    scene_ids = [item.get('scene_id', '') for item in batch]
    
    # Get sequence lengths
    sequence_lengths = [len(imgs) if isinstance(imgs, list) else imgs.shape[0] 
                       for imgs in images_list]
    max_length = max(sequence_lengths)
    
    # Stack images
    if isinstance(images_list[0], list):
        # Convert list of images to tensor
        images_batch = []
        for imgs in images_list:
            if isinstance(imgs[0], torch.Tensor):
                imgs_tensor = torch.stack(imgs)
            else:
                imgs_tensor = torch.stack([torch.from_numpy(img) for img in imgs])
            images_batch.append(imgs_tensor)
    else:
        images_batch = images_list
    
    # Pad sequences if needed
    if len(set(sequence_lengths)) > 1:
        # Variable length sequences - need padding
        B = len(batch)
        C, H, W = images_batch[0].shape[1:]
        
        padded_images = torch.zeros(B, max_length, C, H, W)
        padded_actions = torch.zeros(B, max_length - 1, actions_list[0].shape[-1])
        
        for i, (imgs, acts, seq_len) in enumerate(zip(images_batch, actions_list, sequence_lengths)):
            padded_images[i, :seq_len] = imgs
            padded_actions[i, :seq_len-1] = acts
        
        images_batch = padded_images
        actions_batch = padded_actions
    else:
        # All same length - simple stack
        images_batch = torch.stack(images_batch)
        actions_batch = torch.stack(actions_list)
    
    return {
        'images': images_batch,
        'actions': actions_batch,
        'scene_ids': scene_ids,
        'sequence_lengths': torch.tensor(sequence_lengths)
    }


def video_collate(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Simple collate for video data (just images).
    
    Args:
        batch: List of (T, C, H, W) tensors
    
    Returns:
        (B, T, C, H, W) tensor
    """
    return torch.stack(batch)


def temporal_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that handles temporal augmentation.
    
    Useful when applying random temporal sampling during training.
    """
    
    images = torch.stack([item['images'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    
    batch_dict = {
        'images': images,
        'actions': actions
    }
    
    # Optional: add metadata
    if 'metadata' in batch[0]:
        batch_dict['metadata'] = [item['metadata'] for item in batch]
    
    # Optional: add frame indices (useful for debugging)
    if 'frame_indices' in batch[0]:
        batch_dict['frame_indices'] = torch.stack([item['frame_indices'] for item in batch])
    
    return batch_dict


class SequenceSampler:
    """
    Custom sampler for sequential data.
    
    Ensures that sequences are sampled in order within episodes,
    which can improve training stability.
    """
    
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by episode/scene
        self.episode_indices = self._group_by_episode()
    
    def _group_by_episode(self) -> List[List[int]]:
        """Group dataset indices by episode"""
        episodes = {}
        
        for idx in range(len(self.dataset)):
            # Assume dataset has a method to get episode ID
            if hasattr(self.dataset, 'get_episode_id'):
                episode_id = self.dataset.get_episode_id(idx)
            else:
                # Fallback: use scene_id from the data
                episode_id = self.dataset.sequences[idx].get('scene_id', 'default')
            
            if episode_id not in episodes:
                episodes[episode_id] = []
            episodes[episode_id].append(idx)
        
        return list(episodes.values())
    
    def __iter__(self):
        # Shuffle episodes (but keep sequences within episodes in order)
        if self.shuffle:
            episode_order = torch.randperm(len(self.episode_indices)).tolist()
        else:
            episode_order = list(range(len(self.episode_indices)))
        
        # Flatten to get all indices in shuffled episode order
        all_indices = []
        for ep_idx in episode_order:
            all_indices.extend(self.episode_indices[ep_idx])
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            batch_indices = all_indices[i:i + self.batch_size]
            yield batch_indices
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    collate_fn=None
):
    """
    Convenience function to create DataLoader with best practices.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function (default: world_model_collate)
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    if collate_fn is None:
        collate_fn = world_model_collate
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,  # Drop incomplete batches
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )


if __name__ == '__main__':
    # Test collate function
    
    # Create dummy batch
    batch = [
        {
            'images': torch.randn(10, 3, 256, 256),
            'actions': torch.randn(9, 4),
            'scene_id': 'scene_001'
        },
        {
            'images': torch.randn(10, 3, 256, 256),
            'actions': torch.randn(9, 4),
            'scene_id': 'scene_002'
        },
        {
            'images': torch.randn(8, 3, 256, 256),  # Different length
            'actions': torch.randn(7, 4),
            'scene_id': 'scene_003'
        }
    ]
    
    # Test collate
    collated = world_model_collate(batch)
    
    print("Collated batch:")
    print(f"  Images: {collated['images'].shape}")
    print(f"  Actions: {collated['actions'].shape}")
    print(f"  Scene IDs: {collated['scene_ids']}")
    print(f"  Sequence lengths: {collated['sequence_lengths']}")