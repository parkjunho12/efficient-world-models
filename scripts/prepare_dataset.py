#!/usr/bin/env python3
"""
Preprocess datasets for world model training.

Converts raw datasets to a uniform format for efficient training.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_nuscenes(data_root: Path, output_dir: Path, sequence_length: int = 10):
    """
    Prepare nuScenes dataset.
    
    Args:
        data_root: Path to raw nuScenes data
        output_dir: Output directory
        sequence_length: Length of sequences to create
    """
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("✗ Error: nuscenes-devkit not installed")
        print("Install with: pip install nuscenes-devkit")
        return False
    
    print("\n" + "=" * 80)
    print("Preparing nuScenes Dataset")
    print("=" * 80)
    
    # Load nuScenes
    print(f"Loading from: {data_root}")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=str(data_root), verbose=False)
    except:
        try:
            nusc = NuScenes(version='v1.0-trainval', dataroot=str(data_root), verbose=False)
        except Exception as e:
            print(f"✗ Error loading nuScenes: {e}")
            return False
    
    print(f"✓ Loaded {len(nusc.scene)} scenes")
    
    # Process each scene
    train_sequences = []
    val_sequences = []
    
    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        # Get samples for this scene
        sample_token = scene['first_sample_token']
        samples = []
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']
        
        if len(samples) < sequence_length:
            continue
        
        # Split train/val (80/20)
        is_train = len(train_sequences) / max(len(train_sequences) + len(val_sequences), 1) < 0.8
        split = 'train' if is_train else 'val'
        
        # Create sequences with sliding window
        for i in range(len(samples) - sequence_length + 1):
            seq_samples = samples[i:i + sequence_length]
            
            # Create sequence directory
            seq_name = f"scene_{scene['name']}_seq_{i:04d}"
            seq_dir = output_dir / split / seq_name
            seq_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images
            images_dir = seq_dir / 'images'
            images_dir.mkdir(exist_ok=True)
            
            for j, sample in enumerate(seq_samples):
                # Get front camera image
                cam_token = sample['data']['CAM_FRONT']
                cam_data = nusc.get('sample_data', cam_token)
                
                # Copy image
                img_path = data_root / cam_data['filename']
                if img_path.exists():
                    img = Image.open(img_path)
                    img = img.resize((256, 256))
                    img.save(images_dir / f'frame_{j:03d}.jpg')
            
            # Create dummy actions (you'll need to compute these from vehicle state)
            actions = np.random.randn(sequence_length - 1, 4).astype(np.float32)
            np.save(seq_dir / 'actions.npy', actions)
            
            # Save metadata
            metadata = {
                'scene_name': scene['name'],
                'scene_token': scene['token'],
                'sequence_idx': i,
                'num_frames': sequence_length
            }
            
            with open(seq_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if is_train:
                train_sequences.append(seq_name)
            else:
                val_sequences.append(seq_name)
    
    # Save dataset info
    dataset_info = {
        'dataset': 'nuscenes',
        'version': nusc.version,
        'sequence_length': sequence_length,
        'num_train': len(train_sequences),
        'num_val': len(val_sequences),
        'image_size': [256, 256]
    }
    
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ Dataset prepared!")
    print("=" * 80)
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences: {len(val_sequences)}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    return True


def prepare_carla(data_root: Path, output_dir: Path):
    """
    Prepare CARLA dataset.
    
    Args:
        data_root: Path to CARLA episodes
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Preparing CARLA Dataset")
    print("=" * 80)
    
    episodes_dir = data_root / 'episodes'
    if not episodes_dir.exists():
        print(f"✗ Error: {episodes_dir} not found")
        return False
    
    # Process episodes
    train_sequences = []
    val_sequences = []
    
    for ep_dir in tqdm(sorted(episodes_dir.glob('episode_*')), desc="Processing episodes"):
        # Split train/val
        is_train = len(train_sequences) / max(len(train_sequences) + len(val_sequences), 1) < 0.8
        split = 'train' if is_train else 'val'
        
        # Copy to output
        seq_name = ep_dir.name
        output_ep_dir = output_dir / split / seq_name
        output_ep_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        if (ep_dir / 'images').exists():
            import shutil
            shutil.copytree(ep_dir / 'images', output_ep_dir / 'images', dirs_exist_ok=True)
        
        # Copy actions
        if (ep_dir / 'actions.npy').exists():
            import shutil
            shutil.copy(ep_dir / 'actions.npy', output_ep_dir / 'actions.npy')
        
        if is_train:
            train_sequences.append(seq_name)
        else:
            val_sequences.append(seq_name)
    
    print(f"\n✓ Processed {len(train_sequences)} train + {len(val_sequences)} val episodes")
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['nuscenes', 'carla', 'waymo'],
        help='Dataset type'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Path to raw dataset'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: data/processed/{dataset})'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Sequence length (default: 10)'
    )
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"✗ Error: {data_root} does not exist")
        return 1
    
    output_dir = Path(args.output) if args.output else Path(f'data/processed/{args.dataset}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset == 'nuscenes':
        success = prepare_nuscenes(data_root, output_dir, args.sequence_length)
    elif args.dataset == 'carla':
        success = prepare_carla(data_root, output_dir)
    elif args.dataset == 'waymo':
        print("✗ Waymo preparation not yet implemented")
        success = False
    else:
        print(f"✗ Unknown dataset: {args.dataset}")
        success = False
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())