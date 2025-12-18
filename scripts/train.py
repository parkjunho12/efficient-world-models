#!/usr/bin/env python3
"""Main training script."""

import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.world_model import build_world_model
from training.trainer import Trainer
from training.losses import WorldModelLoss
from training.optimizer import build_optimizer
from data.datasets.nuscenes import NuScenesDataset
from data.transforms.spatial import SpatialTransform
from torch.utils.data import DataLoader
from utils.config.parser import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training/base.yaml')
    parser.add_argument('--data-root', type=str, default='data/processed/nuscenes')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-epochs', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    
    # Build model
    model = build_world_model(config['model'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Data
    transform = SpatialTransform(
        image_size=tuple(config['data']['image_size']),
        augment=True
    )
    
    train_dataset = NuScenesDataset(
        data_root=args.data_root,
        sequence_length=config['data']['sequence_length'],
        transform=transform,
        split='train'
    )
    
    val_dataset = NuScenesDataset(
        data_root=args.data_root,
        sequence_length=config['data']['sequence_length'],
        transform=transform,
        split='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Loss and optimizer
    loss_fn = WorldModelLoss(config['loss'])
    optimizer = build_optimizer(model, config['optimizer'])
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        config=config['training']
    )
    
    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])

if __name__ == '__main__':
    main()