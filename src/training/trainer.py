"""
Main trainer class for world model training.

Handles training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
from pathlib import Path
import time
from tqdm import tqdm

__all__ = ["Trainer"]


class Trainer:
    """
    Main trainer for world model.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - TensorBoard/W&B logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        device: torch.device = torch.device('cuda'),
        config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Training settings
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        self.grad_accum_steps = self.config.get('grad_accum_steps', 1)
        self.grad_clip = self.config.get('grad_clip', 1.0)
        
        # Logging
        self.log_interval = self.config.get('log_interval', 100)
        self.val_interval = self.config.get('val_interval', 1000)
        
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, actions, return_latents=True)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    reconstructed=outputs['reconstructed'],
                    predicted=outputs['predicted'],
                    images=images,
                    latents=outputs.get('latents'),
                    predicted_latents=outputs.get('predicted_latents')
                )
                
                loss = loss_dict['total'] / self.grad_accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Validation
            if self.global_step % self.val_interval == 0:
                val_metrics = self.validate()
                self.save_checkpoint(val_metrics)
                self.model.train()
        
        # Average losses
        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'prediction': 0.0
        }
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(images, actions, return_latents=True)
                
                loss_dict = self.loss_fn(
                    reconstructed=outputs['reconstructed'],
                    predicted=outputs['predicted'],
                    images=images,
                    latents=outputs.get('latents'),
                    predicted_latents=outputs.get('predicted_latents')
                )
            
            for k, v in loss_dict.items():
                if k in val_losses:
                    val_losses[k] += v.item()
        
        # Average
        num_batches = len(self.val_loader)
        for k in val_losses:
            val_losses[k] /= num_batches
        
        print(f"\nValidation - Loss: {val_losses['total']:.4f}")
        
        return val_losses
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint(val_metrics)
            
            print(f"Epoch {epoch + 1} - "
                  f"Train Loss: {train_metrics['total']:.4f}, "
                  f"Val Loss: {val_metrics['total']:.4f}")
    
    def save_checkpoint(self, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if metrics['total'] < self.best_val_loss:
            self.best_val_loss = metrics['total']
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['metrics']['total']
        
        print(f"✓ Loaded checkpoint from epoch {self.epoch}")