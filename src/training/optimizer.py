"""
Optimizer utilities and advanced optimization strategies
Includes AdamW, LAMB, learning rate warmup, and gradient scaling
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List
import math


def build_optimizer(model: torch.nn.Module, config: Dict) -> optim.Optimizer:
    """
    Build optimizer with parameter groups and weight decay
    
    Features:
        - Separate weight decay for different parameter types
        - No weight decay for biases and normalization layers
        - Different learning rates for encoder, dynamics, decoder
    """
    
    # Separate parameters by type
    params_with_decay = []
    params_without_decay = []
    encoder_params = []
    dynamics_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine parameter group
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'dynamics' in name:
            dynamics_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        
        # Determine weight decay
        if 'bias' in name or 'norm' in name or 'bn' in name:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)
    
    # Learning rates
    base_lr = config['lr']
    encoder_lr = config.get('encoder_lr', base_lr)
    dynamics_lr = config.get('dynamics_lr', base_lr)
    decoder_lr = config.get('decoder_lr', base_lr)
    
    # Parameter groups
    param_groups = [
        {
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': config.get('weight_decay', 0.01)
        },
        {
            'params': dynamics_params,
            'lr': dynamics_lr,
            'weight_decay': config.get('weight_decay', 0.01)
        },
        {
            'params': decoder_params,
            'lr': decoder_lr,
            'weight_decay': config.get('weight_decay', 0.01)
        }
    ]
    
    # Build optimizer
    optimizer_type = config.get('type', 'adamw')
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            momentum=config.get('momentum', 0.9),
            nesterov=config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup
    
    Usage:
        scheduler = WarmupScheduler(optimizer, warmup_steps=1000, target_lr=1e-3)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        target_lr: float = None,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        
        # Get initial learning rates
        if target_lr is None:
            self.target_lrs = [group['lr'] for group in optimizer.param_groups]
        else:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [alpha * target_lr for target_lr in self.target_lrs]
        else:
            return self.target_lrs


class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing with warmup
    
    Popular choice for transformer training
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [alpha * base_lr for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class ReduceLROnPlateau:
    """
    Reduce learning rate when validation loss plateaus
    
    More flexible than torch's built-in version
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 1e-8,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0
    
    def step(self, metric: float):
        """Update learning rate based on metric"""
        
        # Check if metric improved
        if self.mode == 'min':
            improved = metric < self.best_value - self.threshold
        else:
            improved = metric > self.best_value + self.threshold
        
        if improved:
            self.best_value = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # Reduce learning rate if plateau
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        """Reduce learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            
            if new_lr < old_lr:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")


class GradientScaler:
    """
    Custom gradient scaler for loss scaling
    Useful when not using automatic mixed precision
    """
    
    def __init__(self, init_scale: float = 2**16, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.growth_tracker = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss before backward pass"""
        return loss * self.scale
    
    def unscale_gradients(self, optimizer: optim.Optimizer):
        """Unscale gradients after backward pass"""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale
    
    def update(self, found_inf: bool):
        """Update scale based on whether overflow occurred"""
        if found_inf:
            # Reduce scale if overflow
            self.scale *= 0.5
            self.growth_tracker = 0
        else:
            # Increase scale periodically if no overflow
            self.growth_tracker += 1
            if self.growth_tracker >= self.growth_interval:
                self.scale *= 2.0
                self.growth_tracker = 0


def get_parameter_groups(
    model: torch.nn.Module,
    config: Dict
) -> List[Dict]:
    """
    Create parameter groups with different learning rates and weight decay
    
    Useful for fine-tuning where you want different learning rates
    for different parts of the model
    """
    
    no_decay = ['bias', 'LayerNorm.weight', 'norm']
    
    # Get all parameters
    params = list(model.named_parameters())
    
    # Create parameter groups
    grouped_parameters = [
        {
            'params': [p for n, p in params 
                      if not any(nd in n for nd in no_decay) and 'encoder' in n],
            'lr': config.get('encoder_lr', config['lr']),
            'weight_decay': config.get('weight_decay', 0.01),
            'name': 'encoder_with_decay'
        },
        {
            'params': [p for n, p in params 
                      if any(nd in n for nd in no_decay) and 'encoder' in n],
            'lr': config.get('encoder_lr', config['lr']),
            'weight_decay': 0.0,
            'name': 'encoder_no_decay'
        },
        {
            'params': [p for n, p in params 
                      if not any(nd in n for nd in no_decay) and 'dynamics' in n],
            'lr': config.get('dynamics_lr', config['lr']),
            'weight_decay': config.get('weight_decay', 0.01),
            'name': 'dynamics_with_decay'
        },
        {
            'params': [p for n, p in params 
                      if any(nd in n for nd in no_decay) and 'dynamics' in n],
            'lr': config.get('dynamics_lr', config['lr']),
            'weight_decay': 0.0,
            'name': 'dynamics_no_decay'
        },
        {
            'params': [p for n, p in params 
                      if not any(nd in n for nd in no_decay) and 'decoder' in n],
            'lr': config.get('decoder_lr', config['lr']),
            'weight_decay': config.get('weight_decay', 0.01),
            'name': 'decoder_with_decay'
        },
        {
            'params': [p for n, p in params 
                      if any(nd in n for nd in no_decay) and 'decoder' in n],
            'lr': config.get('decoder_lr', config['lr']),
            'weight_decay': 0.0,
            'name': 'decoder_no_decay'
        }
    ]
    
    # Filter out empty groups
    grouped_parameters = [g for g in grouped_parameters if len(g['params']) > 0]
    
    return grouped_parameters


if __name__ == '__main__':
    # Test optimizer building
    from models.world_model import build_world_model
    
    config = {
        'model': {
            'latent_dim': 256,
            'action_dim': 4,
            'hidden_dim': 512
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'encoder_lr': 1e-5,
            'dynamics_lr': 1e-4,
            'decoder_lr': 1e-4,
            'weight_decay': 0.01
        }
    }
    
    model = build_world_model(config['model'])
    optimizer = build_optimizer(model, config['optimizer'])
    
    print("Optimizer created with parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {num_params:,} parameters, lr={group['lr']:.6f}")