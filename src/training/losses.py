"""
Loss functions for world model training
Includes perceptual losses, reconstruction losses, and regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.cuda.amp import autocast


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pretrained VGG features
    More aligned with human perception than pixel-wise losses
    """
    
    def __init__(self, layers: list = None):
        super().__init__()
        
        # Load pretrained VGG16
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Default layers for perceptual loss
        if layers is None:
            layers = ['3', '8', '15', '22']  # relu1_2, relu2_2, relu3_3, relu4_3
        
        self.layers = layers
        
        # Extract feature layers
        self.feature_extractor = nn.ModuleDict()
        for layer_id in layers:
            self.feature_extractor[layer_id] = nn.Sequential(
                *list(vgg.children())[:int(layer_id) + 1]
            )
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        
        # ImageNet normalization as buffers (moves with .to(device))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self._moved = False  # ensure we move extractor once
    
    def _ensure_device(self, device: torch.device):
        if not self._moved:
            self.feature_extractor.to(device)
            self._moved = True
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss
        
        Args:
            pred: (B, C, H, W) predicted images
            target: (B, C, H, W) target images
        
        Returns:
            loss: scalar perceptual loss
        """
        self._ensure_device(pred.device)
        
        # Perceptual loss: force fp32 for stability + avoid AMP dtype issues
        with autocast(enabled=False):
            pred = pred.float()
            target = target.float()

            mean = self.mean.to(pred.device, dtype=pred.dtype)
            std = self.std.to(pred.device, dtype=pred.dtype)

            pred_norm = (pred - mean) / std
            target_norm = (target - mean) / std

            loss = 0.0
            for layer_id in self.layers:
                pred_feat = self.feature_extractor[layer_id](pred_norm)
                target_feat = self.feature_extractor[layer_id](target_norm)
                loss = loss + F.mse_loss(pred_feat, target_feat)

            return loss / len(self.layers)


class WorldModelLoss(nn.Module):
    """
    Combined loss function for world model training
    
    Components:
        1. Reconstruction loss (pixel + perceptual)
        2. Prediction loss (pixel + perceptual)
        3. Latent regularization
        4. Temporal consistency
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Loss weights
        self.recon_weight = config.get('recon_weight', 1.0)
        self.pred_weight = config.get('pred_weight', 1.0)
        self.perceptual_weight = config.get('perceptual_weight', 0.1)
        self.latent_reg_weight = config.get('latent_reg_weight', 0.01)
        self.temporal_weight = config.get('temporal_weight', 0.1)
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
    
    def reconstruction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Combined pixel + perceptual reconstruction loss
        """
        # Pixel-wise L1 loss (more robust than L2)
        pixel_loss = F.l1_loss(pred, target)
        
        # Perceptual loss
        if self.perceptual_weight > 0:
            # Flatten batch and time dimensions
            if pred.ndim == 5:
                B, T, C, H, W = pred.shape
                pred_flat = pred.view(B * T, C, H, W)
                target_flat = target.view(B * T, C, H, W)
            else:
                pred_flat = pred
                target_flat = target
            
            perceptual = self.perceptual_loss(pred_flat, target_flat)
            return pixel_loss + self.perceptual_weight * perceptual
        
        return pixel_loss
    
    def prediction_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Prediction loss with temporal weighting
        Later predictions are typically harder, so we can weight them more
        """
        if pred.ndim == 5:
            B, T, C, H, W = pred.shape
            
            # Compute per-timestep loss
            losses = []
            for t in range(T):
                pred_t = pred[:, t]
                target_t = target[:, t]
                loss_t = self.reconstruction_loss(pred_t, target_t)
                
                # Weight later predictions more (optional)
                weight = 1.0 + (t / T) * self.temporal_weight
                losses.append(loss_t * weight)
            
            return torch.stack(losses).mean()
        else:
            return self.reconstruction_loss(pred, target)
    
    def latent_regularization(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Regularize latent space to prevent collapse
        Uses simple L2 regularization
        """
        return torch.mean(latents ** 2)
    
    def temporal_consistency_loss(
        self,
        latents: torch.Tensor,
        predicted_latents: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage smooth transitions in latent space
        
        Args:
            latents: (B, T, D) ground truth latent sequence
            predicted_latents: (B, T-1, D) predicted latent sequence
        """
        # Compare predicted latents with ground truth next latents
        target_latents = latents[:, 1:]
        return F.mse_loss(predicted_latents, target_latents)
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        predicted: torch.Tensor,
        images: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        predicted_latents: Optional[torch.Tensor] = None,
        loss_type: str = 'full'
    ) -> torch.Tensor:
        """
        Compute total loss
        
        Args:
            reconstructed: (B, T, C, H, W) reconstructed images
            predicted: (B, T-1, C, H, W) predicted images
            images: (B, T, C, H, W) ground truth images
            latents: (B, T, D) latent states (optional)
            predicted_latents: (B, T-1, D) predicted latents (optional)
            loss_type: 'reconstruction', 'prediction', or 'full'
        
        Returns:
            loss: scalar loss value
        """
        if loss_type == 'reconstruction':
            return self.reconstruction_loss(reconstructed, images) * self.recon_weight
        
        elif loss_type == 'prediction':
            return self.prediction_loss(predicted, images[:, 1:]) * self.pred_weight
        
        else:  # 'full'
            # Reconstruction loss
            recon_loss = self.reconstruction_loss(reconstructed, images)
            
            # Prediction loss
            pred_loss = self.prediction_loss(predicted, images[:, 1:])
            
            total_loss = (
                self.recon_weight * recon_loss +
                self.pred_weight * pred_loss
            )
            
            # Optional: latent regularization
            if latents is not None and self.latent_reg_weight > 0:
                latent_reg = self.latent_regularization(latents)
                total_loss += self.latent_reg_weight * latent_reg
            
            # Optional: temporal consistency
            if (latents is not None and predicted_latents is not None 
                and self.temporal_weight > 0):
                temp_loss = self.temporal_consistency_loss(latents, predicted_latents)
                total_loss += self.temporal_weight * temp_loss
            
            return total_loss


class AdversarialLoss(nn.Module):
    """
    Optional GAN-style adversarial loss for sharper predictions
    Not recommended for initial training
    """
    
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        """Loss for generator (want discriminator to classify as real)"""
        fake_pred = self.discriminator(fake_images)
        real_labels = torch.ones_like(fake_pred)
        return self.bce_loss(fake_pred, real_labels)
    
    def discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> torch.Tensor:
        """Loss for discriminator"""
        # Real images
        real_pred = self.discriminator(real_images)
        real_labels = torch.ones_like(real_pred)
        real_loss = self.bce_loss(real_pred, real_labels)
        
        # Fake images
        fake_pred = self.discriminator(fake_images.detach())
        fake_labels = torch.zeros_like(fake_pred)
        fake_loss = self.bce_loss(fake_pred, fake_labels)
        
        return (real_loss + fake_loss) / 2


if __name__ == '__main__':
    # Quick test
    config = {
        'recon_weight': 1.0,
        'pred_weight': 1.0,
        'perceptual_weight': 0.1,
        'latent_reg_weight': 0.01,
        'temporal_weight': 0.1
    }
    
    loss_fn = WorldModelLoss(config)
    
    # Test data
    B, T, C, H, W = 4, 5, 3, 256, 256
    reconstructed = torch.randn(B, T, C, H, W)
    predicted = torch.randn(B, T-1, C, H, W)
    images = torch.randn(B, T, C, H, W)
    latents = torch.randn(B, T, 256)
    predicted_latents = torch.randn(B, T-1, 256)
    
    # Test losses
    recon_loss = loss_fn(reconstructed, predicted, images, loss_type='reconstruction')
    pred_loss = loss_fn(reconstructed, predicted, images, loss_type='prediction')
    full_loss = loss_fn(reconstructed, predicted, images, latents, predicted_latents, loss_type='full')
    
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Prediction loss: {pred_loss.item():.4f}")
    print(f"Full loss: {full_loss.item():.4f}")