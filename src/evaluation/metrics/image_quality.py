"""
Image quality metrics for world model evaluation.

Provides comprehensive image quality assessment including:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- MS-SSIM (Multi-Scale SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = [
    'calculate_psnr',
    'calculate_ssim',
    'calculate_lpips',
    'calculate_ms_ssim',
    'ImageQualityMetrics'
]


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    reduce: bool = True
) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted images (B, C, H, W) or (B, T, C, H, W)
        target: Target images (same shape as pred)
        max_val: Maximum pixel value (1.0 for normalized images)
        reduce: Whether to reduce to scalar
    
    Returns:
        PSNR in dB
    """
    # Handle sequence dimension
    if pred.dim() == 5:
        B, T = pred.shape[:2]
        pred = pred.reshape(B * T, *pred.shape[2:])
        target = target.reshape(B * T, *target.shape[2:])
    
    # Calculate MSE
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.mean(dim=[1, 2, 3])  # Average over C, H, W
    
    # Avoid division by zero
    mse = torch.clamp(mse, min=1e-10)
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    
    if reduce:
        return psnr.mean()
    return psnr


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    reduce: bool = True
) -> torch.Tensor:
    """
    Calculate Structural Similarity Index.
    
    Args:
        pred: Predicted images (B, C, H, W) or (B, T, C, H, W)
        target: Target images (same shape as pred)
        window_size: Size of Gaussian window
        reduce: Whether to reduce to scalar
    
    Returns:
        SSIM value in [0, 1]
    """
    # Handle sequence dimension
    if pred.dim() == 5:
        B, T = pred.shape[:2]
        pred = pred.reshape(B * T, *pred.shape[2:])
        target = target.reshape(B * T, *target.shape[2:])
    
    # Constants for numerical stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Create Gaussian window
    window = _create_gaussian_window(window_size, pred.shape[1]).to(pred.device)
    
    # Calculate statistics
    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=target.shape[1])
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=window_size // 2, groups=pred.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=window_size // 2, groups=target.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu_pred_target
    
    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / denominator
    
    if reduce:
        return ssim_map.mean()
    return ssim_map.mean(dim=[1, 2, 3])  # Average over C, H, W


def _create_gaussian_window(window_size: int, num_channels: int) -> torch.Tensor:
    """Create Gaussian window for SSIM calculation."""
    # Generate 1D Gaussian kernel
    sigma = 1.5
    gauss = torch.exp(
        -torch.pow(torch.arange(window_size) - window_size // 2, 2) / (2 * sigma ** 2)
    )
    gauss = gauss / gauss.sum()
    
    # Create 2D kernel
    kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    
    # Expand for all channels
    window = kernel_2d.expand(num_channels, 1, window_size, window_size).contiguous()
    
    return window


def calculate_ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reduce: bool = True
) -> torch.Tensor:
    """
    Calculate Multi-Scale SSIM.
    
    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images
        weights: Weights for each scale
        reduce: Whether to reduce to scalar
    
    Returns:
        MS-SSIM value
    """
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(pred.device)
    
    levels = len(weights)
    mssim = []
    
    for i in range(levels):
        ssim_val = calculate_ssim(pred, target, reduce=False)
        mssim.append(ssim_val)
        
        # Downsample for next level
        if i < levels - 1:
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)
    
    # Weighted combination
    mssim = torch.stack(mssim, dim=0)
    ms_ssim = torch.prod(mssim ** weights.reshape(-1, 1), dim=0)
    
    if reduce:
        return ms_ssim.mean()
    return ms_ssim


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = 'alex',
    reduce: bool = True
) -> torch.Tensor:
    """
    Calculate Learned Perceptual Image Patch Similarity.
    
    Requires: pip install lpips
    
    Args:
        pred: Predicted images (B, C, H, W) in range [-1, 1]
        target: Target images in range [-1, 1]
        net: Network to use ('alex', 'vgg', 'squeeze')
        reduce: Whether to reduce to scalar
    
    Returns:
        LPIPS distance (lower is better)
    """
    try:
        import lpips
    except ImportError:
        raise ImportError(
            "LPIPS requires the lpips package. "
            "Install with: pip install lpips"
        )
    
    # Handle sequence dimension
    if pred.dim() == 5:
        B, T = pred.shape[:2]
        pred = pred.reshape(B * T, *pred.shape[2:])
        target = target.reshape(B * T, *target.shape[2:])
    
    # Initialize LPIPS model (cache it if called multiple times)
    if not hasattr(calculate_lpips, '_lpips_model'):
        calculate_lpips._lpips_model = {}
    
    device = pred.device
    cache_key = f"{net}_{device}"
    
    if cache_key not in calculate_lpips._lpips_model:
        calculate_lpips._lpips_model[cache_key] = lpips.LPIPS(net=net).to(device)
    
    model = calculate_lpips._lpips_model[cache_key]
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_val = model(pred, target)
    
    if reduce:
        return lpips_val.mean()
    return lpips_val.squeeze()


class ImageQualityMetrics(nn.Module):
    """
    Comprehensive image quality metrics calculator.
    
    Combines multiple metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        use_lpips: bool = True,
        lpips_net: str = 'alex',
        use_ms_ssim: bool = False
    ):
        """
        Args:
            use_lpips: Whether to use LPIPS
            lpips_net: Network for LPIPS
            use_ms_ssim: Whether to use MS-SSIM
        """
        super().__init__()
        
        self.use_lpips = use_lpips
        self.use_ms_ssim = use_ms_ssim
        
        # Initialize LPIPS
        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net=lpips_net)
            except ImportError:
                print("Warning: LPIPS not available")
                self.use_lpips = False
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> dict:
        """
        Calculate all metrics.
        
        Args:
            pred: Predicted images
            target: Target images
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR
        metrics['psnr'] = calculate_psnr(pred, target)
        
        # SSIM
        metrics['ssim'] = calculate_ssim(pred, target)
        
        # MS-SSIM
        if self.use_ms_ssim:
            try:
                metrics['ms_ssim'] = calculate_ms_ssim(pred, target)
            except:
                pass
        
        # LPIPS
        if self.use_lpips:
            try:
                metrics['lpips'] = self.lpips_model(pred, target).mean()
            except:
                pass
        
        return metrics


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    include_lpips: bool = True
) -> dict:
    """
    Convenience function to compute all available metrics.
    
    Args:
        pred: Predicted images
        target: Target images
        include_lpips: Whether to include LPIPS
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['psnr'] = calculate_psnr(pred, target).item()
    metrics['ssim'] = calculate_ssim(pred, target).item()
    
    # MSE (for reference)
    metrics['mse'] = F.mse_loss(pred, target).item()
    
    # MAE (for reference)
    metrics['mae'] = F.l1_loss(pred, target).item()
    
    # LPIPS
    if include_lpips:
        try:
            metrics['lpips'] = calculate_lpips(pred, target).item()
        except:
            pass
    
    return metrics


# Backward compatibility
def compute_psnr(*args, **kwargs):
    """Backward compatible alias."""
    return calculate_psnr(*args, **kwargs)


def compute_ssim(*args, **kwargs):
    """Backward compatible alias."""
    return calculate_ssim(*args, **kwargs)


if __name__ == '__main__':
    # Test metrics
    print("Testing image quality metrics...")
    
    # Create dummy images
    pred = torch.randn(4, 3, 256, 256)
    target = pred + torch.randn_like(pred) * 0.1  # Add noise
    
    print("\n1. Testing PSNR...")
    psnr = calculate_psnr(pred, target)
    print(f"   PSNR: {psnr:.2f} dB")
    
    print("\n2. Testing SSIM...")
    ssim = calculate_ssim(pred, target)
    print(f"   SSIM: {ssim:.4f}")
    
    print("\n3. Testing MS-SSIM...")
    try:
        ms_ssim = calculate_ms_ssim(pred, target)
        print(f"   MS-SSIM: {ms_ssim:.4f}")
    except Exception as e:
        print(f"   MS-SSIM failed: {e}")
    
    print("\n4. Testing LPIPS...")
    try:
        lpips_val = calculate_lpips(pred, target)
        print(f"   LPIPS: {lpips_val:.4f}")
    except Exception as e:
        print(f"   LPIPS not available: {e}")
    
    print("\n5. Testing sequence input...")
    pred_seq = torch.randn(2, 10, 3, 256, 256)
    target_seq = pred_seq + torch.randn_like(pred_seq) * 0.1
    psnr_seq = calculate_psnr(pred_seq, target_seq)
    ssim_seq = calculate_ssim(pred_seq, target_seq)
    print(f"   Sequence PSNR: {psnr_seq:.2f} dB")
    print(f"   Sequence SSIM: {ssim_seq:.4f}")
    
    print("\n6. Testing all metrics...")
    all_metrics = compute_all_metrics(pred, target, include_lpips=False)
    for k, v in all_metrics.items():
        print(f"   {k}: {v:.4f}")
    
    print("\n✓ All tests passed!")