"""
Wavelet-based color correction utilities for SEESR super-resolution
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from typing import Union, Tuple, Optional
import pywt


def wavelet_color_fix(
    enhanced_img: Union[Image.Image, np.ndarray, torch.Tensor],
    source_img: Union[Image.Image, np.ndarray, torch.Tensor],
    wavelet: str = 'db4',
    levels: int = 3,
    alpha: float = 0.7
) -> Image.Image:
    """
    Apply wavelet-based color correction to match source image colors
    
    Args:
        enhanced_img: Super-resolved/enhanced image
        source_img: Original source image for color reference
        wavelet: Wavelet type for decomposition
        levels: Number of decomposition levels
        alpha: Blending factor for color correction
        
    Returns:
        Color-corrected PIL Image
    """
    # Convert inputs to numpy arrays
    enhanced_np = _to_numpy(enhanced_img)
    source_np = _to_numpy(source_img)
    
    # Resize source to match enhanced if needed
    if enhanced_np.shape[:2] != source_np.shape[:2]:
        source_np = cv2.resize(source_np, (enhanced_np.shape[1], enhanced_np.shape[0]))
    
    # Convert to LAB color space for better color processing
    enhanced_lab = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2LAB)
    source_lab = cv2.cvtColor(source_np, cv2.COLOR_RGB2LAB)
    
    # Process each channel
    corrected_lab = enhanced_lab.copy()
    
    for i in range(3):  # L, A, B channels
        enhanced_channel = enhanced_lab[:, :, i].astype(np.float32)
        source_channel = source_lab[:, :, i].astype(np.float32)
        
        # Apply wavelet color correction
        corrected_channel = _wavelet_color_correction(
            enhanced_channel, source_channel, wavelet, levels, alpha
        )
        
        corrected_lab[:, :, i] = np.clip(corrected_channel, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(corrected_rgb)


def _wavelet_color_correction(
    enhanced: np.ndarray,
    source: np.ndarray,
    wavelet: str,
    levels: int,
    alpha: float
) -> np.ndarray:
    """
    Apply wavelet-based color correction to a single channel
    """
    try:
        # Wavelet decomposition of both images
        enhanced_coeffs = pywt.wavedec2(enhanced, wavelet, levels=levels)
        source_coeffs = pywt.wavedec2(source, wavelet, levels=levels)
        
        # Correct coefficients
        corrected_coeffs = []
        
        for i, (enh_coeff, src_coeff) in enumerate(zip(enhanced_coeffs, source_coeffs)):
            if i == 0:  # Approximation coefficients (low frequency)
                # Strong correction for low frequencies (global color)
                corrected_coeff = alpha * src_coeff + (1 - alpha) * enh_coeff
            else:  # Detail coefficients (high frequency)
                # Keep enhanced details, apply mild color correction
                detail_alpha = alpha * 0.3  # Reduce alpha for details
                corrected_coeff = tuple(
                    detail_alpha * src_detail + (1 - detail_alpha) * enh_detail
                    for enh_detail, src_detail in zip(enh_coeff, src_coeff)
                )
            
            corrected_coeffs.append(corrected_coeff)
        
        # Reconstruct the corrected image
        corrected = pywt.waverec2(corrected_coeffs, wavelet)
        
        return corrected
        
    except Exception as e:
        print(f"Wavelet correction failed: {e}, using fallback method")
        # Fallback to simple histogram matching
        return _histogram_match(enhanced, source, alpha)


def _histogram_match(enhanced: np.ndarray, source: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fallback histogram matching color correction
    """
    # Calculate histograms
    enhanced_hist, _ = np.histogram(enhanced.flatten(), 256, [0, 256])
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    
    # Calculate CDFs
    enhanced_cdf = enhanced_hist.cumsum()
    source_cdf = source_hist.cumsum()
    
    # Normalize CDFs
    enhanced_cdf = enhanced_cdf / enhanced_cdf[-1]
    source_cdf = source_cdf / source_cdf[-1]
    
    # Create lookup table
    lut = np.zeros(256)
    for i in range(256):
        # Find closest value in source CDF
        closest_idx = np.argmin(np.abs(source_cdf - enhanced_cdf[i]))
        lut[i] = closest_idx
    
    # Apply lookup table
    matched = lut[enhanced.astype(np.uint8)]
    
    # Blend with original
    corrected = alpha * matched + (1 - alpha) * enhanced
    
    return corrected


def adain_color_fix(
    enhanced_img: Union[Image.Image, np.ndarray, torch.Tensor],
    source_img: Union[Image.Image, np.ndarray, torch.Tensor],
    alpha: float = 0.6
) -> Image.Image:
    """
    Apply Adaptive Instance Normalization (AdaIN) color correction
    
    Args:
        enhanced_img: Super-resolved/enhanced image
        source_img: Original source image for color reference
        alpha: Blending factor for color correction
        
    Returns:
        Color-corrected PIL Image
    """
    # Convert to tensors
    enhanced_tensor = _to_tensor(enhanced_img)
    source_tensor = _to_tensor(source_img)
    
    # Resize source to match enhanced if needed
    if enhanced_tensor.shape[-2:] != source_tensor.shape[-2:]:
        source_tensor = F.interpolate(
            source_tensor.unsqueeze(0), 
            size=enhanced_tensor.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    # Apply AdaIN
    corrected_tensor = _adain(enhanced_tensor, source_tensor, alpha)
    
    # Convert back to PIL Image
    corrected_np = corrected_tensor.permute(1, 2, 0).cpu().numpy()
    corrected_np = np.clip(corrected_np * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(corrected_np)


def _adain(content: torch.Tensor, style: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Adaptive Instance Normalization
    """
    content_mean = content.view(content.size(0), -1).mean(dim=1, keepdim=True)
    content_std = content.view(content.size(0), -1).std(dim=1, keepdim=True)
    
    style_mean = style.view(style.size(0), -1).mean(dim=1, keepdim=True)
    style_std = style.view(style.size(0), -1).std(dim=1, keepdim=True)
    
    # Normalize content
    normalized_content = (content.view(content.size(0), -1) - content_mean) / (content_std + 1e-8)
    
    # Apply style statistics
    stylized_content = normalized_content * style_std + style_mean
    
    # Reshape back
    stylized_content = stylized_content.view_as(content)
    
    # Blend with original
    return alpha * stylized_content + (1 - alpha) * content


def luminance_color_fix(
    enhanced_img: Union[Image.Image, np.ndarray, torch.Tensor],
    source_img: Union[Image.Image, np.ndarray, torch.Tensor],
    alpha: float = 0.8
) -> Image.Image:
    """
    Apply luminance-preserving color correction
    
    Args:
        enhanced_img: Super-resolved/enhanced image
        source_img: Original source image for color reference
        alpha: Blending factor for color correction
        
    Returns:
        Color-corrected PIL Image
    """
    # Convert inputs to numpy arrays
    enhanced_np = _to_numpy(enhanced_img)
    source_np = _to_numpy(source_img)
    
    # Resize source to match enhanced if needed
    if enhanced_np.shape[:2] != source_np.shape[:2]:
        source_np = cv2.resize(source_np, (enhanced_np.shape[1], enhanced_np.shape[0]))
    
    # Convert to YUV color space
    enhanced_yuv = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2YUV)
    source_yuv = cv2.cvtColor(source_np, cv2.COLOR_RGB2YUV)
    
    # Keep enhanced luminance (Y), blend chrominance (UV)
    corrected_yuv = enhanced_yuv.copy()
    corrected_yuv[:, :, 1] = alpha * source_yuv[:, :, 1] + (1 - alpha) * enhanced_yuv[:, :, 1]  # U
    corrected_yuv[:, :, 2] = alpha * source_yuv[:, :, 2] + (1 - alpha) * enhanced_yuv[:, :, 2]  # V
    
    # Convert back to RGB
    corrected_rgb = cv2.cvtColor(corrected_yuv, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(corrected_rgb)


def _to_numpy(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert image to numpy array"""
    if isinstance(img, Image.Image):
        return np.array(img)
    elif isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.shape[0] == 3:  # CHW format
            img = img.permute(1, 2, 0)
        return img.cpu().numpy()
    elif isinstance(img, np.ndarray):
        return img
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


def _to_tensor(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert image to tensor"""
    if isinstance(img, Image.Image):
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1)
    elif isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:  # HWC format
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)
    elif isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0)
        return img
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


def apply_color_correction(
    enhanced_img: Union[Image.Image, np.ndarray, torch.Tensor],
    source_img: Union[Image.Image, np.ndarray, torch.Tensor],
    method: str = "wavelet",
    alpha: float = 0.7,
    **kwargs
) -> Image.Image:
    """
    Apply color correction using specified method
    
    Args:
        enhanced_img: Super-resolved/enhanced image
        source_img: Original source image for color reference
        method: Color correction method ('wavelet', 'adain', 'luminance')
        alpha: Blending factor for color correction
        **kwargs: Additional method-specific arguments
        
    Returns:
        Color-corrected PIL Image
    """
    if method == "wavelet":
        return wavelet_color_fix(enhanced_img, source_img, alpha=alpha, **kwargs)
    elif method == "adain":
        return adain_color_fix(enhanced_img, source_img, alpha=alpha)
    elif method == "luminance":
        return luminance_color_fix(enhanced_img, source_img, alpha=alpha)
    else:
        raise ValueError(f"Unknown color correction method: {method}")


# For backward compatibility
def wavelet_color_fix_simple(enhanced_img, source_img):
    """Simple wavelet color fix with default parameters"""
    try:
        return wavelet_color_fix(enhanced_img, source_img)
    except Exception as e:
        print(f"Wavelet color fix failed: {e}, returning enhanced image")
        if isinstance(enhanced_img, Image.Image):
            return enhanced_img
        else:
            enhanced_np = _to_numpy(enhanced_img)
            return Image.fromarray(enhanced_np)
