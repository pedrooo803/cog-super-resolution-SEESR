"""
RAM inference functions for automatic image tagging
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def inference(
    images: torch.Tensor, 
    model: torch.nn.Module,
    threshold: float = 0.68
) -> List[str]:
    """
    Run inference on images using RAM model
    
    Args:
        images: Input image tensor [B, C, H, W] or [C, H, W]
        model: RAM model instance
        threshold: Confidence threshold for tag prediction
        
    Returns:
        List of predicted tags for the first image in batch
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    with torch.no_grad():
        # Get model predictions
        outputs = model(images)
        probs = outputs['probs']
        
        # Get tags for first image
        image_probs = probs[0]  # [num_classes]
        selected_indices = torch.where(image_probs > threshold)[0]
        
        # Convert to tag names
        if hasattr(model, 'tag_list'):
            tags = [model.tag_list[idx.item()] for idx in selected_indices if idx.item() < len(model.tag_list)]
        else:
            tags = [f"tag_{idx.item()}" for idx in selected_indices]
        
        # Join tags into a single string
        tag_string = ", ".join(tags[:20])  # Limit to top 20 tags
        
        return [tag_string]  # Return as list for compatibility


def batch_inference(
    images: torch.Tensor,
    model: torch.nn.Module, 
    threshold: float = 0.68,
    max_tags: int = 20
) -> List[List[str]]:
    """
    Run batch inference on multiple images
    
    Args:
        images: Input image tensor [B, C, H, W]
        model: RAM model instance
        threshold: Confidence threshold for tag prediction
        max_tags: Maximum number of tags per image
        
    Returns:
        List of tag lists for each image
    """
    with torch.no_grad():
        outputs = model(images)
        probs = outputs['probs']
        
        batch_tags = []
        for i in range(probs.shape[0]):
            image_probs = probs[i]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(image_probs, min(max_tags, len(image_probs)))
            
            # Filter by threshold
            valid_mask = top_probs > threshold
            valid_indices = top_indices[valid_mask]
            
            # Convert to tag names
            if hasattr(model, 'tag_list'):
                tags = [model.tag_list[idx.item()] for idx in valid_indices if idx.item() < len(model.tag_list)]
            else:
                tags = [f"tag_{idx.item()}" for idx in valid_indices]
            
            batch_tags.append(tags)
        
        return batch_tags


def preprocess_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    size: int = 384,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    Preprocess image for RAM inference
    
    Args:
        image: Input image (PIL, numpy array, or tensor)
        size: Target size for resizing
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)
        image = (image * 255).cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transform(image)


def extract_features(
    images: torch.Tensor,
    model: torch.nn.Module
) -> torch.Tensor:
    """
    Extract image features for cross-modal alignment
    
    Args:
        images: Input image tensor [B, C, H, W]
        model: RAM model instance
        
    Returns:
        Image feature embeddings [B, embed_dim]
    """
    with torch.no_grad():
        outputs = model(images)
        return outputs['image_embeds']


def generate_caption(
    tags: List[str],
    template: str = "a photo of {tags}"
) -> str:
    """
    Generate a caption from predicted tags
    
    Args:
        tags: List of predicted tags
        template: Caption template
        
    Returns:
        Generated caption
    """
    if not tags:
        return "a photo"
    
    # Join tags with commas
    tag_string = ", ".join(tags[:10])  # Use top 10 tags
    
    return template.format(tags=tag_string)


def filter_tags(
    tags: List[str],
    exclude_list: Optional[List[str]] = None,
    min_confidence: float = 0.5
) -> List[str]:
    """
    Filter and clean predicted tags
    
    Args:
        tags: List of predicted tags
        exclude_list: Tags to exclude
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered tag list
    """
    if exclude_list is None:
        exclude_list = []
    
    # Filter out excluded tags
    filtered_tags = [tag for tag in tags if tag not in exclude_list]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in filtered_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags


def enhance_prompt(
    base_prompt: str,
    predicted_tags: List[str],
    max_tags: int = 5
) -> str:
    """
    Enhance a base prompt with predicted tags
    
    Args:
        base_prompt: Base prompt text
        predicted_tags: Tags predicted by RAM
        max_tags: Maximum number of tags to add
        
    Returns:
        Enhanced prompt
    """
    if not predicted_tags:
        return base_prompt
    
    # Select top tags
    selected_tags = predicted_tags[:max_tags]
    tag_string = ", ".join(selected_tags)
    
    if base_prompt:
        return f"{base_prompt}, {tag_string}"
    else:
        return tag_string
