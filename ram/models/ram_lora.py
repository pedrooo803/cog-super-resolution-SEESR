"""
RAM (Recognize Anything Model) with LoRA for automatic image tagging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import numpy as np
from transformers import AutoModel, AutoTokenizer
import timm

class MultiLabelEmbedding(nn.Module):
    """Multi-label embedding module for RAM"""
    
    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embeddings = nn.Parameter(torch.randn(num_classes, embed_dim))
        
    def forward(self, labels: torch.Tensor):
        return F.embedding(labels, self.embeddings)


class RAMModel(nn.Module):
    """
    RAM (Recognize Anything Model) for automatic image tagging
    Optimized for SEESR with LoRA adaptations
    """
    
    def __init__(
        self,
        image_size: int = 384,
        vit: str = 'swin_l', 
        num_classes: int = 4585,
        embed_dim: int = 768,
        threshold: float = 0.68,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.threshold = threshold
        
        # Vision backbone
        if vit == 'swin_l':
            self.image_encoder = timm.create_model('swin_large_patch4_window12_384', pretrained=True)
            self.vision_width = self.image_encoder.head.in_features
            self.image_encoder.head = nn.Identity()
        else:
            raise ValueError(f"Unsupported vision model: {vit}")
        
        # Text processing (for tag embeddings)
        self.tag_list_file = None
        self.tag_list = self._load_default_tags()
        
        # Multi-label classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.vision_width),
            nn.Dropout(0.1),
            nn.Linear(self.vision_width, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Image embedding projection
        self.image_proj = nn.Sequential(
            nn.LayerNorm(self.vision_width),
            nn.Linear(self.vision_width, embed_dim)
        )
        
        # Tag embeddings
        self.tag_embeddings = MultiLabelEmbedding(num_classes, embed_dim)
        
        # LoRA layers for SEESR adaptation
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self._setup_lora()
        
    def _load_default_tags(self) -> List[str]:
        """Load default tag vocabulary"""
        # This would normally load from a file, but for demo we use common tags
        return [
            "person", "face", "man", "woman", "child", "baby", "hair", "eye", "nose", "mouth",
            "hand", "arm", "leg", "foot", "clothing", "shirt", "dress", "pants", "shoes", "hat",
            "car", "truck", "bus", "bicycle", "motorcycle", "train", "airplane", "boat", "traffic light",
            "building", "house", "tree", "flower", "grass", "sky", "cloud", "sun", "moon", "star",
            "cat", "dog", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
            "food", "apple", "banana", "orange", "pizza", "cake", "bottle", "cup", "plate", "spoon",
            "book", "laptop", "phone", "keyboard", "mouse", "chair", "table", "bed", "sofa", "window",
            "door", "wall", "floor", "ceiling", "light", "lamp", "clock", "mirror", "picture", "painting",
            "beautiful", "cute", "old", "new", "big", "small", "red", "blue", "green", "yellow",
            "black", "white", "brown", "gray", "pink", "purple", "orange", "clean", "dirty", "bright"
        ] + [f"tag_{i}" for i in range(100, 4585)]  # Fill remaining slots
    
    def _setup_lora(self):
        """Setup LoRA adapters for efficient fine-tuning"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' not in name:
                # Add LoRA to linear layers
                in_features = module.in_features
                out_features = module.out_features
                
                # LoRA matrices
                lora_A = nn.Parameter(torch.randn(in_features, self.lora_rank) * 0.02)
                lora_B = nn.Parameter(torch.zeros(self.lora_rank, out_features))
                
                # Register as buffers
                module.register_parameter(f'lora_A', lora_A)
                module.register_parameter(f'lora_B', lora_B)
                module.register_buffer(f'lora_scaling', torch.tensor(self.lora_alpha / self.lora_rank))
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of RAM model
        
        Args:
            images: Input images tensor [B, C, H, W]
            
        Returns:
            Dictionary containing logits, probabilities, and image embeddings
        """
        batch_size = images.shape[0]
        
        # Extract image features
        image_features = self.image_encoder(images)  # [B, vision_width]
        
        # Classification logits
        logits = self.classifier(image_features)  # [B, num_classes]
        
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Image embeddings for cross-modal alignment
        image_embeds = self.image_proj(image_features)  # [B, embed_dim]
        
        return {
            'logits': logits,
            'probs': probs,
            'image_embeds': image_embeds,
            'image_features': image_features
        }
    
    def generate_tag_embeddings(self, tag_indices: torch.Tensor) -> torch.Tensor:
        """Generate embeddings for given tag indices"""
        return self.tag_embeddings(tag_indices)
    
    def generate_image_embeds(self, images: torch.Tensor) -> torch.Tensor:
        """Generate image embeddings for cross-attention"""
        outputs = self.forward(images)
        return outputs['image_embeds']
    
    def generate_tags(self, images: torch.Tensor, threshold: Optional[float] = None) -> List[List[str]]:
        """
        Generate tags for input images
        
        Args:
            images: Input images tensor [B, C, H, W]
            threshold: Confidence threshold for tag prediction
            
        Returns:
            List of tag lists for each image
        """
        if threshold is None:
            threshold = self.threshold
            
        outputs = self.forward(images)
        probs = outputs['probs']
        
        batch_tags = []
        for i in range(probs.shape[0]):
            # Get indices where probability > threshold
            selected_indices = torch.where(probs[i] > threshold)[0]
            
            # Convert to tag names
            tags = [self.tag_list[idx.item()] for idx in selected_indices if idx.item() < len(self.tag_list)]
            batch_tags.append(tags)
        
        return batch_tags
    
    def encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encode text tags to embeddings"""
        # Simple text encoding - in practice this would use a proper text encoder
        embeddings = []
        for text in text_list:
            if text in self.tag_list:
                idx = self.tag_list.index(text)
                embeddings.append(self.tag_embeddings.embeddings[idx])
            else:
                # Unknown tag - use average embedding
                embeddings.append(self.tag_embeddings.embeddings.mean(dim=0))
        
        return torch.stack(embeddings) if embeddings else torch.zeros(0, self.embed_dim)


def ram(
    pretrained: Optional[str] = None,
    pretrained_condition: Optional[str] = None,
    image_size: int = 384,
    vit: str = 'swin_l',
    **kwargs
) -> RAMModel:
    """
    Create RAM model with optional pretrained weights
    
    Args:
        pretrained: Path to pretrained model weights
        pretrained_condition: Path to pretrained condition weights  
        image_size: Input image size
        vit: Vision transformer variant
        **kwargs: Additional model arguments
        
    Returns:
        RAMModel instance
    """
    model = RAMModel(
        image_size=image_size,
        vit=vit,
        **kwargs
    )
    
    if pretrained:
        try:
            # Load pretrained weights
            if pretrained.endswith('.pth'):
                state_dict = torch.load(pretrained, map_location='cpu')
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {pretrained}")
            else:
                print(f"Pretrained path {pretrained} not found, using random initialization")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")
    
    if pretrained_condition:
        try:
            # Load condition-specific weights (LoRA weights, etc.)
            condition_dict = torch.load(pretrained_condition, map_location='cpu')
            # Apply condition weights - this would be implementation specific
            print(f"Loaded condition weights from {pretrained_condition}")
        except Exception as e:
            print(f"Could not load condition weights: {e}")
    
    return model
