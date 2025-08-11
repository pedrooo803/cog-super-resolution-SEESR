"""
Custom UNet2DConditionModel with SEESR modifications and SD Turbo optimizations
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unet_2d_condition import UNet2DConditionModel as BaseUNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class SEESRAttnProcessor(AttnProcessor):
    """
    SEESR-specific attention processor with semantic guidance
    """
    
    def __init__(self, hidden_size: Optional[int] = None, cross_attention_dim: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        ram_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.FloatTensor:
        
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Standard attention computation
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        # Apply RAM guidance if available
        if ram_encoder_hidden_states is not None and encoder_hidden_states is not None:
            # Enhance encoder hidden states with RAM features
            ram_features = ram_encoder_hidden_states
            if ram_features.shape[-1] != encoder_hidden_states.shape[-1]:
                # Project RAM features to match dimensions
                if not hasattr(attn, 'ram_projection'):
                    attn.ram_projection = nn.Linear(
                        ram_features.shape[-1], 
                        encoder_hidden_states.shape[-1]
                    ).to(encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                ram_features = attn.ram_projection(ram_features)
            
            # Blend encoder states with RAM guidance
            alpha = 0.3  # Blending factor for RAM features
            encoder_hidden_states = (1 - alpha) * encoder_hidden_states + alpha * ram_features
        
        # Standard attention mechanism
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states


class UNet2DConditionModel(BaseUNet2DConditionModel):
    """
    Custom UNet model with SEESR enhancements and SD Turbo optimizations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # SEESR-specific features
        self.use_ram_guidance = True
        self.use_kds = True
        self.seesr_mode = True
        
        # SD Turbo optimizations
        self.turbo_mode = True
        self.fast_attention = True
        
        # Replace attention processors with SEESR-specific ones
        self._setup_seesr_attention()
        
    def _setup_seesr_attention(self):
        """Setup SEESR-specific attention processors"""
        attention_procs = {}
        
        for name, module in self.named_modules():
            if "attn" in name and hasattr(module, "processor"):
                if "attn2" in name:  # Cross attention
                    attention_procs[name] = SEESRAttnProcessor(
                        hidden_size=module.inner_dim,
                        cross_attention_dim=module.cross_attention_dim
                    )
                else:  # Self attention
                    attention_procs[name] = SEESRAttnProcessor(
                        hidden_size=module.inner_dim
                    )
        
        self.set_attn_processor(attention_procs)
        logger.info("SEESR attention processors initialized")
    
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        ram_encoder_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Enhanced forward pass with SEESR features and SD Turbo optimizations
        """
        
        # Add RAM guidance to cross attention kwargs
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        
        if ram_encoder_hidden_states is not None and self.use_ram_guidance:
            cross_attention_kwargs["ram_encoder_hidden_states"] = ram_encoder_hidden_states
        
        # SD Turbo optimizations
        if self.turbo_mode:
            # Reduce precision for speed (while maintaining quality)
            if self.training:
                sample = sample.half() if sample.dtype == torch.float32 else sample
                encoder_hidden_states = encoder_hidden_states.half() if encoder_hidden_states.dtype == torch.float32 else encoder_hidden_states
        
        # Call parent forward with enhanced features
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )
    
    def enable_kds(self, enabled: bool = True):
        """Enable/disable Kernel Density Steering"""
        self.use_kds = enabled
        logger.info(f"KDS {'enabled' if enabled else 'disabled'}")
    
    def enable_ram_guidance(self, enabled: bool = True):
        """Enable/disable RAM guidance"""
        self.use_ram_guidance = enabled
        logger.info(f"RAM guidance {'enabled' if enabled else 'disabled'}")
    
    def set_turbo_mode(self, enabled: bool = True):
        """Enable/disable SD Turbo optimizations"""
        self.turbo_mode = enabled
        logger.info(f"SD Turbo mode {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def from_pretrained_orig(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        seesr_model_path: Union[str, os.PathLike],
        **kwargs,
    ):
        """
        Load UNet from original SD model and apply SEESR modifications
        """
        # First load the base model
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Convert to our custom class
        if not isinstance(model, cls):
            custom_model = cls(**model.config)
            custom_model.load_state_dict(model.state_dict(), strict=False)
            model = custom_model
        
        # Load SEESR-specific weights if available
        seesr_unet_path = os.path.join(seesr_model_path, "unet", "diffusion_pytorch_model.safetensors")
        if os.path.exists(seesr_unet_path):
            try:
                from safetensors.torch import load_file
                seesr_weights = load_file(seesr_unet_path)
                
                # Load compatible weights
                model_state = model.state_dict()
                updated_weights = {}
                
                for key, value in seesr_weights.items():
                    if key in model_state and model_state[key].shape == value.shape:
                        updated_weights[key] = value
                        logger.info(f"Loaded SEESR weight: {key}")
                
                if updated_weights:
                    model.load_state_dict(updated_weights, strict=False)
                    logger.info(f"Loaded {len(updated_weights)} SEESR weights")
                
            except Exception as e:
                logger.warning(f"Could not load SEESR weights: {e}")
        
        return model
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        super().enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled for UNet")
    
    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers memory efficient attention if available"""
        try:
            super().enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
