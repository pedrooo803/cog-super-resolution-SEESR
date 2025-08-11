"""
SEESR Pipeline with SD Turbo optimizations for efficient super-resolution
"""

import inspect
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline as BaseStableDiffusionControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

logger = logging.get_logger(__name__)


def kernel_density_steering(
    sample: torch.Tensor,
    particles: torch.Tensor,
    bandwidth: float = 0.1,
    num_particles: int = 10
) -> torch.Tensor:
    """
    Apply Kernel Density Steering (KDS) for enhanced generation control
    """
    batch_size = sample.shape[0]
    
    # Generate particles around the current sample
    noise = torch.randn(num_particles, *sample.shape[1:], device=sample.device, dtype=sample.dtype)
    particles = sample.unsqueeze(0) + bandwidth * noise
    
    # Compute density weights
    distances = torch.norm(particles - sample.unsqueeze(0), dim=list(range(1, len(sample.shape))), keepdim=True)
    weights = torch.exp(-distances ** 2 / (2 * bandwidth ** 2))
    weights = weights / weights.sum(dim=0, keepdim=True)
    
    # Apply weighted steering
    steered_sample = (particles * weights).sum(dim=0)
    
    return steered_sample


class StableDiffusionControlNetPipeline(BaseStableDiffusionControlNetPipeline):
    """
    Enhanced pipeline for SEESR with SD Turbo optimizations
    """
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Any = None,
        feature_extractor: CLIPImageProcessor = None,
        requires_safety_checker: bool = True,
        **kwargs,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
            **kwargs,
        )
        
        # SEESR-specific settings
        self.use_kds = True
        self.use_tiled_vae = False
        self.tiled_vae_encoder_tile_size = 1024
        self.tiled_vae_decoder_tile_size = 224
        
        # SD Turbo optimizations
        self.turbo_mode = True
        
    def _init_tiled_vae(self, encoder_tile_size: int = 1024, decoder_tile_size: int = 224):
        """Initialize tiled VAE for memory efficiency with large images"""
        self.use_tiled_vae = True
        self.tiled_vae_encoder_tile_size = encoder_tile_size
        self.tiled_vae_decoder_tile_size = decoder_tile_size
        logger.info(f"Tiled VAE initialized: encoder={encoder_tile_size}, decoder={decoder_tile_size}")
    
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
        """Encode image using tiled VAE if enabled"""
        if self.use_tiled_vae and max(image.shape[-2:]) > self.tiled_vae_encoder_tile_size:
            return self._tiled_encode(image, generator)
        else:
            if isinstance(generator, list):
                image_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator=generator[i])
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            
            image_latents = self.vae.config.scaling_factor * image_latents
            return image_latents
    
    def _decode_vae_latents(self, latents: torch.Tensor):
        """Decode latents using tiled VAE if enabled"""
        latents = 1 / self.vae.config.scaling_factor * latents
        
        if self.use_tiled_vae and max(latents.shape[-2:]) > self.tiled_vae_decoder_tile_size // 8:
            return self._tiled_decode(latents)
        else:
            image = self.vae.decode(latents).sample
            return image
    
    def _tiled_encode(self, image: torch.Tensor, generator: torch.Generator = None):
        """Encode image using tiling for memory efficiency"""
        batch_size, channels, height, width = image.shape
        tile_size = self.tiled_vae_encoder_tile_size
        
        if height <= tile_size and width <= tile_size:
            return self.vae.encode(image).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        
        # Calculate tiles
        tiles_h = math.ceil(height / tile_size)
        tiles_w = math.ceil(width / tile_size)
        
        latent_tiles = []
        for h in range(tiles_h):
            row_tiles = []
            for w in range(tiles_w):
                h_start = h * tile_size
                h_end = min((h + 1) * tile_size, height)
                w_start = w * tile_size
                w_end = min((w + 1) * tile_size, width)
                
                tile = image[:, :, h_start:h_end, w_start:w_end]
                
                # Pad tile if necessary
                if tile.shape[-2] < tile_size or tile.shape[-1] < tile_size:
                    pad_h = tile_size - tile.shape[-2]
                    pad_w = tile_size - tile.shape[-1]
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                
                tile_latent = self.vae.encode(tile).latent_dist.sample(generator=generator)
                tile_latent = self.vae.config.scaling_factor * tile_latent
                
                # Remove padding from latent
                if pad_h > 0 or pad_w > 0:
                    latent_h = tile_latent.shape[-2] - pad_h // 8
                    latent_w = tile_latent.shape[-1] - pad_w // 8
                    tile_latent = tile_latent[:, :, :latent_h, :latent_w]
                
                row_tiles.append(tile_latent)
            latent_tiles.append(torch.cat(row_tiles, dim=-1))
        
        return torch.cat(latent_tiles, dim=-2)
    
    def _tiled_decode(self, latents: torch.Tensor):
        """Decode latents using tiling for memory efficiency"""
        batch_size, channels, latent_height, latent_width = latents.shape
        tile_size = self.tiled_vae_decoder_tile_size // 8
        
        if latent_height <= tile_size and latent_width <= tile_size:
            return self.vae.decode(latents).sample
        
        # Calculate tiles
        tiles_h = math.ceil(latent_height / tile_size)
        tiles_w = math.ceil(latent_width / tile_size)
        
        image_tiles = []
        for h in range(tiles_h):
            row_tiles = []
            for w in range(tiles_w):
                h_start = h * tile_size
                h_end = min((h + 1) * tile_size, latent_height)
                w_start = w * tile_size
                w_end = min((w + 1) * tile_size, latent_width)
                
                tile_latent = latents[:, :, h_start:h_end, w_start:w_end]
                tile_image = self.vae.decode(tile_latent).sample
                
                row_tiles.append(tile_image)
            image_tiles.append(torch.cat(row_tiles, dim=-1))
        
        return torch.cat(image_tiles, dim=-2)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[
            torch.FloatTensor,
            PIL.Image.Image,
            np.ndarray,
            List[torch.FloatTensor],
            List[PIL.Image.Image],
            List[np.ndarray],
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        # SEESR-specific parameters
        start_point: str = "lr",
        start_steps: int = 999,
        ram_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        latent_tiled_size: int = 320,
        latent_tiled_overlap: int = 4,
        use_KDS: bool = True,
        bandwidth: float = 0.1,
        num_particles: int = 10,
        conditioning_scale: float = 1.0,
    ):
        """
        Enhanced SEESR pipeline call with SD Turbo optimizations
        """
        
        # 0. Default height and width to controlnet
        if isinstance(self.controlnet, list):
            controlnet = self.controlnet[0]
        else:
            controlnet = self.controlnet

        height = height or controlnet.config.sample_size * self.vae_scale_factor
        width = width or controlnet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, list) and len(controlnet) != len(image):
            raise ValueError(
                f"When passing a list of ControlNets, the number of ControlNets ({len(controlnet)}) must match the number of images ({len(image)})"
            )

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare image
        if isinstance(controlnet, (list, tuple)):
            control_images = []
            for control_image in image:
                control_image = self.prepare_image(
                    image=control_image,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                control_images.append(control_image)
            image = control_images
        else:
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet_conditioning_scale, float) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                cross_attn_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs else {}
                if ram_encoder_hidden_states is not None:
                    cross_attn_kwargs["ram_encoder_hidden_states"] = ram_encoder_hidden_states

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attn_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Apply KDS if enabled
                if use_KDS and self.use_kds:
                    latents = kernel_density_steering(latents, latents, bandwidth, num_particles)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self._decode_vae_latents(latents)
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
