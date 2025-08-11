import os
import sys
import tempfile
from typing import Iterator, List
import torch
import numpy as np
from PIL import Image
import cv2
from cog import BasePredictor, Input, Path
from pytorch_lightning import seed_everything
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from huggingface_hub import hf_hub_download, snapshot_download
from torchvision import transforms
import logging

# Import custom utilities
from utils.xformers_utils import (
    is_xformers_available, 
    optimize_models_attention, 
    print_attention_status
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the SEESR model with SD Turbo for efficient super-resolution"""
        logger.info("🚀 Loading SEESR with SD Turbo models for Replicate...")
        
        # Initialize device with optimizations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Using device: {self.device} with dtype: {self.weight_dtype}")
        
        # GPU memory optimization for Replicate
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
        try:
            # Download required models
            logger.info("📥 Downloading models...")
            
            # Download SEESR model
            snapshot_download(
                repo_id="alexnasa/SEESR",
                local_dir="deployment/preset/models/seesr",
                cache_dir="/tmp/huggingface_cache"  # Use tmp for Replicate
            )
            
            # Download SD Turbo
            snapshot_download(
                repo_id="stabilityai/sd-turbo",
                local_dir="deployment/preset/models/sd-turbo",
                cache_dir="/tmp/huggingface_cache"
            )
            
            # Download RAM model for tagging
            snapshot_download(
                repo_id="xinyu1205/recognize_anything_model",
                local_dir="deployment/preset/models/ram",
                cache_dir="/tmp/huggingface_cache"
            )
            
            # Set model paths
            self.pretrained_model_path = 'deployment/preset/models/sd-turbo'
            self.seesr_model_path = 'deployment/preset/models/seesr'
            
            # Load core components
            self.scheduler = DDIMScheduler.from_pretrained(
                self.pretrained_model_path, subfolder="scheduler"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_model_path, subfolder="text_encoder"
            )
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_model_path, subfolder="tokenizer"
            )
            self.vae = AutoencoderKL.from_pretrained(
                self.pretrained_model_path, subfolder="vae"
            )
            
            # Import custom models (these would need to be included in the project)
            try:
                from models.unet_2d_condition import UNet2DConditionModel
                from models.controlnet import ControlNetModel
                from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
                from ram.models.ram_lora import ram
                from ram import inference_ram as inference
                from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
                
                # Load custom UNet and ControlNet
                self.unet = UNet2DConditionModel.from_pretrained_orig(
                    self.pretrained_model_path, self.seesr_model_path, subfolder="unet"
                )
                self.controlnet = ControlNetModel.from_pretrained(
                    self.seesr_model_path, subfolder="controlnet"
                )
                
                # Freeze models
                self.vae.requires_grad_(False)
                self.text_encoder.requires_grad_(False)
                self.unet.requires_grad_(False)
                self.controlnet.requires_grad_(False)
                
                # Move to device
                self.text_encoder.to(self.device, dtype=self.weight_dtype)
                self.vae.to(self.device, dtype=self.weight_dtype)
                self.unet.to(self.device, dtype=self.weight_dtype)
                self.controlnet.to(self.device, dtype=self.weight_dtype)
                
                # Create validation pipeline
                self.validation_pipeline = StableDiffusionControlNetPipeline(
                    vae=self.vae,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    feature_extractor=None,
                    unet=self.unet,
                    controlnet=self.controlnet,
                    scheduler=self.scheduler,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                
                # Initialize tiled VAE for memory efficiency on Replicate
                # Configurazione ottimizzata per GPU T4 (16GB) e A40 (48GB)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
                if gpu_memory < 20:  # T4 or similar
                    encoder_tile_size = 512
                    decoder_tile_size = 128
                    logger.info("🔧 Configured for T4 GPU (16GB) - reduced tile sizes")
                else:  # A40 or larger
                    encoder_tile_size = 1024
                    decoder_tile_size = 224
                    logger.info("🔧 Configured for A40+ GPU (48GB+) - standard tile sizes")
                
                self.validation_pipeline._init_tiled_vae(
                    encoder_tile_size=encoder_tile_size,
                    decoder_tile_size=decoder_tile_size
                )
                
                # Enable memory optimizations
                if hasattr(self.validation_pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        self.validation_pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("✅ xformers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable xformers: {e}")
                
                # Enable attention slicing for additional memory savings on smaller GPUs
                if gpu_memory < 20:
                    self.validation_pipeline.enable_attention_slicing(1)
                    logger.info("✅ Attention slicing enabled for memory optimization")
                
                # Apply our xformers utilities for cross-platform compatibility
                try:
                    optimize_models_attention([self.unet, self.controlnet])
                    print_attention_status()
                except Exception as e:
                    logger.warning(f"Could not apply xformers optimizations: {e}")
                
                # Load RAM model for automatic tagging
                self.tag_model = ram(
                    pretrained='deployment/preset/models/ram/ram_swin_large_14m.pth',
                    pretrained_condition='deployment/preset/models/ram/DAPE.pth',
                    image_size=384,
                    vit='swin_l'
                )
                self.tag_model.eval()
                self.tag_model.to(self.device, dtype=self.weight_dtype)
                
                # Setup transforms
                self.tensor_transforms = transforms.Compose([
                    transforms.ToTensor(),
                ])
                
                self.ram_transforms = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                self.model_loaded = True
                logger.info("SEESR with SD Turbo loaded successfully")
                
            except ImportError as e:
                logger.error(f"Could not import custom models: {e}")
                self.model_loaded = False
                # Fallback to basic SD Turbo
                self._setup_fallback()
                
        except Exception as e:
            logger.error(f"Error during model setup: {e}")
            self.model_loaded = False
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback SD Turbo pipeline"""
        try:
            from diffusers import AutoPipelineForImage2Image
            
            self.fallback_pipeline = AutoPipelineForImage2Image.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=self.weight_dtype,
                variant="fp16" if self.weight_dtype == torch.float16 else None
            ).to(self.device)
            
            if hasattr(self.fallback_pipeline, "enable_xformers_memory_efficient_attention"):
                self.fallback_pipeline.enable_xformers_memory_efficient_attention()
            
            logger.info("Fallback SD Turbo pipeline loaded")
            
        except Exception as e:
            logger.error(f"Could not load fallback pipeline: {e}")
            self.fallback_pipeline = None

    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        user_prompt: str = Input(
            description="User prompt to guide the super-resolution",
            default=""
        ),
        positive_prompt: str = Input(
            description="Positive prompt",
            default="clean, high-resolution, 8k, best quality, masterpiece"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="dotted, noise, blur, lowres, oversmooth, longbody, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps (SD Turbo optimized for 1-4 steps on Replicate)",
            default=4,
            ge=1,
            le=8  # Limitato per Replicate efficiency
        ),
        scale_factor: int = Input(
            description="Super-resolution scale factor (max 4x recommended for Replicate)",
            default=4,
            ge=1,
            le=6  # Limitato per memory management
        ),
        cfg_scale: float = Input(
            description="Classifier Free Guidance Scale (SD Turbo optimized at 1.0)",
            default=1.0,
            ge=0.5,
            le=1.5  # Range ottimizzato per SD Turbo
        ),
        use_kds: bool = Input(
            description="Use Kernel Density Steering",
            default=True
        ),
        bandwidth: float = Input(
            description="Bandwidth for KDS",
            default=0.1,
            ge=0.1,
            le=0.8
        ),
        num_particles: int = Input(
            description="Number of particles for KDS",
            default=10,
            ge=1,
            le=16
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=231,
            ge=0
        ),
        latent_tiled_size: int = Input(
            description="Diffusion tile size for memory management",
            default=320,
            ge=128,
            le=480
        ),
        latent_tiled_overlap: int = Input(
            description="Diffusion tile overlap",
            default=4,
            ge=4,
            le=16
        ),
    ) -> Path:
        """Run SEESR super-resolution with SD Turbo"""
        
        # Set random seed
        seed_everything(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Load and preprocess image
        input_image = Image.open(image).convert("RGB")
        ori_width, ori_height = input_image.size
        logger.info(f"Input image size: {ori_width}x{ori_height}")
        
        if self.model_loaded:
            try:
                # Use full SEESR pipeline
                result_image = self._process_seesr(
                    input_image=input_image,
                    user_prompt=user_prompt,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    scale_factor=scale_factor,
                    cfg_scale=cfg_scale,
                    use_kds=use_kds,
                    bandwidth=bandwidth,
                    num_particles=num_particles,
                    generator=generator,
                    latent_tiled_size=latent_tiled_size,
                    latent_tiled_overlap=latent_tiled_overlap,
                    ori_width=ori_width,
                    ori_height=ori_height
                )
            except Exception as e:
                logger.error(f"SEESR processing failed: {e}")
                result_image = self._process_fallback(input_image, scale_factor, generator)
        else:
            # Use fallback pipeline
            result_image = self._process_fallback(input_image, scale_factor, generator)
        
        # Save output image
        output_path = Path(tempfile.mkdtemp()) / "upscaled.png"
        result_image.save(output_path, "PNG", quality=95, optimize=True)
        
        logger.info(f"Super-resolution complete. Output saved to {output_path}")
        logger.info(f"Final image size: {result_image.size}")
        
        return output_path
    
    def _process_seesr(self, input_image, user_prompt, positive_prompt, negative_prompt, 
                      num_inference_steps, scale_factor, cfg_scale, use_kds, bandwidth,
                      num_particles, generator, latent_tiled_size, latent_tiled_overlap,
                      ori_width, ori_height):
        """Process image using full SEESR pipeline"""
        
        process_size = 512
        resize_preproc = transforms.Compose([
            transforms.Resize(process_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        
        # Generate tags using RAM model
        lq = self.tensor_transforms(input_image).unsqueeze(0).to(self.device).half()
        lq = self.ram_transforms(lq)
        
        # Get image tags and embeddings
        from ram import inference_ram as inference
        res = inference(lq, self.tag_model)
        ram_encoder_hidden_states = self.tag_model.generate_image_embeds(lq)
        
        # Build validation prompt
        validation_prompt = f"{res[0]}, {positive_prompt},"
        validation_prompt = validation_prompt if user_prompt == '' else f"{user_prompt}, {validation_prompt}"
        
        # Resize input image
        rscale = scale_factor
        input_image = input_image.resize((int(input_image.size[0] * rscale), int(input_image.size[1] * rscale)))
        
        resize_flag = False
        if min(input_image.size) < process_size:
            input_image = resize_preproc(input_image)
            input_image = input_image.resize((input_image.size[0] // 8 * 8, input_image.size[1] // 8 * 8))
            resize_flag = True
        
        width, height = input_image.size
        
        # Run SEESR pipeline
        with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
            image = self.validation_pipeline(
                validation_prompt,
                input_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                guidance_scale=cfg_scale,
                conditioning_scale=1,
                start_point='lr',
                start_steps=999,
                ram_encoder_hidden_states=ram_encoder_hidden_states,
                latent_tiled_size=latent_tiled_size,
                latent_tiled_overlap=latent_tiled_overlap,
                use_KDS=use_kds,
                bandwidth=bandwidth,
                num_particles=num_particles
            ).images[0]
        
        # Apply wavelet color fix
        try:
            from utils.wavelet_color_fix import wavelet_color_fix
            image = wavelet_color_fix(image, input_image)
        except:
            logger.warning("Could not apply wavelet color fix")
        
        # Resize back to original scale if needed
        if resize_flag:
            image = image.resize((ori_width * rscale, ori_height * rscale))
        
        return image
    
    def _process_fallback(self, input_image, scale_factor, generator):
        """Fallback processing using basic SD Turbo"""
        if self.fallback_pipeline is None:
            # Basic bicubic upscaling as last resort
            target_size = (
                int(input_image.size[0] * scale_factor),
                int(input_image.size[1] * scale_factor)
            )
            return input_image.resize(target_size, Image.BICUBIC)
        
        try:
            # Use SD Turbo for upscaling
            low_res_image = input_image.resize((512, 512), Image.LANCZOS)
            
            with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                upscaled_image = self.fallback_pipeline(
                    prompt="high quality, detailed, sharp",
                    image=low_res_image,
                    num_inference_steps=1,  # SD Turbo works well with 1 step
                    guidance_scale=1.0,
                    generator=generator
                ).images[0]
            
            # Resize to target size
            target_size = (
                int(input_image.size[0] * scale_factor),
                int(input_image.size[1] * scale_factor)
            )
            return upscaled_image.resize(target_size, Image.LANCZOS)
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            target_size = (
                int(input_image.size[0] * scale_factor),
                int(input_image.size[1] * scale_factor)
            )
            return input_image.resize(target_size, Image.BICUBIC)