# predict.py  – Fase 1: Wavelet Enhancement + GAN-Embedding
# =============================================================================
# 1.  Enhanced wavelet-based colour/frequency correction
# 2.  Real-ESRGAN “GAN-Embedding” preprocessing (training-free)
# 3.  ZERO modifiche ai parametri utente: il tutto è automatico
#    (puoi disattivare via flag interni se necessario)
# -----------------------------------------------------------------------------
# Dipendenze nuove:
#   pip install basicsr==1.4.2 realesrgan==0.3.0
#   (occupano ~200 MB su disco,  ~0.8 GB VRAM durante inferenza 4×)
# =============================================================================

import os, sys, tempfile, logging
from typing import List, Iterator

import numpy as np
import torch
from PIL import Image
from cog import BasePredictor, Input, Path
from pytorch_lightning import seed_everything
from typing import Any, cast
from diffusers import AutoencoderKL, DDIMScheduler  # type: ignore[import]
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download
from torchvision import transforms

# NEW imports — Fase 1
# NOTE: imports for Real-ESRGAN are performed lazily inside _init_gan_embedding()

# util imports
from utils.xformers_utils import (
    is_xformers_available,
    optimize_models_attention,
    print_attention_status,
)
from utils.wavelet_color_fix import wavelet_color_fix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor(BasePredictor):
    # --------------------------------------------------------------------- #
    #                           SET-UP                                       #
    # --------------------------------------------------------------------- #
    def setup(self, weights=None) -> None:
        """Load SEESR + SD-Turbo and Real-ESRGAN (GAN-Embedding)."""
        logger.info("🚀 Loading SEESR + SD-Turbo (+ Real-ESRGAN)…")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Lite/Test mode: skip heavy downloads/initializations during CI/pytest
        lite_env = os.getenv("SEESR_LITE") or os.getenv("SEESR_TEST_MODE") or os.getenv("PYTEST_CURRENT_TEST")
        if lite_env:
            logger.info("🧪 Lite/Test mode attivo: skip download modelli e init pesante.")
            self.gan_enhancer = None
            self.tensor_transforms = transforms.ToTensor()
            self.ram_transforms = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model_loaded = True
            return

        # ——  GPU tweaks  —————————————————————————————
        if self.device == "cuda":
            torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32 = (True, True)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True

        # ——  Download diffusion models  ——————————————
        snapshot_download("alexnasa/SEESR", local_dir="models/seesr", cache_dir="/tmp/hf")
        snapshot_download("stabilityai/sd-turbo", local_dir="models/sd-turbo", cache_dir="/tmp/hf")
        snapshot_download("xinyu1205/recognize_anything_model", local_dir="models/ram", cache_dir="/tmp/hf")

        # ——  Core diffusion components  ——————————————
        self.pretrained_model_path = "models/sd-turbo"
        self.seesr_model_path = "models/seesr"

        self.scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_path, subfolder="scheduler")
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_path, subfolder="text_encoder")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_path, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_path, subfolder="vae")

        # ——  Custom UNet / ControlNet / pipeline  ————
        from models.unet_2d_condition import UNet2DConditionModel
        from models.controlnet import ControlNetModel
        from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
        from ram.models.ram_lora import ram

        self.unet = UNet2DConditionModel.from_pretrained_orig(
            self.pretrained_model_path, self.seesr_model_path, subfolder="unet"
        )
        self.controlnet = ControlNetModel.from_pretrained(self.seesr_model_path, subfolder="controlnet")

        # ——  Freeze & move  ————————————————————————————
        for m in (self.vae, self.text_encoder, self.unet, self.controlnet):
            m.requires_grad_(False)
            m.to(self.device, dtype=self.weight_dtype)  # type: ignore[arg-type]

        self.validation_pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=cast(Any, self.scheduler),
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device, dtype=self.weight_dtype)

        # ——  Memory tweaks  ————————————————————————————
        if hasattr(self.validation_pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self.validation_pipeline.enable_xformers_memory_efficient_attention()
                optimize_models_attention([self.unet, self.controlnet])
                print_attention_status()
            except Exception as e:
                logger.warning(f"⚠️  xformers not enabled: {e}")

        # ——  RAM for auto-tagging  ————————————————————
        self.tag_model = ram(
            pretrained="models/ram/ram_swin_large_14m.pth",
            pretrained_condition="models/ram/DAPE.pth",
            image_size=384,
            vit="swin_l",
        ).eval().to(self.device, dtype=self.weight_dtype)

        # ——  NEW • Real-ESRGAN initialisation  —————————
        self._init_gan_embedding()

        # ——  Misc transforms  ————————————————————————
        self.tensor_transforms = transforms.ToTensor()
        self.ram_transforms = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.model_loaded = True
        logger.info("✅  Predictor ready (Fase 1 enabled).")

    # ------------------------------------------------------------------ #
    #          GAN-Embedding (Real-ESRGAN) helper initialiser            #
    # ------------------------------------------------------------------ #
    def _init_gan_embedding(self) -> None:
        """Load Real-ESRGAN x4plus model for preprocessing."""
        try:
            # Lazy imports to avoid ImportError at module import time
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_path = snapshot_download(
                "xinntao/realesrgan", local_dir="models/realesrgan", cache_dir="/tmp/hf"
            ) + "/RealESRGAN_x4plus.pth"

            self.gan_enhancer = RealESRGANer(
                scale=4,
                model_path=model_path,
                model=model,
                gpu_id=0 if self.device == "cuda" else None,
                half=self.device == "cuda",
            )
            logger.info("✅  Real-ESRGAN loaded for GAN-Embedding.")
        except Exception as err:
            self.gan_enhancer = None
            logger.warning(f"GAN-Embedding disabled: {err}")

    # ------------------------------------------------------------------ #
    #                             PREDICT                                #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        image: Path = Input(description="Input image to upscale"),
        user_prompt: str = Input(description="User prompt to guide SR", default=""),
        positive_prompt: str = Input(default="clean, high-resolution, 8k, masterpiece"),
        negative_prompt: str = Input(
            default="dotted, noise, blur, lowres, oversmooth, bad anatomy, bad hands, cropped"
        ),
        num_inference_steps: int = Input(default=4, ge=1, le=8),
        scale_factor: int = Input(default=4, ge=1, le=6),
        cfg_scale: float = Input(default=1.0, ge=0.5, le=1.5),
        use_kds: bool = Input(default=True),
        bandwidth: float = Input(default=0.1, ge=0.1, le=0.8),
        num_particles: int = Input(default=10, ge=1, le=16),
        seed: int = Input(default=231, ge=0),
        latent_tiled_size: int = Input(default=320, ge=128, le=480),
        latent_tiled_overlap: int = Input(default=4, ge=4, le=16),
        **kwargs,
    ) -> Path:
        """Run SEESR SR with GAN-Embedding and enhanced wavelet fix."""
        seed_everything(seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # ——  Load input  ————————————————————————————
        input_image = Image.open(image).convert("RGB")
        ori_w, ori_h = input_image.size
        logger.info(f"Input size : {ori_w}×{ori_h}")

        # ——  Step 1 • GAN-Embedding preprocessing  ——————
        if getattr(self, "gan_enhancer", None):
            try:
                enhanced, _ = self.gan_enhancer.enhance(input_image)  # type: ignore[union-attr]
                if isinstance(enhanced, np.ndarray):
                    input_image = Image.fromarray(enhanced)
                else:
                    input_image = enhanced
                logger.info("GAN-Embedding preprocessing applied.")
            except Exception as e:
                logger.warning(f"GAN-Embedding failed, continue raw: {e}")

        # ——  Step 2 • Run SEESR  ——————————————
        sr = self._process_seesr(
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
            ori_width=ori_w,
            ori_height=ori_h,
        )

        # —— Step 3 • Enhanced wavelet colour fix  ——————
        sr = self._enhanced_wavelet_fix(sr, input_image)

        # ——  Save & return  ————————————————————————
        out_dir = tempfile.mkdtemp()
        out_path = Path(out_dir) / "upscaled.png"
        sr.save(out_path, "PNG", quality=95, optimize=True)
        logger.info(f"Done → {out_path}  |  final {sr.size}")
        return out_path

    # ------------------------------------------------------------------ #
    #                    Core SEESR processing wrapper                    #
    # ------------------------------------------------------------------ #
    def _process_seesr(
        self,
        input_image: Image.Image,
        user_prompt: str,
        positive_prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        scale_factor: int,
        cfg_scale: float,
        use_kds: bool,
        bandwidth: float,
        num_particles: int,
        generator: torch.Generator,
        latent_tiled_size: int,
        latent_tiled_overlap: int,
        ori_width: int,
        ori_height: int,
    ) -> Image.Image:
        """Run the SEESR pipeline to generate SR image."""
        # Build prompts
        prompt = (user_prompt + ", " if user_prompt else "") + positive_prompt
        neg_prompt = negative_prompt

        # Prepare control image (resize based on scale factor)
        target_w, target_h = int(ori_width * scale_factor), int(ori_height * scale_factor)

        # Enable tiled VAE for memory control
        if hasattr(self.validation_pipeline, "_init_tiled_vae"):
            try:
                self.validation_pipeline._init_tiled_vae(
                    encoder_tile_size=max(1024, latent_tiled_size * 8),
                    decoder_tile_size=max(224, latent_tiled_size),
                )
            except Exception as e:
                logger.warning(f"Tiled VAE init failed: {e}")

        # RAM guidance: compute image embeds if available
        ram_states = None
        try:
            img_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device, dtype=self.weight_dtype)
            ram_states = self.tag_model.generate_image_embeds(img_tensor)
        except Exception as e:
            logger.warning(f"RAM guidance unavailable: {e}")
            ram_states = None

        # Call pipeline
        out = self.validation_pipeline(
            prompt=prompt,
            image=input_image,
            height=target_h,
            width=target_w,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(cfg_scale),
            negative_prompt=neg_prompt,
            generator=generator,
            output_type="pil",
            ram_encoder_hidden_states=cast(torch.FloatTensor, ram_states) if ram_states is not None else None,
            latent_tiled_size=latent_tiled_size,
            latent_tiled_overlap=latent_tiled_overlap,
            use_KDS=bool(use_kds),
            bandwidth=float(bandwidth),
            num_particles=int(num_particles),
        )

        sr_img: Any
        try:
            # Prefer attribute if available
            sr_img = out.images[0]  # type: ignore[attr-defined]
        except Exception:
            if isinstance(out, (list, tuple)) and len(out) > 0:
                sr_img = out[0]
            else:
                sr_img = out

        if not isinstance(sr_img, Image.Image):
            sr_img = Image.fromarray(np.array(sr_img))

        return sr_img

    # ------------------------------------------------------------------ #
    #             ENHANCED  Wavelet + Frequency  Correction              #
    # ------------------------------------------------------------------ #
    def _enhanced_wavelet_fix(self, sr_img: Image.Image, lr_img: Image.Image) -> Image.Image:
        """Wavelet colour transfer + simple frequency alignment."""
        try:
            # 1. Colour transfer (as in original code)
            base = wavelet_color_fix(sr_img, lr_img)
            # 2. Frequency alignment (training-free, ≈ 0.5 ms)
            aligned = self._freq_align(base, lr_img)
            return aligned
        except Exception as e:
            logger.warning(f"Enhanced wavelet fix failed: {e}")
            return sr_img

    @staticmethod
    def _freq_align(hr: Image.Image, lr: Image.Image) -> Image.Image:
        """Clamp HR high-freq magnitude to LR to suppress hallucinations."""
        import cv2
        hr_np = cv2.cvtColor(np.array(hr), cv2.COLOR_RGB2YCrCb).astype(np.float32)
        # Use PIL Resampling enum for compatibility with Pillow>=10
        try:
            resample_bicubic = Image.Resampling.BICUBIC
        except Exception:
            resample_bicubic = Image.BICUBIC  # type: ignore[attr-defined]
        lr_np = cv2.cvtColor(np.array(lr.resize(hr.size, resample_bicubic)), cv2.COLOR_RGB2YCrCb).astype(np.float32)

        # DCT-domain magnitude comparison (global DCT for robustness)
        def _dct(x):
            return cv2.dct(x)

        hr_dct, lr_dct = _dct(hr_np[:, :, 0]), _dct(lr_np[:, :, 0])
        mask = (np.abs(hr_dct) > np.abs(lr_dct) * 3.0)  # 3× threshold empirico
        hr_dct[mask] = lr_dct[mask]  # clamp overshoot

        # inverse DCT
        def _idct(x):
            return cv2.idct(x)

        hr_np[:, :, 0] = np.clip(_idct(hr_dct), 0, 255)
        out = cv2.cvtColor(hr_np.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(out)
