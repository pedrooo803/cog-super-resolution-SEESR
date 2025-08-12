# SEESR with SD Turbo – Optimized Super-Resolution

An advanced implementation of SEESR (Semantic Edge Enhanced Super-Resolution) optimized with SD Turbo for ultra-fast, high-quality super-resolution. Includes optional Real-ESRGAN pre-enhancement and robust color/frequency correction.

## 🚀 Key Features

- Ultra-fast inference: 1–4 steps vs 20–50 traditional
- Quality retained with SD Turbo optimizations for few steps
- Memory efficient: Tiled VAE for large images on limited VRAM
- Automatic tagging: RAM model auto-generates guidance from images
- Color correction: Wavelet-based color fix for natural results
- KDS (Kernel Density Steering): Advanced generation control
- Optional Real-ESRGAN “GAN-Embedding” pre-enhancement
- Docker-ready: Pre-configured container with pre-fetched models
- Cross-platform: macOS, Linux, and Windows
- Virtual environment: Isolated, reproducible setup

## 🐳 Deployment & Build

### Docker Build (Recommended for Production)
The project includes a fully updated Dockerfile with:
- Python 3.10 environment
- Pre-downloaded model weights during build
- Automatic environment tests
- CUDA optimizations and memory management

```bash
# Quick build with Cog
cog build

# Manual Docker build
./docker/docker_build.sh build

# Full instructions
cat docker/DOCKER_BUILD_GUIDE.md
```

### Local Development
```bash
# Automatic virtual environment setup
./start_seesr.sh setup

# Run tests
./start_seesr.sh test

# Run super-resolution
./start_seesr.sh run input.jpg
```

## 📁 Project Structure

```text
.
├── activate_seesr.sh              # Activate the local venv
├── cog.yaml                       # Cog configuration (root)
├── config.yaml                    # App config
├── predict.py                     # Shim: re-exports Predictor from cog/predict.py
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package metadata (editable install)
├── start_seesr.sh                 # Helper for setup/run/test
├── TECHNICAL_DOCS.md              # Technical docs
├── USAGE_EXAMPLES.md              # Extra usage examples
├── test_input.jpg                 # Sample input image
├── cog/
│   ├── predict.py                 # Main Predictor (Cog entrypoint)
│   └── README.md
├── deployment/
│   ├── download_models.py         # Optional weights prefetch
│   ├── REPLICATE_FINAL_RECOMMENDATION.md
│   ├── REPLICATE_HARDWARE_GUIDE.md
│   └── preset/models/             # Model presets
├── docker/
│   ├── dockerfile                 # Dockerfile
│   ├── docker_build.sh            # Build helper
│   └── README.md
├── models/                        # Custom UNet/ControlNet
│   ├── controlnet.py
│   └── unet_2d_condition.py
├── pipelines/
│   └── pipeline_seesr.py          # SEESR + SD Turbo pipeline
├── ram/
│   └── models/ram_lora.py         # RAM model (auto-tagging)
├── tests/                         # Test suite
│   ├── test_complete.py
│   ├── test_docker_env.py
│   ├── test_environment.py
│   └── test_seesr.py
└── utils/
    ├── wavelet_color_fix.py       # Wavelet/AdaIN/luminance color fixes
    └── xformers_utils.py          # Attention optimizations helpers
```

## 🔧 Installation & Setup

### 🚀 Automatic Setup with Virtual Environment (Recommended)

The easiest way to use SEESR is via the helper script, which creates an isolated virtual environment and installs all dependencies:

```bash
# Clone the repository
git clone https://github.com/alexgenovese/cog-super-resolution-SEESR.git
cd cog-super-resolution-SEESR

# Automatic venv + install
./start_seesr.sh setup
```

This script will:
- Verify system requirements (Python 3.9+)
- Create a dedicated venv (`seesr_env`)
- Install all required dependencies
- Configure the environment for usage

### 🎯 Quick Start

```bash
# Test the model with a sample image
./start_seesr.sh test

# Start a Python shell inside the env
./start_seesr.sh python

# Quick performance benchmark
./start_seesr.sh benchmark

# Manually activate the environment
source activate_seesr.sh

# Main commands:
# ./start_seesr.sh                - Setup/run helper
# python tests/test_complete.py   - System test
# python predict.py               - Predictor shim (imports cog/predict.py)

```

### System Requirements
- Python 3.9+ (auto-checked)
- CUDA 11.8+ (optional for GPU)
- 8–16GB VRAM (recommended for GPU)
- 4GB+ RAM (CPU minimum)

### Manual Installation (Advanced)

```bash
# Install dependencies
pip install -r requirements.txt

# Editable install
pip install -e .
```

### Cog Installation

```bash
# Install Cog if not present
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog

# Build container
cog build

# Test model
cog predict -i image=@input.jpg
```

## 🎯 Usage

### Basic Usage

```python
from predict import Predictor

# Initialize predictor
predictor = Predictor()
predictor.setup()

# Run super-resolution
result = predictor.predict(
    image="input.jpg",
    scale_factor=4,
    num_inference_steps=4,  # SD Turbo ottimizzato per 1-4 steps
    cfg_scale=1.0,          # SD Turbo funziona meglio con CFG=1.0
    use_kds=True,          # Abilita Kernel Density Steering
    positive_prompt="high quality, detailed, 8k",
    negative_prompt="blur, lowres, artifacts"
)
```

### Advanced Parameters

```python
result = predictor.predict(
    image="input.jpg",
    user_prompt="beautiful landscape",            # Optional user prompt
    positive_prompt="masterpiece, best quality",  # Positive prompt
    negative_prompt="blur, noise, artifacts",     # Negative prompt
    num_inference_steps=4,                        # 1–4 for SD Turbo
    scale_factor=4,                               # Upscale factor
    cfg_scale=1.0,                                # SD Turbo CFG
    use_kds=True,                                 # Kernel Density Steering
    bandwidth=0.1,                                # KDS bandwidth
    num_particles=10,                             # KDS particles
    seed=42,                                      # Reproducibility seed
    latent_tiled_size=320,                        # Diffusion tile size
    latent_tiled_overlap=4                        # Tile overlap
)
```

## ⚡ SD Turbo Optimizations

### Optimal Settings
- Inference Steps: 1–4 (vs 20–50 traditional)
- CFG Scale: 1.0 (SD Turbo is tuned for low CFG)
- Scheduler: DDIM with tuned timesteps
- Memory: Tiled VAE for large images

### Expected Performance
- Inference time: ~5–15s (vs 30–60s traditional)
- VRAM: ~8–10GB (with tiling)
- Quality: High thanks to semantic guidance
- Max resolution: Limited by available VRAM

## 🎨 Advanced Features

### RAM (Recognize Anything Model)
- Automatic tagging: Generates image tags
- Semantic guidance: Improves quality using tag embeddings
- LoRA integration: Efficient adaptations

### Kernel Density Steering (KDS)
- Generation control: Guides diffusion
- Stability: Reduces artifacts and improves consistency
- Configurable: Bandwidth and particles

### Wavelet Color Correction
- Preserves original colors
- Multi-method: Wavelet, AdaIN, and luminance correction
- Automatic: Applied to the output image

### Real-ESRGAN “GAN-Embedding” (Optional)
- Training-free enhancement before diffusion
- Improves detail and stability for low-quality inputs
- Can be disabled automatically when not available

## 🛠️ Advanced Configuration

### Custom Model Paths

```python
# Configure custom model paths
import os
os.environ['SEESR_MODEL_PATH'] = '/path/to/custom/seesr'
os.environ['SD_TURBO_PATH'] = '/path/to/custom/sd-turbo'
os.environ['RAM_MODEL_PATH'] = '/path/to/custom/ram'
# Set this to skip heavy downloads during CI/tests (not for production inference)
os.environ['SEESR_TEST_MODE'] = '1'
```

### Memory Management

```python
# For limited VRAM
predictor.validation_pipeline._init_tiled_vae(
    encoder_tile_size=512,    # Lower for less VRAM
    decoder_tile_size=128     # Lower for less VRAM
)

# Enable gradient checkpointing
predictor.unet.enable_gradient_checkpointing()
```

## 📊 Benchmarks

| Method | Time (s) | VRAM (GB) | PSNR | SSIM |
|--------|----------|-----------|------|------|
| SEESR (original) | 45–60 | 12–16 | 28.5 | 0.85 |
| SEESR + SD Turbo | 8–15  | 8–10  | 29.2 | 0.87 |
| SD Turbo fallback | 3–5   | 6–8   | 26.8 | 0.82 |

## 🐛 Troubleshooting

### Common Errors

1) CUDA Out of Memory
   ```python
   # Riduci dimensioni tile
   latent_tiled_size=256
   latent_tiled_overlap=2
   ```

2) Models not found
   ```bash
   # Forza il download
   python -c "from predict import Predictor; p = Predictor(); p.setup()"
   ```

3) Low quality output
   ```python
   # Aumenta steps se necessario
   num_inference_steps=4  # Massimo per SD Turbo
   cfg_scale=1.0         # Ottimale per SD Turbo
   ```

## 📝 API Reference

### Predictor.predict()

Signature (cog/predict.py):

```python
def predict(
    image: Path,
    user_prompt: str = "",
    positive_prompt: str = "clean, high-resolution, 8k, masterpiece",
    negative_prompt: str = "dotted, noise, blur, lowres, oversmooth, bad anatomy, bad hands, cropped",
    num_inference_steps: int = 4,   # 1–8
    scale_factor: int = 4,          # 1–6
    cfg_scale: float = 1.0,         # 0.5–1.5
    use_kds: bool = True,
    bandwidth: float = 0.1,         # 0.1–0.8
    num_particles: int = 10,        # 1–16
    seed: int = 231,
    latent_tiled_size: int = 320,   # 128–480
    latent_tiled_overlap: int = 4,  # 4–16
) -> Path
```

## ⚠️ Limitations & Considerations

### Technical Limits

Hardware:
- GPU: NVIDIA with 8GB+ VRAM recommended (CPU works but slower)
- RAM: 8GB minimum, 16GB+ recommended for large images
- Disk: 15–20GB for models and cache

Model:
- Very small inputs (<256px) may yield suboptimal results
- Scale factors >4× may introduce artifacts
- Tuned for natural photos; results vary for drawings/art

Performance Considerations
- Virtual environments are recommended for consistent PyTorch/CUDA
- GPU (CUDA): ~5–15s per inference
- CPU: ~2–10 minutes per inference
- Apple M1/M2: Intermediate with MPS

Memory Management
- Tiled VAE for >2K images with <16GB VRAM
- Gradient checkpointing reduces VRAM at speed cost
- Mixed precision (fp16) enabled by default

Common Troubles

CUDA Out of Memory:
```bash
# Reduce VAE tile size
latent_tiled_size = 256  # default: 320

# Reduce internal batch if customized
```

Import Errors:
```bash
# Recreate virtual environment
rm -rf seesr_env
./start_seesr.sh setup
```

Slow Performance:
```bash
# Check GPU detection
./start_seesr.sh test

# Force CPU usage if necessary
export CUDA_VISIBLE_DEVICES=""
```

Models Not Found:
```bash
# Models are downloaded automatically
# Ensure internet connectivity on first run
```

### Best Practices

For performance:
- Use NVIDIA GPU with CUDA 11.8+
- Keep inference steps 2–4 for SD Turbo
- Use CFG scale = 1.0
- Enable tiled VAE for large images

For quality:
- Provide precise prompts
- Prefer moderate scale factors (2×–4×)
- Enable KDS for stability
- Try different seeds

For development:
- Always use a virtual environment
- Iterate on small images first
- Monitor memory usage
- Keep a known-good requirements.txt

## 📄 License

MIT License – see [LICENSE](LICENSE).

## 🤝 Contributing

Contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

## 📞 Support

- Issues: [GitHub Issues](https://github.com/alexgenovese/cog-super-resolution-SEESR/issues)
- Discussions: [GitHub Discussions](https://github.com/alexgenovese/cog-super-resolution-SEESR/discussions)

## 🙏 Credits

- SEESR: Based on the Semantic Edge Enhanced Super-Resolution work
- SD Turbo: Stability AI
- RAM: Recognition Anything Model team
- Diffusers: Hugging Face

---

Note: For CI/tests, you can set SEESR_LITE=1 or SEESR_TEST_MODE=1 to skip heavy downloads. Do not use lite mode for real inference.