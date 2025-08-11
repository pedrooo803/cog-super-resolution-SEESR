# 🎯 Cog Directory

Questa directory contiene i file di configurazione e predizione per il deployment con Cog/Replicate.

## 📁 File Cog

### Core Files
- **`cog.yaml`** - Configurazione principale Cog per Replicate
- **`predict.py`** - Predictor principale con classe Predictor

## 🚀 Quick Start

### Local Development
```bash
# Install Cog
curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
chmod +x /usr/local/bin/cog

# Build model
cog build

# Test prediction
cog predict -i image=@input.jpg
```

### Deploy to Replicate
```bash
# Login to Replicate
cog login

# Push model
cog push r8.im/username/seesr-sd-turbo
```

## 🔧 Configurazione cog.yaml

### Build Settings
- **GPU**: true (CUDA 11.8)
- **Python**: 3.10
- **System Packages**: OpenGL, FFmpeg, etc.
- **Python Packages**: Complete dependency list

### Model Configuration
- **Predictor**: `cog/predict.py:Predictor`
- **GPU Memory**: Ottimizzato per T4 (16GB) e A40 (48GB)
- **Dependencies**: Versioni sincronizzate con requirements.txt

## 🎯 Predictor Features

### Input Parameters
- **image**: Input image (PNG/JPG)
- **user_prompt**: Custom prompt guidance
- **positive_prompt**: Quality enhancement prompt
- **negative_prompt**: Artifact prevention prompt
- **num_inference_steps**: 1-8 steps (SD Turbo optimized)
- **scale_factor**: 1-6x upscaling
- **cfg_scale**: 0.5-1.5 guidance (SD Turbo optimized)
- **use_kds**: Kernel Density Steering
- **seed**: Reproducibility seed

### Model Integration
- ✅ **SEESR Model** - Custom super-resolution
- ✅ **SD Turbo** - Ultra-fast diffusion (1-4 steps)
- ✅ **RAM Model** - Automatic image tagging
- ✅ **ControlNet** - Structure preservation
- ✅ **Wavelet Color Fix** - Natural color correction

### Performance Optimizations
- ✅ **Tiled VAE** - Memory efficient processing
- ✅ **xformers** - Attention optimization
- ✅ **Mixed Precision** - FP16 inference
- ✅ **GPU Memory Management** - Dynamic allocation
- ✅ **Fallback Pipeline** - CPU compatibility

## 🚀 Development Workflow

### Test Locally
```bash
# Quick test
cog predict -i image=@test.jpg

# With custom parameters  
cog predict \
  -i image=@input.jpg \
  -i user_prompt="portrait photo" \
  -i num_inference_steps=4 \
  -i scale_factor=4
```

### Debug Mode
```bash
# Run with logs
cog run python3 cog/predict.py

# Interactive container
cog run bash
```

### Performance Testing
```bash
# Benchmark different settings
cog predict -i image=@benchmark.jpg -i num_inference_steps=1
cog predict -i image=@benchmark.jpg -i num_inference_steps=4
cog predict -i image=@benchmark.jpg -i num_inference_steps=8
```

## 📊 Expected Performance

### GPU T4 (16GB)
- **1-step**: ~2 seconds
- **4-step**: ~5 seconds
- **8-step**: ~10 seconds

### GPU A40 (48GB)
- **1-step**: ~1.5 seconds
- **4-step**: ~3 seconds
- **8-step**: ~6 seconds

### CPU Fallback
- **4-step**: ~60-120 seconds (depends on CPU)

## 🔄 Model Updates

### Update Dependencies
1. Modify `cog.yaml` packages list
2. Update `requirements.txt` if needed
3. Rebuild: `cog build`

### Update Model Weights
1. Modify paths in `predict.py`
2. Update download URLs in `../deployment/download_models.py`
3. Rebuild: `cog build`

### Update Parameters
1. Modify Input definitions in `predict.py`
2. Test locally: `cog predict`
3. Deploy: `cog push`

## 🐛 Troubleshooting

### Build Issues
```bash
# Clean build
cog build --no-cache

# Check logs
cog logs

# Debug build
cog run bash
```

### Runtime Issues
```bash
# Check GPU
cog run nvidia-smi

# Test imports
cog run python3 -c "import torch; print(torch.cuda.is_available())"

# Memory debugging
cog run python3 -c "import torch; print(torch.cuda.memory_summary())"
```

### Performance Issues
```bash
# Profile inference
cog predict -i image=@test.jpg --profile

# Memory optimization
# Reduce scale_factor or use tiled processing
```
