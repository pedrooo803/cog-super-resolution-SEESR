# 📦 Deployment Directory

Questa directory contiene modelli, configurazioni e script per il deployment del progetto SEESR.

## 📁 Struttura Directory

### Models & Presets
- **`preset/`** - Directory contenente tutti i modelli pre-scaricati
  - `models/seesr/` - Modelli SEESR custom
  - `models/sd-turbo/` - Modelli Stable Diffusion Turbo
  - `models/ram/` - Modelli RAM per tagging automatico

### Scripts
- **`download_models.py`** - Script per download automatico modelli

### Configuration Files
- **`replicate_config_analysis.json`** - Analisi configurazioni Replicate
- **`REPLICATE_FINAL_RECOMMENDATION.md`** - Raccomandazioni finali Replicate
- **`REPLICATE_HARDWARE_GUIDE.md`** - Guida hardware Replicate

## 🚀 Model Management

### Download Models
```bash
# Download automatico tutti i modelli
python deployment/download_models.py

# Da Docker
docker run seesr-sd-turbo python3 deployment/download_models.py
```

### Model Locations
```
deployment/preset/models/
├── seesr/          # ~8GB - SEESR super-resolution models
├── sd-turbo/       # ~4GB - Stable Diffusion Turbo  
└── ram/            # ~3GB - RAM tagging models
```

## 📋 Models Details

### SEESR Model (`alexnasa/SEESR`)
- **Purpose**: Semantic Edge Enhanced Super-Resolution
- **Components**: UNet2D, ControlNet custom
- **Size**: ~8GB
- **Usage**: Main super-resolution processing

### SD Turbo Model (`stabilityai/sd-turbo`)
- **Purpose**: Ultra-fast diffusion (1-4 steps)
- **Components**: UNet2D, VAE, Text Encoder, Scheduler
- **Size**: ~4GB  
- **Usage**: Fast high-quality generation

### RAM Model (`xinyu1205/recognize_anything_model`)
- **Purpose**: Automatic image tagging and description
- **Components**: Vision Transformer, LoRA adapters
- **Size**: ~3GB
- **Usage**: Generate automatic prompts from images

## ⚙️ Configuration Files

### Replicate Configuration
- **Hardware recommendations** per diversi workload
- **Memory optimization** settings
- **Performance benchmarks** su GPU T4/A40
- **Cost analysis** per inference

### Download Script Features
- ✅ **Automatic retry** on failed downloads
- ✅ **Resume capability** for interrupted downloads
- ✅ **Integrity checking** of downloaded files
- ✅ **Progress reporting** with size information
- ✅ **Error handling** with detailed logging

## 🚀 Deployment Strategies

### Local Development
```bash
# Setup con modelli locali
./start_seesr.sh setup
python deployment/download_models.py
```

### Docker Production
```bash
# I modelli vengono scaricati automaticamente durante build
docker build -t seesr-sd-turbo .
```

### Replicate Cloud
```bash
# I modelli sono pre-configurati nel container
cog push r8.im/username/seesr-sd-turbo
```

## 💾 Storage Requirements

### Development
- **Local cache**: `~/.cache/huggingface/` (~20GB)
- **Models directory**: `deployment/preset/models/` (~15GB)
- **Total**: ~35GB

### Production (Docker)
- **Container size**: ~27GB (include modelli)
- **Runtime cache**: ~5GB aggiuntivi
- **Total**: ~32GB

### Cloud (Replicate)
- **Container size**: ~27GB
- **Persistent storage**: Gestito automaticamente
- **Cold start**: ~30-60 secondi

## 🔄 Model Updates

### Update Existing Models
```bash
# Forza re-download
rm -rf deployment/preset/models/seesr
python deployment/download_models.py
```

### Add New Models
1. Modifica `download_models.py`
2. Aggiungi nuova configurazione modello
3. Aggiorna path nei file predict.py
4. Ricostruisci container

### Version Management
```bash
# Tag specifiche versioni
git tag v1.0-models
git push origin v1.0-models

# Rollback se necessario
git checkout v1.0-models
docker build -t seesr-sd-turbo:v1.0 .
```

## 🔍 Monitoring & Health

### Model Verification
```bash
# Verifica integrità modelli
python deployment/download_models.py --verify-only

# Check spazio utilizzato
du -sh deployment/preset/models/*
```

### Performance Monitoring
- **Download speed**: Monitoraggio velocità download
- **Model loading time**: Tempo caricamento all'avvio
- **Memory usage**: Utilizzo memoria durante inference
- **Error rates**: Frequenza errori download/caricamento

## 🐛 Troubleshooting

### Download Issues
```bash
# Pulizia cache corrotta
rm -rf ~/.cache/huggingface/transformers/
rm -rf /tmp/huggingface_cache/

# Re-download con log verboso
python deployment/download_models.py --verbose
```

### Storage Issues
```bash
# Pulizia modelli non utilizzati
rm -rf deployment/preset/models/*/blobs/
python deployment/download_models.py

# Symbolic links per risparmio spazio
ln -s ~/.cache/huggingface/transformers/ deployment/preset/models/
```

### Network Issues
```bash
# Configurazione proxy se necessario
export HF_HUB_CACHE="/tmp/huggingface_cache"
export HF_HOME="/tmp/huggingface"

# Download manuale con wget/curl
wget -c https://huggingface.co/alexnasa/SEESR/resolve/main/model.safetensors
```
