# 🐳 Docker Directory

Questa directory contiene tutti i file necessari per il deployment Docker del progetto SEESR.

## 📁 File Docker

### Core Files
- **`dockerfile`** - Dockerfile principale ottimizzato per produzione
- **`docker_build.sh`** - Script helper per build e gestione container
- **`DOCKER_BUILD_GUIDE.md`** - Guida completa per il build Docker

## 🚀 Quick Start

### Build Container
```bash
# Build automatico (rileva Cog/Docker)
./docker/docker_build.sh build

# Build manuale Docker
cd docker && docker build -t seesr-sd-turbo -f dockerfile ..

# Build con Cog (da root directory)
cog build -f cog/cog.yaml
```

### Test Container
```bash
# Test completo
./docker/docker_build.sh test

# Test specifici
./docker/docker_build.sh inference
```

## 🔧 Caratteristiche Dockerfile

### Environment
- **Base**: `nvidia/cuda:11.8-devel-ubuntu20.04`
- **Python**: 3.10 (allineato con cog.yaml)
- **Dependencies**: Complete dal requirements.txt
- **Models**: Pre-scaricati durante build

### Ottimizzazioni
- ✅ **CUDA optimizations** con TF32 e cuDNN v8
- ✅ **Memory management** ottimizzato per GPU T4/A40
- ✅ **Multi-stage caching** per build più veloci
- ✅ **Health check** avanzato per monitoraggio
- ✅ **Environment variables** ottimizzate

### Dimensioni
- **Base image**: ~4GB
- **Dependencies**: ~8GB  
- **Models**: ~15GB
- **Total**: ~27GB

## 🔄 Build Process

Il Dockerfile esegue:

1. **System Setup** - Installa Python 3.10 e dipendenze sistema
2. **Python Dependencies** - Installa requirements.txt completo
3. **Model Download** - Scarica modelli SEESR, SD Turbo, RAM
4. **Environment Test** - Valida installazione con test automatici
5. **Optimization** - Applica ottimizzazioni CUDA e memory

## 📋 Scripts Helper

### docker_build.sh
Script completo per gestione container:

```bash
./docker_build.sh build     # Build container
./docker_build.sh test      # Test container
./docker_build.sh run       # Shell interattivo
./docker_build.sh deploy    # Deploy su registry
./docker_build.sh clean     # Pulizia cache
./docker_build.sh status    # Stato container
```

## 🚀 Deployment

### Docker Hub
```bash
./docker_build.sh deploy
# Seleziona opzione 1 (Docker Hub)
```

### Replicate
```bash
cog push r8.im/username/seesr-sd-turbo
```

### Salvataggio Locale
```bash
./docker_build.sh deploy
# Seleziona opzione 3 (tar.gz)
```

## 🐛 Troubleshooting

### Build Fallito
```bash
# Build senza cache
docker build --no-cache -t seesr-sd-turbo -f dockerfile ..

# Verifica log
./docker_build.sh logs
```

### Memory Issues
```bash
# Aumenta memoria Docker
docker build --memory=8g -t seesr-sd-turbo -f dockerfile ..

# Pulizia sistema
./docker_build.sh clean
```

### GPU Issues
```bash
# Verifica NVIDIA runtime
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```
