# 🐳 Docker Build Guide - SEESR con SD Turbo

## Panoramica

Il nuovo Dockerfile è stato completamente aggiornato per essere allineato con il progetto corrente e ottimizzato per l'environment di produzione. Include:

- ✅ **Environment completo** con Python 3.10 (allineato con cog.yaml)
- ✅ **Pre-download automatico** dei modelli durante il build
- ✅ **Test ambiente** integrati per validazione
- ✅ **Ottimizzazioni CUDA** e memory management
- ✅ **Health check** avanzato per monitoraggio

## Caratteristiche Principali

### 🔧 Environment Setup
- **Python 3.10** (aggiornato da 3.9)
- **CUDA 11.8** con ottimizzazioni complete
- **Dipendenze complete** dal requirements.txt
- **Moduli custom** del progetto installati

### 📦 Modelli Pre-scaricati
Durante il build vengono automaticamente scaricati:
- **SEESR Model** (`alexnasa/SEESR`)
- **SD Turbo Model** (`stabilityai/sd-turbo`)
- **RAM Model** (`xinyu1205/recognize_anything_model`)

### 🚀 Ottimizzazioni Performance
- **Memory management** ottimizzato per GPU T4 e A40
- **CUDA optimizations** con TF32 e cuDNN v8
- **Attention optimizations** con xformers
- **Tile processing** per gestione memoria

## Build Instructions

### 1. Build Locale

```bash
# Build del container
docker build -t seesr-sd-turbo .

# Verifica del build
docker run --rm seesr-sd-turbo python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Build con Cog (Consigliato)

```bash
# Assicurati che Cog sia installato
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog

# Build ottimizzato per Replicate
cog build

# Test del container
cog predict -i image=@input.jpg
```

### 3. Build con Script Automatico

```bash
# Usa lo script di build incluso
./start_seesr.sh build

# Test completo
./start_seesr.sh test
```

## Struttura Build Process

### Fase 1: System Dependencies
```dockerfile
# Python 3.10 + dipendenze sistema
apt-get install python3.10 python3.10-pip libgl1-mesa-glx ffmpeg...
```

### Fase 2: Python Environment  
```dockerfile
# Installa requirements.txt completo
pip install torch torchvision diffusers transformers accelerate...
```

### Fase 3: Model Download
```dockerfile
# Esegue scripts/download_models.py
python3 scripts/download_models.py
```

### Fase 4: Environment Validation
```dockerfile
# Esegue scripts/test_docker_env.py
python3 scripts/test_docker_env.py
```

## Script Inclusi

### `scripts/download_models.py`
- Download automatico di tutti i modelli necessari
- Gestione errori e retry logic
- Verifica integrità download
- Calcolo spazio utilizzato

### `scripts/test_docker_env.py`
- Test completo dell'ambiente Docker
- Verifica importazioni moduli
- Test CUDA availability
- Controllo directory e modelli
- Test moduli custom del progetto

## Variabili d'Ambiente

Il container imposta automaticamente:

```bash
# Python Path
PYTHONPATH="/src:${PYTHONPATH}"

# Cache Directories
HF_HOME="/root/.cache/huggingface"
TORCH_HOME="/root/.cache/torch"
HF_HUB_CACHE="/tmp/huggingface_cache"

# CUDA Optimizations
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
FORCE_CUDA="1"
TORCH_CUDNN_V8_API_ENABLED="1"

# Memory Management
PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
OMP_NUM_THREADS="4"
```

## Troubleshooting

### ❌ Problema: Build fallisce durante download modelli
```bash
# Soluzione: Build senza cache
docker build --no-cache -t seesr-sd-turbo .

# Oppure: Skip download durante build
docker build --build-arg SKIP_MODELS=true -t seesr-sd-turbo .
```

### ❌ Problema: Import errors per moduli custom
```bash
# I moduli custom vengono caricati a runtime, non durante build
# Questo è normale e non impedisce il funzionamento
```

### ❌ Problema: CUDA non disponibile
```bash
# Verifica driver NVIDIA
nvidia-smi

# Build con supporto CPU fallback
docker build -t seesr-sd-turbo .
```

### ❌ Problema: Out of memory durante build
```bash
# Aumenta memoria Docker
docker system prune -a
docker build --memory=8g -t seesr-sd-turbo .
```

## Health Check

Il container include un health check automatico:

```bash
# Check manuale
docker run --rm seesr-sd-turbo python3 -c "import torch, diffusers; print('OK')"

# Monitor health
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Deployment

### Replicate
```bash
# Push a Replicate (con account configurato)
cog push r8.im/username/seesr-sd-turbo
```

### Docker Hub
```bash
# Tag e push
docker tag seesr-sd-turbo username/seesr-sd-turbo:latest
docker push username/seesr-sd-turbo:latest
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: seesr-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: seesr
  template:
    metadata:
      labels:
        app: seesr
    spec:
      containers:
      - name: seesr
        image: seesr-sd-turbo:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            memory: "8Gi"
```

## Dimensioni e Performance

### Dimensioni Attese
- **Base image**: ~4GB (CUDA 11.8 Ubuntu 20.04)
- **Dependencies**: ~8GB (PyTorch, Diffusers, etc.)
- **Models**: ~15GB (SEESR + SD Turbo + RAM)
- **Total**: ~27GB

### Performance
- **Build time**: 15-30 minuti (dipende da rete)
- **Cold start**: 30-60 secondi
- **Warm inference**: 2-5 secondi (4 step SD Turbo)

## Utilizzo Avanzato

### Custom Model Path
```bash
# Mount custom models
docker run -v /local/models:/src/preset/models seesr-sd-turbo
```

### Development Mode
```bash
# Mount code per sviluppo
docker run -v $(pwd):/src -p 5000:5000 seesr-sd-turbo
```

### GPU Specifico
```bash
# Usa GPU specifico
docker run --gpus device=0 seesr-sd-turbo
```

---

🎯 **Il container è ora pronto per deployment in produzione con tutti i modelli pre-scaricati e l'ambiente completamente configurato!**
