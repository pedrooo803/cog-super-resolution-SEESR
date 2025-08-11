#!/usr/bin/env python3
"""
Script per il download dei modelli necessari per SEESR
Eseguito durante il build del docker container
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    """Download tutti i modelli necessari per SEESR"""
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub non installato. Installazione...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    # Definisci le directory dei modelli
    models_config = [
        {
            'repo_id': 'alexnasa/SEESR',
            'local_dir': 'deployment/preset/models/seesr',
            'description': 'SEESR model'
        },
        {
            'repo_id': 'stabilityai/sd-turbo',
            'local_dir': 'deployment/preset/models/sd-turbo', 
            'description': 'SD Turbo model'
        },
        {
            'repo_id': 'xinyu1205/recognize_anything_model',
            'local_dir': 'deployment/preset/models/ram',
            'description': 'RAM model for tagging'
        }
    ]
    
    logger.info("🚀 Inizio download dei modelli...")
    
    success_count = 0
    
    for model_config in models_config:
        try:
            logger.info(f"📥 Downloading {model_config['description']}...")
            
            # Crea la directory se non esiste
            Path(model_config['local_dir']).mkdir(parents=True, exist_ok=True)
            
            # Download del modello
            snapshot_download(
                repo_id=model_config['repo_id'],
                local_dir=model_config['local_dir'],
                cache_dir='/tmp/huggingface_cache',
                resume_download=True
            )
            
            logger.info(f"✅ {model_config['description']} scaricato con successo")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Errore nel download di {model_config['description']}: {e}")
            continue
    
    logger.info(f"🎉 Download completato: {success_count}/{len(models_config)} modelli scaricati")
    
    # Verifica spazio utilizzato
    logger.info("📊 Spazio utilizzato dai modelli:")
    try:
        for model_config in models_config:
            if Path(model_config['local_dir']).exists():
                # Calcola dimensione directory
                total_size = sum(f.stat().st_size for f in Path(model_config['local_dir']).rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                logger.info(f"  - {model_config['description']}: {size_mb:.1f} MB")
    except Exception as e:
        logger.warning(f"Non riesco a calcolare lo spazio utilizzato: {e}")
    
    return success_count == len(models_config)

def verify_environment():
    """Verifica che l'ambiente sia configurato correttamente"""
    
    logger.info("🔍 Verifica dell'ambiente...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'Torchvision', 
        'diffusers': 'Diffusers',
        'transformers': 'Transformers',
        'accelerate': 'Accelerate',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'timm': 'Timm',
        'safetensors': 'SafeTensors',
        'omegaconf': 'OmegaConf',
        'einops': 'Einops',
        'huggingface_hub': 'Hugging Face Hub',
        'PyWavelets': 'PyWavelets'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                __import__(package)
            logger.info(f"✅ {name}")
        except ImportError:
            logger.error(f"❌ {name}")
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"Pacchetti mancanti: {', '.join(missing_packages)}")
        return False
    
    # Verifica CUDA
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        logger.info(f"✅ CUDA disponibile: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"✅ CUDA version: {torch.version.cuda}")
    except Exception as e:
        logger.warning(f"Errore nella verifica CUDA: {e}")
    
    logger.info("🎯 Ambiente verificato con successo!")
    return True

if __name__ == "__main__":
    logger.info("Script di download e verifica modelli SEESR")
    
    # Download dei modelli
    download_success = download_models()
    
    # Verifica dell'ambiente
    env_success = verify_environment()
    
    if download_success and env_success:
        logger.info("🎉 Setup completato con successo!")
        sys.exit(0)
    else:
        logger.error("⚠️ Setup completato con alcuni errori")
        sys.exit(1)
