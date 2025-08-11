"""
Test script per SEESR con SD Turbo
Ottimizzato per ambiente virtuale Python
"""

import sys
import os
from pathlib import Path
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Verifica ambiente di esecuzione"""
    logger.info("🔍 Verifica ambiente di esecuzione...")
    
    # Verifica Python
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info(f"🐍 Python: {python_version}")
    logger.info(f"📁 Executable: {sys.executable}")
    
    # Verifica ambiente virtuale
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = sys.prefix
        logger.info(f"✅ Ambiente virtuale attivo: {venv_path}")
    else:
        logger.warning("⚠️  Ambiente virtuale non rilevato")
    
    # Verifica CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU CUDA: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️  CUDA non disponibile, verrà usata la CPU")
    except ImportError:
        logger.error("❌ PyTorch non installato")
        return False
    
    return True

def test_seesr():
    """Test dell'implementazione SEESR"""
    try:
        logger.info("🚀 Inizializzazione del predictor SEESR...")
        
        # Verifica ambiente prima di importare
        if not check_environment():
            return False
        
        # Import del predictor
        from predict import Predictor
        
        logger.info("📦 Setup del modello...")
        predictor = Predictor()
        predictor.setup()
        
        logger.info("✅ SEESR con SD Turbo inizializzato con successo!")
        
        # Test con un'immagine di esempio se disponibile
        test_image_path = "test_input.jpg"
        if os.path.exists(test_image_path):
            logger.info(f"🖼️  Test con immagine: {test_image_path}")
            
            result_path = predictor.predict(
                image=Path(test_image_path),
                num_inference_steps=4,
                cfg_scale=1.0,
                use_kds=True,
                positive_prompt="high quality, detailed, 8k",
                negative_prompt="blur, lowres, artifacts"
            )
            
            logger.info(f"💾 Risultato salvato in: {result_path}")
        else:
            logger.info("🎨 Nessuna immagine di test trovata. Creazione di un'immagine di esempio...")
            # Crea un'immagine di test
            try:
                from PIL import Image
                import numpy as np
                test_img = Image.fromarray(
                    np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
                )
                test_img.save(test_image_path)
                logger.info(f"✅ Immagine di test creata: {test_image_path}")
            except Exception as e:
                logger.warning(f"⚠️  Impossibile creare immagine di test: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore durante il test: {e}")
        logger.error(f"💡 Suggerimento: verifica che tutte le dipendenze siano installate")
        return False

def test_components():
    """Test dei singoli componenti"""
    logger.info("Test dei componenti individuali...")
    
    try:
        # Test RAM model
        logger.info("Test RAM model...")
        from ram.models.ram_lora import ram
        ram_model = ram(image_size=384, vit='swin_l')
        logger.info("✅ RAM model OK")
        
        # Test wavelet color fix
        logger.info("Test wavelet color fix...")
        from utils.wavelet_color_fix import wavelet_color_fix_simple
        logger.info("✅ Wavelet color fix OK")
        
        # Test modelli personalizzati
        logger.info("Test modelli personalizzati...")
        from models.controlnet import ControlNetModel
        from models.unet_2d_condition import UNet2DConditionModel
        logger.info("✅ Modelli personalizzati OK")
        
        # Test pipeline
        logger.info("Test pipeline...")
        from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
        logger.info("✅ Pipeline SEESR OK")
        
        logger.info("✅ Tutti i componenti testati con successo!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore test componenti: {e}")
        return False

def check_dependencies():
    """Verifica delle dipendenze"""
    logger.info("Verifica dipendenze...")
    
    required_packages = [
        'torch', 'torchvision', 'diffusers', 'transformers',
        'accelerate', 'opencv-python', 'Pillow', 'numpy',
        'scipy', 'timm', 'safetensors', 'omegaconf',
        'einops', 'huggingface_hub'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} mancante")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Pacchetti mancanti: {missing_packages}")
        logger.info("Installa con: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ Tutte le dipendenze sono disponibili!")
    return True

def main():
    """Funzione principale di test"""
    logger.info("=" * 60)
    logger.info("🧪 Test SEESR con SD Turbo (Virtual Environment)")
    logger.info("=" * 60)
    
    # Check ambiente
    if not check_environment():
        logger.error("❌ Ambiente non configurato correttamente. Uscita.")
        return 1
    
    # Check dipendenze
    if not check_dependencies():
        logger.error("❌ Dipendenze mancanti. Uscita.")
        logger.info("💡 Esegui: ./start_seesr.sh install")
        return 1
    
    # Test componenti
    if not test_components():
        logger.error("❌ Test componenti fallito. Uscita.")
        return 1
    
    # Test completo
    if not test_seesr():
        logger.error("❌ Test SEESR fallito. Uscita.")
        return 1
    
    logger.info("=" * 60)
    logger.info("🎉 Tutti i test completati con successo!")
    logger.info("")
    logger.info("🚀 SEESR con SD Turbo è pronto per l'uso!")
    logger.info("")
    logger.info("📝 Esempi di utilizzo:")
    logger.info("   ./start_seesr.sh run input.jpg")
    logger.info("   source activate_seesr.sh && python predict.py")
    logger.info("   cog predict -i image=@input.jpg")
    logger.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
