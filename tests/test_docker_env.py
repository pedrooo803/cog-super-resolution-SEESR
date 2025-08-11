#!/usr/bin/env python3
"""
Script di test per verificare che l'ambiente Docker sia configurato correttamente
Da utilizzare durante il build del container per validare l'installazione
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Testa che tutti i moduli necessari siano importabili"""
    print("🔍 Test importazione moduli...")
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('torchvision', 'Torchvision'),
        ('diffusers', 'Diffusers'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('timm', 'Timm'),
        ('safetensors', 'SafeTensors'),
        ('omegaconf', 'OmegaConf'),
        ('einops', 'Einops'),
        ('huggingface_hub', 'Hugging Face Hub'),
        ('PyWavelets', 'PyWavelets')
    ]
    
    failed_imports = []
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def test_cuda():
    """Testa la disponibilità di CUDA"""
    print("\n🔍 Test CUDA...")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"  ✅ CUDA disponibile: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"  ✅ Numero GPU: {device_count}")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  ✅ GPU {i}: {gpu_name} ({memory_total:.1f} GB)")
        
        print(f"  ✅ PyTorch version: {torch.__version__}")
        print(f"  ✅ CUDA version: {torch.version.cuda}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Errore CUDA: {e}")
        return False

def test_models_download():
    """Verifica che i modelli siano stati scaricati"""
    print("\n🔍 Test modelli scaricati...")
    
    model_dirs = [
        ('deployment/preset/models/seesr', 'SEESR'),
        ('deployment/preset/models/sd-turbo', 'SD Turbo'),
        ('deployment/preset/models/ram', 'RAM'),
    ]
    
    all_present = True
    
    for model_dir, name in model_dirs:
        path = Path(model_dir)
        if path.exists() and any(path.iterdir()):
            print(f"  ✅ {name}: {model_dir}")
        else:
            print(f"  ❌ {name}: {model_dir} (non trovato o vuoto)")
            all_present = False
    
    return all_present

def test_custom_modules():
    """Testa che i moduli custom del progetto siano importabili"""
    print("\n🔍 Test moduli custom...")
    
    custom_modules = [
        ('utils.xformers_utils', 'XFormers Utils'),
        ('utils.wavelet_color_fix', 'Wavelet Color Fix'),
        ('models.unet_2d_condition', 'Custom UNet2D'),
        ('models.controlnet', 'Custom ControlNet'),
        ('pipelines.pipeline_seesr', 'SEESR Pipeline'),
        ('ram.models.ram_lora', 'RAM LoRA'),
        ('ram.inference_ram', 'RAM Inference'),
    ]
    
    failed_custom = []
    
    for module, name in custom_modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ⚠️ {name}: {e}")
            failed_custom.append(name)
    
    # I moduli custom possono non essere disponibili al momento del build
    return True, failed_custom

def test_directories():
    """Verifica che le directory necessarie esistano"""
    print("\n🔍 Test directory...")
    
    required_dirs = [
        'deployment/preset/models',
        'output',
        '/root/.cache/torch',
        '/root/.cache/huggingface',
        '/tmp/huggingface_cache'
    ]
    
    all_exist = True
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            all_exist = False
    
    return all_exist

def main():
    """Esegue tutti i test"""
    print("🚀 Test ambiente SEESR Docker")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test importazioni
    imports_ok, failed_imports = test_imports()
    if not imports_ok:
        print(f"\n❌ Importazioni fallite: {', '.join(failed_imports)}")
        all_tests_passed = False
    
    # Test CUDA
    cuda_ok = test_cuda()
    if not cuda_ok:
        print("\n⚠️ CUDA non disponibile o problematico")
        # Non consideriamo CUDA come errore fatale per il build
    
    # Test directory
    dirs_ok = test_directories()
    if not dirs_ok:
        print("\n❌ Alcune directory necessarie non esistono")
        all_tests_passed = False
    
    # Test modelli
    models_ok = test_models_download()
    if not models_ok:
        print("\n⚠️ Alcuni modelli non sono stati scaricati")
        # Non consideriamo i modelli mancanti come errore fatale
    
    # Test moduli custom
    custom_ok, failed_custom = test_custom_modules()
    if failed_custom:
        print(f"\n⚠️ Moduli custom non disponibili: {', '.join(failed_custom)}")
        print("   (Normale durante il build - saranno disponibili a runtime)")
    
    print("\n" + "=" * 50)
    
    if all_tests_passed:
        print("🎉 Tutti i test critici sono passati!")
        print("🐳 Container pronto per l'uso")
        return 0
    else:
        print("❌ Alcuni test critici sono falliti")
        print("⚠️ Il container potrebbe non funzionare correttamente")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Errore durante i test: {e}")
        print("\n📍 Stack trace:")
        traceback.print_exc()
        sys.exit(1)
