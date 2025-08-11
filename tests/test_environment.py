#!/usr/bin/env python3
"""
Script di test veloce per verificare l'ambiente SEESR
"""

import sys
import os

def test_environment():
    """Testa l'ambiente virtuale e le dipendenze principali"""
    
    print("🧪 Test Ambiente SEESR")
    print("=" * 50)
    
    # Test Python
    print(f"✅ Python: {sys.version}")
    print(f"✅ Percorso eseguibile: {sys.executable}")
    
    # Test virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Ambiente virtuale: ATTIVO")
        print(f"   Percorso env: {sys.prefix}")
    else:
        print("⚠️  Ambiente virtuale: NON ATTIVO")
    
    # Test dipendenze critiche
    print("\n📦 Test Dipendenze:")
    
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),  
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy")
    ]
    
    failed_imports = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    # Test moduli SEESR
    print("\n🎯 Test Moduli SEESR:")
    
    # Test se i moduli custom esistono
    seesr_modules = [
        ("models", "Modelli Custom"),
        ("pipelines", "Pipeline SEESR"),
        ("ram", "RAM Model"),
        ("utils", "Utilità")
    ]
    
    for module, name in seesr_modules:
        if os.path.exists(module):
            print(f"✅ {name}: Directory trovata")
        else:
            print(f"⚠️  {name}: Directory non trovata")
    
    # Verifica predict.py
    if os.path.exists("predict.py"):
        print("✅ Predictor principale: predict.py trovato")
    else:
        print("❌ Predictor principale: predict.py NON trovato")
    
    # Summary
    print("\n" + "=" * 50)
    if failed_imports:
        print(f"⚠️  Dipendenze mancanti: {', '.join(failed_imports)}")
        print("   Esegui: ./start_seesr.sh setup")
        return False
    else:
        print("🎉 Ambiente SEESR configurato correttamente!")
        print("   Pronto per l'uso")
        return True

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
