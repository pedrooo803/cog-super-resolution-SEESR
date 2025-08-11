"""
Utility per gestire xformers opzionale
Gestisce l'importazione condizionale di xformers e fornisce fallback
"""

import warnings
from typing import Optional, Any

# Flag globale per indicare se xformers è disponibile
XFORMERS_AVAILABLE = False

try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
    print("✅ xformers disponibile - attention ottimizzato abilitato")
except ImportError:
    print("⚠️  xformers non disponibile - utilizzo attention standard")
    warnings.warn(
        "xformers non è installato. Le prestazioni potrebbero essere ridotte. "
        "Su macOS, xformers può avere problemi di compilazione. "
        "Il sistema continuerà a funzionare con attention standard."
    )

def is_xformers_available() -> bool:
    """Verifica se xformers è disponibile"""
    return XFORMERS_AVAILABLE

def get_attention_processor() -> Optional[Any]:
    """
    Restituisce il processore di attention appropriato
    - Se xformers è disponibile: AttnProcessor2_0 (ottimizzato)
    - Altrimenti: AttnProcessor (standard)
    """
    if XFORMERS_AVAILABLE:
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            return AttnProcessor2_0()
        except ImportError:
            pass
    
    # Fallback al processore standard
    try:
        from diffusers.models.attention_processor import AttnProcessor
        return AttnProcessor()
    except ImportError:
        return None

def enable_xformers_attention(model: Any) -> bool:
    """
    Abilita l'attention ottimizzato se possibile
    
    Args:
        model: Modello da ottimizzare (UNet, ControlNet, etc.)
        
    Returns:
        bool: True se l'ottimizzazione è stata abilitata
    """
    if not XFORMERS_AVAILABLE:
        return False
    
    try:
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            model.enable_xformers_memory_efficient_attention()
            print(f"✅ xformers attention abilitato per {type(model).__name__}")
            return True
        else:
            print(f"⚠️  {type(model).__name__} non supporta xformers attention")
            return False
    except Exception as e:
        print(f"❌ Errore abilitando xformers per {type(model).__name__}: {e}")
        return False

def optimize_models_attention(*models: Any) -> int:
    """
    Ottimizza l'attention per più modelli
    
    Args:
        *models: Lista di modelli da ottimizzare
        
    Returns:
        int: Numero di modelli ottimizzati con successo
    """
    optimized_count = 0
    
    for model in models:
        if model is not None and enable_xformers_attention(model):
            optimized_count += 1
    
    if optimized_count > 0:
        print(f"🚀 {optimized_count}/{len(models)} modelli ottimizzati con xformers")
    else:
        print("ℹ️  Utilizzo attention standard (nessuna ottimizzazione xformers)")
    
    return optimized_count

def get_memory_efficient_attention_kwargs() -> dict:
    """
    Restituisce i kwargs appropriati per l'attention efficiente
    """
    if XFORMERS_AVAILABLE:
        return {
            "enable_xformers_memory_efficient_attention": True,
            "attention_op": None  # Usa il default di xformers
        }
    else:
        return {
            "enable_xformers_memory_efficient_attention": False
        }

# Informazioni di compatibilità
COMPATIBILITY_INFO = {
    "xformers_available": XFORMERS_AVAILABLE,
    "fallback_attention": not XFORMERS_AVAILABLE,
    "performance_impact": "Minimo" if XFORMERS_AVAILABLE else "Moderato",
    "recommendations": [
        "xformers opzionale per prestazioni ottimali",
        "Funzionalità completa disponibile senza xformers",
        "Su macOS, installation di xformers può essere problematica"
    ]
}

def print_attention_status():
    """Stampa lo stato dell'attention system"""
    print("\n🔧 Status Attention System:")
    print(f"   xformers: {'✅ Disponibile' if XFORMERS_AVAILABLE else '❌ Non disponibile'}")
    print(f"   Fallback: {'Standard PyTorch attention' if not XFORMERS_AVAILABLE else 'Non necessario'}")
    print(f"   Performance: {COMPATIBILITY_INFO['performance_impact']} impact")
    
    if not XFORMERS_AVAILABLE:
        print("\n💡 Nota: Il sistema funziona completamente senza xformers.")
        print("   Le prestazioni potrebbero essere leggermente ridotte.")

if __name__ == "__main__":
    print_attention_status()
