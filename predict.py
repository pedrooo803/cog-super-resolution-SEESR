"""
Shim di compatibilità per i test: re-esporta Predictor da cog/predict.py
senza dipendere dal pacchetto esterno "cog" installato via pip.
"""

import importlib.util
import sys
from pathlib import Path

_cog_predict_path = Path(__file__).parent / "cog" / "predict.py"
spec = importlib.util.spec_from_file_location("seesr_cog_predict", str(_cog_predict_path))
if spec is None or spec.loader is None:
	raise ImportError(f"Impossibile caricare il modulo da {_cog_predict_path}")
_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _module
spec.loader.exec_module(_module)

Predictor = _module.Predictor  # re-export

