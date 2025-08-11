# Esempi di Utilizzo SEESR con Ambiente Virtuale

## 🚀 Setup Iniziale

```bash
# Setup completo dell'ambiente
./start_seesr.sh setup

# Test dell'ambiente  
./start_seesr.sh test
```

## 💼 Utilizzo Quotidiano

### Attivazione Ambiente

```bash
# Opzione 1: Attivazione automatica tramite script
source activate_seesr.sh

# Opzione 2: Attivazione manuale
source seesr_env/bin/activate

# Verifica attivazione (dovresti vedere (seesr_env) nel prompt)
which python
```

### Esempi di Super-Resolution

```bash
# Test con immagine di esempio
./start_seesr.sh test

# Processamento singola immagine
./start_seesr.sh python -c "
from predict import Predictor
predictor = Predictor()
predictor.setup()
result = predictor.predict(
    image=open('input.jpg', 'rb'),
    prompt='high quality, detailed',
    num_inference_steps=2,
    guidance_scale=1.0,
    controlnet_conditioning_scale=0.8
)
result.save('output.jpg')
"

# Benchmark prestazioni
./start_seesr.sh benchmark
```

### Test dell'Ambiente

```bash
# Test completo dell'ambiente
python test_environment.py

# Test rapido delle dipendenze
python -c "
import torch
import diffusers
import transformers
print('✅ Dipendenze principali caricate')
print(f'PyTorch: {torch.__version__}')
print(f'Diffusers: {diffusers.__version__}')
print(f'Transformers: {transformers.__version__}')
"
```

## 🛠️ Sviluppo e Debug

### Installazione di Nuove Dipendenze

```bash
# Attiva l'ambiente
source activate_seesr.sh

# Installa nuove dipendenze
pip install nuova-dipendenza

# Aggiorna requirements.txt se necessario
pip freeze > requirements_new.txt
```

### Debug e Troubleshooting

```bash
# Verifica stato ambiente
./start_seesr.sh status

# Reinstalla dipendenze
./start_seesr.sh setup --force

# Reset completo dell'ambiente
rm -rf seesr_env
./start_seesr.sh setup
```

### Log e Monitoraggio

```bash
# Avvio con logging dettagliato
SEESR_LOG_LEVEL=DEBUG ./start_seesr.sh python your_script.py

# Monitoraggio utilizzo GPU (se disponibile)
watch -n 1 nvidia-smi

# Monitoraggio memoria
./start_seesr.sh python -c "
import psutil
import torch
print(f'RAM: {psutil.virtual_memory().percent}%')
if torch.cuda.is_available():
    print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB')
"
```

## 🎯 Casi d'Uso Comuni

### Batch Processing

```python
# batch_process.py
from predict import Predictor
import os
from PIL import Image

predictor = Predictor()
predictor.setup()

input_dir = "input_images"
output_dir = "output_images"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"seesr_{filename}")
        
        with open(input_path, 'rb') as f:
            result = predictor.predict(
                image=f,
                prompt="high quality, sharp details",
                num_inference_steps=2
            )
            result.save(output_path)
        
        print(f"Processato: {filename}")
```

### Script Personalizzato

```python
# custom_seesr.py
import sys
sys.path.append('.')

from predict import Predictor
from PIL import Image
import argparse

def main():
    parser = argparse.ArgumentParser(description="SEESR Custom Script")
    parser.add_argument("--input", required=True, help="Immagine input")
    parser.add_argument("--output", required=True, help="Immagine output")
    parser.add_argument("--prompt", default="high quality", help="Prompt")
    parser.add_argument("--steps", type=int, default=2, help="Inference steps")
    
    args = parser.parse_args()
    
    predictor = Predictor()
    predictor.setup()
    
    with open(args.input, 'rb') as f:
        result = predictor.predict(
            image=f,
            prompt=args.prompt,
            num_inference_steps=args.steps
        )
        result.save(args.output)
    
    print(f"Super-resolution completata: {args.output}")

if __name__ == "__main__":
    main()
```

Utilizzo:
```bash
# Attiva l'ambiente e esegui
source activate_seesr.sh
python custom_seesr.py --input input.jpg --output output.jpg --prompt "photorealistic, 4k"
```

## 🔧 Manutenzione

### Aggiornamento Dipendenze

```bash
# Backup requirements attuali
cp requirements.txt requirements_backup.txt

# Aggiorna all'ultima versione
source activate_seesr.sh
pip install --upgrade -r requirements.txt

# Test dopo aggiornamento
python test_environment.py
```

### Pulizia Cache

```bash
# Pulisci cache Python
find . -type d -name "__pycache__" -exec rm -rf {} +

# Pulisci cache modelli (attenzione: ridownload necessario)
rm -rf ~/.cache/huggingface/

# Pulisci file temporanei
rm -rf temp/ tmp/ output/
```

## 📋 Checklist Rapida

Prima di ogni sessione di lavoro:

- [ ] `source activate_seesr.sh` - Attiva l'ambiente
- [ ] `python test_environment.py` - Verifica dipendenze
- [ ] `./start_seesr.sh test` - Test funzionalità base
- [ ] Verifica spazio disco per output

Per problemi:

- [ ] `./start_seesr.sh setup --force` - Reinstalla ambiente
- [ ] Controlla log per errori
- [ ] Verifica versioni dipendenze con `pip list`
- [ ] Test su immagine semplice prima di batch processing
