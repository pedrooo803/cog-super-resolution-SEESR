#!/bin/bash

# Script di avvio rapido per SEESR con SD Turbo
# Usage: ./start_seesr.sh [input_image] [options]
# Utilizza un ambiente virtuale Python isolato per massima consistenza

set -e

echo "🚀 SEESR con SD Turbo - Avvio Rapido (Virtual Environment)"
echo "=========================================================="

# Configurazione ambiente virtuale
VENV_NAME="seesr_env"
VENV_PATH="./${VENV_NAME}"
PYTHON_EXECUTABLE="${VENV_PATH}/bin/python"
PIP_EXECUTABLE="${VENV_PATH}/bin/pip"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Funzione per stampe colorate
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_venv() {
    echo -e "${PURPLE}🐍 $1${NC}"
}

# Funzione per verificare se siamo nell'ambiente virtuale
check_venv() {
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        return 0  # Già nell'ambiente virtuale
    elif [[ -f "$PYTHON_EXECUTABLE" ]]; then
        return 1  # Ambiente virtuale esiste ma non attivato
    else
        return 2  # Ambiente virtuale non esiste
    fi
}

# Funzione per creare l'ambiente virtuale
create_venv() {
    print_venv "Creazione ambiente virtuale: $VENV_NAME"
    
    # Verifica che python3 abbia il modulo venv
    if ! python3 -m venv --help &> /dev/null; then
        print_error "Modulo venv non disponibile. Installa python3-venv:"
        print_info "Ubuntu/Debian: sudo apt install python3-venv"
        print_info "macOS: brew install python3 (dovrebbe già includerlo)"
        exit 1
    fi
    
    # Crea l'ambiente virtuale
    python3 -m venv "$VENV_PATH"
    
    # Aggiorna pip nell'ambiente virtuale
    "$PIP_EXECUTABLE" install --upgrade pip
    
    print_success "Ambiente virtuale creato: $VENV_PATH"
}

# Funzione per attivare l'ambiente virtuale
activate_venv() {
    if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
        print_error "Ambiente virtuale non trovato in $VENV_PATH"
        return 1
    fi
    
    source "$VENV_PATH/bin/activate"
    print_venv "Ambiente virtuale attivato: $VENV_NAME"
    
    # Aggiorna le variabili dei percorsi
    PYTHON_EXECUTABLE="python"
    PIP_EXECUTABLE="pip"
}

# Funzione per installare le dipendenze nell'ambiente virtuale
install_dependencies() {
    print_venv "Installazione dipendenze nell'ambiente virtuale..."
    
    # Installa wheel per evitare problemi di compilazione
    "$PIP_EXECUTABLE" install wheel
    
    # Installa le dipendenze principali
    "$PIP_EXECUTABLE" install -r requirements.txt
    
    # Installa dipendenze aggiuntive specifiche per SEESR
    "$PIP_EXECUTABLE" install PyWavelets opencv-python-headless
    
    # Installa il progetto in modalità editable
    "$PIP_EXECUTABLE" install -e .
    
    print_success "Dipendenze installate nell'ambiente virtuale"
}

# Funzione per verificare le dipendenze
verify_dependencies() {
    print_venv "Verifica dipendenze nell'ambiente virtuale..."
    
    local required_packages=(
        "torch" "torchvision" "diffusers" "transformers"
        "accelerate" "opencv-python" "Pillow" "numpy"
        "scipy" "timm" "safetensors" "omegaconf"
        "einops" "huggingface_hub" "PyWavelets"
    )
    
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if "$PYTHON_EXECUTABLE" -c "import ${package//-/_}" &> /dev/null; then
            echo -e "  ✅ $package"
        else
            echo -e "  ❌ $package"
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -eq 0 ]]; then
        print_success "Tutte le dipendenze sono disponibili!"
        return 0
    else
        print_error "Dipendenze mancanti: ${missing_packages[*]}"
        return 1
    fi
}

# Funzione per setup completo dell'ambiente
setup_environment() {
    print_info "Setup ambiente di sviluppo SEESR..."
    
    # Controlla stato ambiente virtuale
    set +e  # Disabilita temporaneamente set -e per gestire i codici di ritorno
    check_venv
    local venv_status=$?
    set -e  # Riabilita set -e
    
    case $venv_status in
        0)
            print_venv "Già nell'ambiente virtuale attivato"
            ;;
        1)
            print_venv "Ambiente virtuale esistente trovato, attivazione..."
            activate_venv
            ;;
        2)
            print_venv "Ambiente virtuale non trovato, creazione..."
            create_venv
            activate_venv
            install_dependencies
            ;;
    esac
    
    # Verifica se le dipendenze devono essere installate/aggiornate
    if [[ ! -f "${VENV_PATH}/.deps_installed" ]] || [[ requirements.txt -nt "${VENV_PATH}/.deps_installed" ]]; then
        print_venv "Aggiornamento dipendenze necessario..."
        install_dependencies
        touch "${VENV_PATH}/.deps_installed"
    fi
    
    # Verifica dipendenze
    if ! verify_dependencies; then
        print_warning "Alcune dipendenze mancano, reinstallazione..."
        install_dependencies
        touch "${VENV_PATH}/.deps_installed"
    fi
    
    print_success "Ambiente di sviluppo pronto!"
}

# Verifica se Python è installato
if ! command -v python3 &> /dev/null; then
    print_error "Python3 non trovato. Installa Python 3.9 o superiore."
    exit 1
fi

print_success "Python3 trovato: $(python3 --version)"

# Verifica versione Python
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.9"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    print_error "Python 3.9+ richiesto, trovato: $python_version"
    exit 1
fi

print_success "Versione Python compatibile: $python_version"

# Verifica CUDA (opzionale)
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU rilevata"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    print_warning "NVIDIA GPU non rilevata. Verrà usata la CPU (più lento)."
fi

# Setup dell'ambiente virtuale
setup_environment

# Verifica se Cog è installato (per deployment)
if command -v cog &> /dev/null; then
    print_success "Cog trovato - modalità deployment disponibile"
    COG_AVAILABLE=true
else
    print_warning "Cog non trovato - solo modalità locale"
    COG_AVAILABLE=false
fi

# Funzione per testare il sistema
test_system() {
    print_info "Test del sistema nell'ambiente virtuale..."
    
    # Assicurati che siamo nell'ambiente virtuale
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        activate_venv
    fi
    
    if "$PYTHON_EXECUTABLE" test_seesr.py; then
        print_success "Test sistema completato con successo!"
        return 0
    else
        print_error "Test sistema fallito"
        return 1
    fi
}

# Funzione per eseguire super-resolution
run_super_resolution() {
    local input_image="$1"
    local output_dir="${2:-output}"
    
    if [ ! -f "$input_image" ]; then
        print_error "File immagine non trovato: $input_image"
        return 1
    fi
    
    print_info "Esecuzione super-resolution su: $input_image"
    
    # Crea directory output se non esistente
    mkdir -p "$output_dir"
    
    # Assicurati che siamo nell'ambiente virtuale
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        activate_venv
    fi
    
    if [ "$COG_AVAILABLE" = true ]; then
        # Usa Cog se disponibile
        print_info "Usando Cog per l'inferenza..."
        cog predict -i image=@"$input_image" -o "$output_dir/result.png"
    else
        # Usa Python nell'ambiente virtuale
        print_info "Usando Python (venv) per l'inferenza..."
        "$PYTHON_EXECUTABLE" -c "
import sys
import os
sys.path.insert(0, os.getcwd())

from predict import Predictor
from pathlib import Path

print('🐍 Ambiente Python:', sys.executable)
print('📦 Versione Python:', sys.version)

predictor = Predictor()
predictor.setup()

result = predictor.predict(
    image=Path('$input_image'),
    num_inference_steps=4,
    cfg_scale=1.0,
    use_kds=True,
    positive_prompt='high quality, detailed, 8k, masterpiece',
    negative_prompt='blur, lowres, artifacts, noise'
)

print(f'✅ Risultato salvato in: {result}')
"
    fi
    
    print_success "Super-resolution completata!"
}

# Funzione per mostrare l'help
show_help() {
    echo "Uso: $0 [COMANDO] [OPZIONI]"
    echo ""
    echo "COMANDI:"
    echo "  setup                   - Setup ambiente virtuale"
    echo "  test                    - Testa il sistema"
    echo "  run <immagine>         - Esegui super-resolution"
    echo "  build                  - Build container Cog"
    echo "  benchmark              - Esegui benchmark"
    echo "  clean                  - Pulisci cache e file temporanei"
    echo "  shell                  - Apri shell nell'ambiente virtuale"
    echo "  install                - Installa/aggiorna dipendenze"
    echo "  status                 - Mostra stato ambiente"
    echo ""
    echo "ESEMPI:"
    echo "  $0 setup               # Prima configurazione"
    echo "  $0 test                # Test del sistema"
    echo "  $0 run input.jpg       # Super-resolution"
    echo "  $0 shell               # Shell interattiva"
    echo "  $0 build               # Build per deployment"
    echo ""
    echo "AMBIENTE VIRTUALE:"
    echo "  Ubicazione: $VENV_PATH"
    echo "  Python: $PYTHON_EXECUTABLE"
    echo "  Pip: $PIP_EXECUTABLE"
    echo ""
    echo "Per opzioni avanzate:"
    echo "  source $VENV_PATH/bin/activate && python predict.py"
    echo "  cog predict -i image=@input.jpg"
}

# Funzione per aprire shell nell'ambiente virtuale
open_shell() {
    if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
        print_error "Ambiente virtuale non trovato. Esegui: $0 setup"
        return 1
    fi
    
    print_venv "Apertura shell nell'ambiente virtuale..."
    print_info "Per uscire dalla shell, digita 'exit'"
    print_info "Python disponibile come: python"
    print_info "Pip disponibile come: pip"
    echo ""
    
    # Attiva l'ambiente e apri una nuova shell
    export PS1="(${VENV_NAME}) \u@\h:\w\$ "
    bash --init-file <(echo "source '$VENV_PATH/bin/activate'; echo '🐍 Ambiente virtuale SEESR attivato'")
}

# Funzione per mostrare stato ambiente
show_status() {
    echo "📊 Stato Ambiente SEESR"
    echo "========================"
    echo ""
    
    # Stato ambiente virtuale
    if [[ -d "$VENV_PATH" ]]; then
        print_success "Ambiente virtuale: $VENV_PATH"
        echo "  Python: $("$VENV_PATH/bin/python" --version 2>&1)"
        echo "  Pip: $("$VENV_PATH/bin/pip" --version | cut -d' ' -f1-2)"
    else
        print_warning "Ambiente virtuale non trovato"
    fi
    
    echo ""
    
    # Stato dipendenze
    if [[ -f "${VENV_PATH}/.deps_installed" ]]; then
        print_success "Dipendenze installate: $(date -r "${VENV_PATH}/.deps_installed" '+%Y-%m-%d %H:%M:%S')"
    else
        print_warning "Dipendenze non installate"
    fi
    
    echo ""
    
    # GPU status
    if command -v nvidia-smi &> /dev/null; then
        print_success "GPU NVIDIA disponibile"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    else
        print_warning "GPU NVIDIA non rilevata"
    fi
    
    echo ""
    
    # Spazio disco
    local disk_usage=$(du -sh . 2>/dev/null | cut -f1)
    echo "💾 Spazio utilizzato: $disk_usage"
    
    # File presenti
    echo "📁 File progetto:"
    echo "  predict.py: $(if [[ -f predict.py ]]; then echo "✅"; else echo "❌"; fi)"
    echo "  requirements.txt: $(if [[ -f requirements.txt ]]; then echo "✅"; else echo "❌"; fi)"
    echo "  cog.yaml: $(if [[ -f cog.yaml ]]; then echo "✅"; else echo "❌"; fi)"
}

# Funzione per installazione/aggiornamento forzato
force_install() {
    print_venv "Installazione/aggiornamento forzato dipendenze..."
    
    if [[ ! -d "$VENV_PATH" ]]; then
        create_venv
    fi
    
    activate_venv
    install_dependencies
    touch "${VENV_PATH}/.deps_installed"
    
    print_success "Installazione completata!"
}

# Funzione per build Cog
build_cog() {
    if [ "$COG_AVAILABLE" = true ]; then
        print_info "Build container Cog..."
        cog build
        print_success "Build Cog completata!"
    else
        print_error "Cog non disponibile. Installa Cog per il deployment."
        return 1
    fi
}

# Funzione per benchmark
run_benchmark() {
    print_info "Esecuzione benchmark nell'ambiente virtuale..."
    
    # Assicurati che siamo nell'ambiente virtuale
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        activate_venv
    fi
    
    # Crea immagine di test se non esiste
    if [ ! -f "benchmark_input.jpg" ]; then
        "$PYTHON_EXECUTABLE" -c "
from PIL import Image
import numpy as np

# Crea un'immagine di test
img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
img.save('benchmark_input.jpg')
print('🖼️ Immagine di test creata: benchmark_input.jpg')
"
    fi
    
    # Esegui benchmark con informazioni dettagliate
    print_info "Misurando performance..."
    echo "🔧 Configurazione:"
    echo "  - Python: $("$PYTHON_EXECUTABLE" --version)"
    echo "  - Ambiente: $VENV_NAME"
    if command -v nvidia-smi &> /dev/null; then
        echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    fi
    echo ""
    
    start_time=$(date +%s)
    
    run_super_resolution "benchmark_input.jpg" "benchmark_output"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    print_success "Benchmark completato in ${duration} secondi"
    
    # Mostra statistiche addizionali
    if [[ -f "benchmark_output/result.png" ]]; then
        local file_size=$(du -h "benchmark_output/result.png" | cut -f1)
        print_info "Dimensione output: $file_size"
    fi
}

# Funzione per pulizia
clean_cache() {
    print_info "Pulizia cache e file temporanei..."
    
    # Rimuovi cache Python
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Rimuovi file temporanei
    rm -f benchmark_input.jpg
    rm -rf benchmark_output/
    rm -rf output/
    
    # Pulizia cache modelli (opzionale)
    read -p "Vuoi pulire anche la cache dei modelli? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/.cache/torch/
        rm -rf ~/.cache/huggingface/
        rm -rf preset/models/*/
        print_success "Cache modelli pulita"
    fi
    
    # Pulizia ambiente virtuale (opzionale)
    read -p "Vuoi rimuovere l'ambiente virtuale? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        print_success "Ambiente virtuale rimosso"
    fi
    
    print_success "Pulizia completata"
}

# Parsing argomenti
case "${1:-help}" in
    "setup")
        setup_environment
        show_status
        ;;
    "test")
        setup_environment
        test_system
        ;;
    "run")
        if [ -z "$2" ]; then
            print_error "Specifica un file immagine"
            echo "Uso: $0 run <immagine>"
            exit 1
        fi
        setup_environment
        run_super_resolution "$2" "${3:-output}"
        ;;
    "build")
        setup_environment
        build_cog
        ;;
    "benchmark")
        setup_environment
        run_benchmark
        ;;
    "clean")
        clean_cache
        ;;
    "shell")
        setup_environment
        open_shell
        ;;
    "status")
        show_status
        ;;
    "install")
        force_install
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Comando sconosciuto: $1"
        show_help
        exit 1
        ;;
esac

print_success "Operazione completata!"
echo ""
print_venv "Ambiente virtuale: $VENV_NAME"
print_info "Per ulteriori informazioni: $0 help"
