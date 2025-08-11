#!/bin/bash

# Script per build e test del Docker container SEESR
# Usage: ./docker_build.sh [build|test|run|deploy]

set -e

# Configurazione
IMAGE_NAME="seesr-sd-turbo"
TAG="latest"
FULL_IMAGE="$IMAGE_NAME:$TAG"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

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

print_header() {
    echo -e "${PURPLE}🐳 $1${NC}"
}

# Funzione per verificare se Docker è disponibile
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker non trovato. Installa Docker Desktop."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon non in esecuzione. Avvia Docker Desktop."
        exit 1
    fi
    
    print_success "Docker disponibile"
}

# Funzione per verificare se Cog è disponibile
check_cog() {
    if command -v cog &> /dev/null; then
        print_success "Cog disponibile"
        return 0
    else
        print_warning "Cog non trovato - solo build Docker manuale"
        return 1
    fi
}

# Funzione per build Docker manuale
build_docker() {
    print_header "Build Docker Container"
    
    print_info "Building image: $FULL_IMAGE"
    echo "Questo può richiedere 15-30 minuti per il download dei modelli..."
    echo ""
    
    # Build con progress
    docker build \
        --progress=plain \
        --tag "$FULL_IMAGE" \
        -f dockerfile \
        .. 2>&1 | tee docker_build.log
    
    if [ $? -eq 0 ]; then
        print_success "Build completato: $FULL_IMAGE"
        
        # Mostra dimensione finale
        local size=$(docker images "$FULL_IMAGE" --format "table {{.Size}}" | tail -n 1)
        print_info "Dimensione finale: $size"
        
        return 0
    else
        print_error "Build fallito. Vedi docker_build.log per dettagli."
        return 1
    fi
}

# Funzione per build con Cog
build_cog() {
    print_header "Build con Cog"
    
    print_info "Building con Cog (ottimizzato per Replicate)..."
    
    if cog build 2>&1 | tee cog_build.log; then
        print_success "Build Cog completato"
        return 0
    else
        print_error "Build Cog fallito. Vedi cog_build.log per dettagli."
        return 1
    fi
}

# Funzione per test del container
test_container() {
    print_header "Test Docker Container"
    
    # Verifica che l'immagine esista
    if ! docker images | grep -q "$IMAGE_NAME"; then
        print_error "Immagine $FULL_IMAGE non trovata. Esegui prima il build."
        return 1
    fi
    
    print_info "Test 1: Verifica importazioni Python..."
    if docker run --rm "$FULL_IMAGE" python3 -c "
import torch, diffusers, transformers, accelerate
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Diffusers: {diffusers.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ CUDA: {torch.cuda.is_available()}')
"; then
        print_success "Test importazioni passed"
    else
        print_error "Test importazioni failed"
        return 1
    fi
    
    print_info "Test 2: Verifica modelli scaricati..."
    if docker run --rm "$FULL_IMAGE" python3 -c "
import os
from pathlib import Path

models = [
    'deployment/preset/models/seesr',
    'deployment/preset/models/sd-turbo', 
    'deployment/preset/models/ram'
]

for model_dir in models:
    path = Path(model_dir)
    if path.exists() and any(path.iterdir()):
        print(f'✅ {model_dir}')
    else:
        print(f'❌ {model_dir}')
        exit(1)
        
print('✅ Tutti i modelli presenti')
"; then
        print_success "Test modelli passed"
    else
        print_error "Test modelli failed"
        return 1
    fi
    
    print_info "Test 3: Health check..."
    if docker run --rm "$FULL_IMAGE" python3 -c "
import sys
try:
    import torch
    import diffusers
    cuda_ok = torch.cuda.is_available() if torch.cuda.is_available() else True
    print(f'✅ Health check: CUDA={cuda_ok}, PyTorch={torch.__version__}')
    sys.exit(0 if cuda_ok else 1)
except Exception as e:
    print(f'❌ Health check failed: {e}')
    sys.exit(1)
"; then
        print_success "Health check passed"
    else
        print_warning "Health check failed (normale se non hai GPU)"
    fi
    
    print_success "Tutti i test completati!"
    return 0
}

# Funzione per test inference
test_inference() {
    print_header "Test Inference"
    
    # Crea immagine di test se non esiste
    if [ ! -f "test_input.jpg" ]; then
        print_info "Creazione immagine di test..."
        docker run --rm "$FULL_IMAGE" python3 -c "
from PIL import Image
import numpy as np

# Crea un'immagine di test
img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
img.save('/tmp/test_input.jpg')
"
        docker cp $(docker create "$FULL_IMAGE"):/tmp/test_input.jpg test_input.jpg
        print_success "Immagine di test creata: test_input.jpg"
    fi
    
    # Test con Cog se disponibile
    if check_cog; then
        print_info "Test inference con Cog..."
        if cog predict -i image=@test_input.jpg -o test_output.png; then
            print_success "Test inference Cog completato"
            print_info "Output salvato in: test_output.png"
        else
            print_error "Test inference Cog fallito"
        fi
    else
        print_info "Test inference con Docker..."
        # TODO: Implementare test inference diretto con Docker
        print_warning "Test inference Docker non ancora implementato"
    fi
}

# Funzione per run interattivo
run_interactive() {
    print_header "Run Interattivo"
    
    print_info "Avvio container interattivo..."
    print_info "Usa 'exit' per uscire"
    echo ""
    
    docker run -it --rm \
        --gpus all \
        -v $(pwd):/workspace \
        -w /workspace \
        "$FULL_IMAGE" \
        /bin/bash
}

# Funzione per deploy
deploy_container() {
    print_header "Deploy Container"
    
    echo "Opzioni di deploy:"
    echo "1. Docker Hub"
    echo "2. Replicate (con Cog)"
    echo "3. Locale (salvare tar)"
    echo ""
    
    read -p "Scegli opzione (1-3): " choice
    
    case $choice in
        1)
            read -p "Username Docker Hub: " username
            docker tag "$FULL_IMAGE" "$username/$IMAGE_NAME:$TAG"
            docker push "$username/$IMAGE_NAME:$TAG"
            print_success "Push a Docker Hub completato"
            ;;
        2)
            if check_cog; then
                read -p "Replicate model name (username/model): " model_name
                cog push "r8.im/$model_name"
                print_success "Push a Replicate completato"
            else
                print_error "Cog non disponibile per deploy Replicate"
            fi
            ;;
        3)
            docker save "$FULL_IMAGE" | gzip > "${IMAGE_NAME}_${TAG}.tar.gz"
            print_success "Immagine salvata in: ${IMAGE_NAME}_${TAG}.tar.gz"
            ;;
        *)
            print_error "Opzione non valida"
            ;;
    esac
}

# Funzione per clean
clean_containers() {
    print_header "Pulizia Container e Cache"
    
    # Rimuovi container stopped
    print_info "Rimozione container stopped..."
    docker container prune -f
    
    # Rimuovi immagini dangling
    print_info "Rimozione immagini dangling..."
    docker image prune -f
    
    # Rimuovi build cache
    read -p "Vuoi rimuovere anche la build cache? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker builder prune -f
        print_success "Build cache rimossa"
    fi
    
    # Rimuovi l'immagine SEESR
    read -p "Vuoi rimuovere l'immagine $FULL_IMAGE? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rmi "$FULL_IMAGE" 2>/dev/null || true
        print_success "Immagine $FULL_IMAGE rimossa"
    fi
    
    print_success "Pulizia completata"
}

# Funzione per mostrare help
show_help() {
    echo "Docker Build Script per SEESR"
    echo ""
    echo "USAGE:"
    echo "  $0 [COMMAND]"
    echo ""
    echo "COMMANDS:"
    echo "  build       - Build del Docker container"
    echo "  test        - Test del container"
    echo "  inference   - Test inference completo"
    echo "  run         - Avvia container interattivo"
    echo "  deploy      - Deploy del container"
    echo "  clean       - Pulizia container e cache"
    echo "  logs        - Mostra log build"
    echo "  status      - Stato container e immagini"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 build       # Build con auto-detect Cog/Docker"
    echo "  $0 test        # Test completo del container"
    echo "  $0 inference   # Test inference end-to-end"
    echo "  $0 run         # Shell interattivo nel container"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - Docker Desktop installato e running"
    echo "  - Cog (opzionale, per build ottimizzato)"
    echo "  - NVIDIA GPU (opzionale, per CUDA)"
}

# Funzione per mostrare status
show_status() {
    print_header "Status Docker"
    
    echo "🐳 Docker Images:"
    docker images | grep -E "(REPOSITORY|$IMAGE_NAME)" || echo "Nessuna immagine SEESR trovata"
    echo ""
    
    echo "📦 Container SEESR:"
    docker ps -a | grep -E "(NAMES|$IMAGE_NAME)" || echo "Nessun container SEESR in esecuzione"
    echo ""
    
    echo "💾 Spazio utilizzato:"
    docker system df
    echo ""
    
    if [ -f "docker_build.log" ]; then
        echo "📋 Ultimo build: $(date -r docker_build.log '+%Y-%m-%d %H:%M:%S')"
    fi
    
    if [ -f "cog_build.log" ]; then
        echo "📋 Ultimo build Cog: $(date -r cog_build.log '+%Y-%m-%d %H:%M:%S')"
    fi
}

# Main script
main() {
    print_header "SEESR Docker Build Tool"
    
    # Verifica prerequisiti
    check_docker
    
    case "${1:-help}" in
        "build")
            if check_cog; then
                echo "Scegli metodo di build:"
                echo "1. Cog (consigliato per Replicate)"
                echo "2. Docker (manuale)"
                echo ""
                read -p "Opzione (1-2): " choice
                case $choice in
                    1) build_cog ;;
                    2) build_docker ;;
                    *) build_cog ;; # Default
                esac
            else
                build_docker
            fi
            ;;
        "test")
            test_container
            ;;
        "inference")
            test_inference
            ;;
        "run")
            run_interactive
            ;;
        "deploy")
            deploy_container
            ;;
        "clean")
            clean_containers
            ;;
        "logs")
            if [ -f "docker_build.log" ]; then
                tail -50 docker_build.log
            elif [ -f "cog_build.log" ]; then
                tail -50 cog_build.log
            else
                print_error "Nessun log di build trovato"
            fi
            ;;
        "status")
            show_status
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
}

# Esegui main con tutti gli argomenti
main "$@"
