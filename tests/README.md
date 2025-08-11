# 🧪 Tests Directory

Questa directory contiene tutti i test per il progetto SEESR.

## 📁 File di Test

### Test Generali
- **`test_seesr.py`** - Test completo del sistema SEESR
- **`test_complete.py`** - Test di integrazione completa  
- **`test_environment.py`** - Test dell'ambiente di sviluppo

### Test Deployment
- **`test_docker_env.py`** - Test dell'ambiente Docker
- **`test_replicate_configs.py`** - Test configurazioni Replicate

## 🚀 Come Eseguire i Test

### Test Locale (Virtual Environment)
```bash
# Test singolo
./start_seesr.sh test

# Test specifico
python tests/test_seesr.py
python tests/test_environment.py
```

### Test Docker
```bash
# Test ambiente Docker
./docker/docker_build.sh test

# Test completo container
docker run --rm seesr-sd-turbo python3 tests/test_docker_env.py
```

### Test Replicate
```bash
# Test configurazioni Replicate
python tests/test_replicate_configs.py
```

## 📊 Copertura Test

I test coprono:
- ✅ Import dei moduli Python
- ✅ Disponibilità CUDA/GPU
- ✅ Download e presenza modelli
- ✅ Configurazione ambiente
- ✅ Funzionalità SEESR pipeline
- ✅ Integrazione con SD Turbo
- ✅ Memory management
- ✅ Cross-platform compatibility

## 🐛 Troubleshooting

### Test Falliti
```bash
# Reinstalla dipendenze
./start_seesr.sh install

# Pulisci e ricostruisci ambiente
./start_seesr.sh clean
./start_seesr.sh setup
```

### Test Docker Falliti
```bash
# Ricostruisci container
./docker/docker_build.sh build

# Debug container
./docker/docker_build.sh run
```
