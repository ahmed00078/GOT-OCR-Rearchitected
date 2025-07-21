# 7. Troubleshooting

## Problèmes d'Installation

### Erreur : "No module named 'torch'"
```bash
# Solution
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Erreur : "CUDA out of memory"
```bash
# Solution 1 : Réduire la taille du batch
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Solution 2 : Utiliser CPU
export CUDA_VISIBLE_DEVICES=""
```

### Erreur : "Model not found"
```bash
# Solution : Vérifier le cache HuggingFace
export HF_HOME=/path/to/models
huggingface-cli download stepfun-ai/GOT-OCR-2.0-hf
```

## Problèmes de Performance

### Traitement très lent
**Symptômes** : Temps de traitement > 30 secondes
**Solutions** :
```python
# config.py
ENABLE_QUANTIZATION: bool = True
LOW_CPU_MEM_USAGE: bool = True
USE_GPU_IF_AVAILABLE: bool = True
```

### Mémoire insuffisante
**Symptômes** : Erreur "RuntimeError: CUDA out of memory"
**Solutions** :
```bash
# Réduire la résolution des images
# Traiter une image à la fois
# Utiliser la quantization
```

## Problèmes d'Extraction

### Résultats vides ou incorrects
**Symptômes** : JSON vide ou données manquantes
**Solutions** :
- **Vérifier l'image** : Résolution minimum 300 DPI
- **Instructions précises** : Listez exactement les champs souhaités
- **Format** : Demandez explicitement le format JSON

### Timeout API
**Symptômes** : Erreur 504 Gateway Timeout
**Solutions** :
```python
# Augmenter le timeout
UVICORN_TIMEOUT: int = 1200  # 20 minutes

# Ou dans le client
requests.post(url, timeout=120)
```

## Problèmes de Connectivité

### Serveur ne démarre pas
**Symptômes** : Port 8000 déjà utilisé
**Solutions** :
```bash
# Changer le port
uvicorn main:app --port 8001

# Ou tuer le processus
lsof -ti:8000 | xargs kill -9
```

### Frontend non accessible
**Symptômes** : 404 Not Found
**Solutions** :
```bash
# Vérifier le serveur
curl http://localhost:8000/health

# Redémarrer le serveur
python main.py
```

## Logs et Débogage

### Activation des logs détaillés
```python
# utils/logger.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Vérification du statut
```bash
# Santé du serveur
curl http://localhost:8000/health

# Modèles chargés
curl http://localhost:8000/demo
```

### Fichiers de log
```bash
# Logs par défaut
tail -f /var/log/got-ocr.log

# Logs personnalisés
export LOG_LEVEL=DEBUG
python main.py > debug.log 2>&1
```

## Solutions Rapides

| Problème | Solution Rapide |
|----------|----------------|
| Mémoire CUDA | `export CUDA_VISIBLE_DEVICES=""` |
| Modèle lent | `ENABLE_QUANTIZATION=True` |
| Timeout | `UVICORN_TIMEOUT=1200` |
| Port occupé | `uvicorn main:app --port 8001` |
| Extraction vide | Vérifier résolution image |
| Erreur JSON | Demander format JSON explicitement |

## Support

- **Issues** : https://github.com/ahmed00078/GOT-OCR-Rearchitected/issues
- **Logs** : Joignez toujours les logs d'erreur
- **Config** : Mentionnez votre configuration (OS, GPU, Python)