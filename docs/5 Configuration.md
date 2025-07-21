## 5. Configuration Avancée

### Personnalisation des Modèles

#### Modèle OCR (config.py)
```python
# Modèle OCR principal
MODEL_NAME: str = "stepfun-ai/GOT-OCR-2.0-hf"
MAX_NEW_TOKENS: int = 4096

# Layout segmentation
LAYOUT_ENABLED: bool = True
LAYOUT_MODEL_NAME: str = "PP-DocLayout_plus-L"
```

#### Modèle IA (config.py)
```python
# Modèle de raisonnement
REASONING_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
REASONING_MAX_TOKENS: int = 4096
REASONING_TEMPERATURE: float = 0.1

# Optimisations
ENABLE_QUANTIZATION: bool = True
LOW_CPU_MEM_USAGE: bool = True
```

### Instructions d'Extraction Custom

#### Exemples d'Instructions
```python
# Extraction de contacts
"extract all contact information including names, emails, phones"

# Extraction de factures
"extract invoice number, date, total amount, and line items"

# Extraction de produits
"extract product name, price, description, and specifications"
```

#### Bonnes Pratiques
- **Soyez spécifique** : Précisez les champs exacts
- **Format de sortie** : Demandez JSON si nécessaire
- **Langue** : Utilisez français ou anglais
- **Contexte** : Donnez le type de document

### Performance et Optimisation

#### Paramètres Performance
```python
# Limites fichiers
MAX_FILE_SIZE: int = 50 * 1024 * 1024     # 50MB
MAX_PDF_SIZE: int = 100 * 1024 * 1024     # 100MB

# PDF Processing
PDF_CONVERSION_DPI: int = 300
PDF_MAX_PAGES: int = 50

# Batch processing
MULTIPAGE_BATCH_SIZE: int = 5
MULTIPAGE_MAX_FILES: int = 20
```

#### Optimisations GPU
```python
# GPU settings
USE_GPU_IF_AVAILABLE: bool = True
CUDA_VISIBLE_DEVICES = "0"  # Variable d'environnement
```

### Variables d'Environnement

```bash
# Modèles et cache
export HF_HOME=/path/to/models
export TRANSFORMERS_CACHE=/path/to/cache

# GPU
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Performance
export UVICORN_TIMEOUT=600
export OMP_NUM_THREADS=4
```