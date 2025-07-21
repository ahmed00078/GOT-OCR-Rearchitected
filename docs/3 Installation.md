## 3. Installation & Configuration

### Prérequis Système
- **Python 3.8+**
- **CUDA** (optionnel, pour accélération GPU)
- **8GB RAM minimum** (16GB recommandé)
- **Espace disque** : 10GB pour les modèles

### Installation Rapide

```bash
# Cloner le projet
git clone https://github.com/ahmed00078/GOT-OCR-Rearchitected.git
cd GOT-OCR-Rearchitected

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration des Modèles

#### Modèle OCR (Automatique)
Le modèle GOT-OCR se télécharge automatiquement au premier lancement.

#### Modèle IA (Personnalisable)
Modifier dans `config.py` :
```python
REASONING_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"  # ← Changez ici
```

**Modèles supportés** :
- `Qwen/Qwen2.5-7B-Instruct` (défaut)
- Tous les models publique sur huggingFace

### Variables d'Environnement (Optionnel)
```bash
export CUDA_VISIBLE_DEVICES=0  # GPU à utiliser
export HF_HOME=/path/to/models  # Cache des modèles
```

### Démarrage du Serveur

```bash
# Démarrer l'API
python main.py

# Ou avec Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Vérification
- **Frontend** : http://localhost:8000
- **API Docs** : http://localhost:8000/docs
- **Santé** : http://localhost:8000/health