# 🚀 Commandes HPC - GOT-OCR 2.0 + SmolLM2

Guide complet pour déployer et utiliser le système OCR + IA sur le cluster HPC.

## 📋 Versions disponibles

| Version          | Description          | Fichiers                |
| ---------------- | -------------------- | ----------------------- |
| **Optimized**    | Version de base OCR  | `got-ocr-optimized.sif` |
| **Multi-page**   | OCR + Support PDF    | `got-ocr-multipage.sif` |
| **🧠 Reasoning** | **OCR + SmolLM2 IA** | `got-ocr-reasoning.sif` |

---

## 🧠 Déploiement GOT-OCR + SmolLM2 (RECOMMANDÉ)

### Étape 1: Construction de l'image

```bash
# Construire la nouvelle image avec SmolLM2
sudo singularity build HPC/got-ocr-reasoning.sif HPC/got-ocr-reasoning.def

# Vérifier la taille de l'image
ls -lh HPC/got-ocr-reasoning.sif
```

### Étape 2: Préparation du déploiement

```bash
# Rendre le script exécutable
chmod +x HPC/run_reasoning_cluster.slurm

# Vérifier les fichiers nécessaires
ls -la HPC/got-ocr-reasoning.*
```

### Étape 3: Transfert vers le cluster

```bash
# Transférer l'image et les scripts
scp HPC/got-ocr-reasoning.sif asidimoh@10.11.8.2:/data/asidimoh/GOT2/SmolLM2/
scp HPC/run_reasoning_cluster.slurm asidimoh@10.11.8.2:/data/asidimoh/GOT2/SmolLM2/

# Optionnel: Transférer des exemples
scp examples/practical_usage.py asidimoh@10.11.8.2:/data/asidimoh/GOT2/SmolLM2/
scp test_smollm2_integration.py asidimoh@10.11.8.2:/data/asidimoh/GOT2/SmolLM2/
```

### Étape 4: Connexion et lancement

```bash
# Se connecter au cluster
ssh asidimoh@10.11.8.2

# Aller au répertoire de travail
cd /data/asidimoh/GOT2/SmolLM2

# Soumettre le job SmolLM2
sbatch run_reasoning_cluster.slurm
```

---

## 🔍 Monitoring et gestion

### Vérifier l'état des jobs

```bash
# Vérifier l'état de votre job
squeue -u $USER

# Vérifier les jobs SmolLM2 spécifiquement
squeue -u $USER --name=got-ocr-reasoning

# Informations détaillées
scontrol show job <JOB_ID>
```

### Suivre les logs en temps réel

```bash
# Logs de sortie (avec SmolLM2)
tail -f got-ocr-reasoning-*.out

# Logs d'erreur
tail -f got-ocr-reasoning-*.err

# Monitoring des ressources
tail -f resource_monitor_*.log
```

### Accès au service

```bash
# Une fois le job lancé, récupérer le nom du nœud
cat got-ocr-reasoning-*.out | grep "Node:"

# Créer un tunnel SSH (remplacer <NODE> par le nom réel)
ssh -L 8000:<NODE>:8000 asidimoh@10.11.8.2

# Accéder aux services:
# Interface principale: http://localhost:8000/frontend/enhanced_index.html
# API Documentation: http://localhost:8000/docs
# Smart Extract: http://localhost:8000/smart-extract
# Health Check: http://localhost:8000/health
```

---

## 🧪 Tests et validation

### Test rapide de fonctionnement

```bash
# Test de base (depuis le cluster)
curl http://$(hostname):8000/health

# Test complet des capacités SmolLM2
curl http://$(hostname):8000/extraction-types
```

### Test avec fichier

```bash
# Tester l'extraction carbone (exemple)
curl -X POST "http://$(hostname):8000/smart-extract" \
  -F "extraction_type=carbon_footprint" \
  -F "task=Multi-page OCR" \
  -F "images=@document_carbone.pdf"

# Tester l'extraction de spécifications techniques
curl -X POST "http://$(hostname):8000/smart-extract" \
  -F "extraction_type=technical_specs" \
  -F "task=Multi-page OCR" \
  -F "images=@fiche_technique.pdf"
```

### Test avec script Python

```bash
# Si vous avez transféré les scripts de test
python test_smollm2_integration.py --api-url http://$(hostname):8000

# Test avec exemples pratiques
python examples/practical_usage.py
```

---

## 🐛 Debug et résolution de problèmes

### Lancement en mode debug

```bash
# Test manuel du conteneur
singularity run --nv got-ocr-reasoning.sif

# Shell interactif dans le conteneur
singularity shell --nv got-ocr-reasoning.sif

# Dans le conteneur, vérifier la structure
ls -la /app
python -c "from services.reasoning_service import SmolLM2ReasoningService; print('SmolLM2 OK')"
```

### Vérification des ressources

```bash
# Vérifier l'utilisation GPU
nvidia-smi

# Vérifier la mémoire
free -h

# Vérifier l'espace disque
df -h

# Logs détaillés du système
dmesg | tail -20
```

### Debug des modèles

```bash
# Vérifier le cache des modèles
ls -la ~/.cache/huggingface/transformers/

# Forcer le téléchargement des modèles
singularity exec --nv got-ocr-reasoning.sif python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Téléchargement SmolLM2...')
AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct')
AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B-Instruct')
print('Modèles téléchargés!')
"
```

---

## ⚡ Optimisations de performance

### Configuration GPU intensive

```bash
# Pour les gros documents PDF (modifier dans le script SLURM)
export REASONING_BATCH_SIZE=5
export PDF_MAX_PAGES=100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
```

### Configuration économe en mémoire

```bash
# Pour les ressources limitées
export REASONING_BATCH_SIZE=1
export ENABLE_QUANTIZATION=true
export PDF_MAX_PAGES=20
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Monitoring des performances

```bash
# Surveiller l'utilisation en temps réel
watch -n 5 'nvidia-smi; echo "---"; free -h'

# Logs de performance détaillés
tail -f resource_monitor_*.log | awk -F',' '{print $1, "CPU:", $2"%", "RAM:", $3"%", "GPU:", $4"MB"}'
```

---

## 🔄 Versions précédentes (pour référence)

### Version Multi-page (sans SmolLM2)

```bash
# Construction
sudo singularity build HPC/got-ocr-multipage.sif HPC/got-ocr-multipage.def

# Déploiement
scp HPC/got-ocr-multipage.sif asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/
scp HPC/run_multipage_cluster.slurm asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/
ssh asidimoh@10.11.8.2
cd /data/asidimoh/GOT2/Try6
sbatch run_multipage_cluster.slurm
```

### Version Optimized (OCR basique)

```bash
# Construction
sudo singularity build HPC/got-ocr-optimized.sif HPC/got-ocr-optimized.def

# Déploiement
scp HPC/got-ocr-optimized.sif asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/
scp HPC/run_optimized_cluster.slurm asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/
ssh asidimoh@10.11.8.2
cd /data/asidimoh/GOT2/Try6
sbatch run_optimized_cluster.slurm
```

---

## 📊 Métriques de performance attendues

### Configuration cluster typique

| Métrique    | OCR Standard | OCR + SmolLM2 | Multi-page (5p) |
| ----------- | ------------ | ------------- | --------------- |
| Temps CPU   | 8-15s        | 45-90s        | 3-8min          |
| Temps GPU   | 3-8s         | 12-25s        | 1-3min          |
| Mémoire GPU | 2-4GB        | 6-8GB         | 8-12GB          |
| Confiance   | 90-95%       | 75-85%        | 70-80%          |

### Limites recommandées

- **PDF max**: 50 pages (100 pages avec GPU puissant)
- **Taille fichier**: 100MB maximum
- **Batch size**: 2-3 pour SmolLM2
- **Timeout**: 30 minutes pour gros documents

---

## 🆘 Support et ressources

### Ressources utiles

- **Documentation API**: `http://<node>:8000/docs`
- **Interface avancée**: `http://<node>:8000/frontend/enhanced_index.html`
- **Types d'extraction**: `http://<node>:8000/extraction-types`
- **Monitoring**: `http://<node>:8000/health`

### Contacts et aide

- **Issues GitHub**: [Créer une issue](https://github.com/ahmed00078/GOT-OCR-Rearchitected/issues)
- **Email support**: ahmedsidimohammed78@gmail.com
- **Documentation**: Voir `README.md` et `INSTALLATION_GUIDE.md`

### Logs importants

```bash
# Logs principaux à vérifier en cas de problème
tail -f got-ocr-reasoning-*.out    # Logs de démarrage
tail -f got-ocr-reasoning-*.err    # Logs d'erreur
tail -f resource_monitor_*.log     # Monitoring ressources
dmesg | grep -i gpu                # Logs GPU système
```

---

## 🎯 Cas d'usage spécialisés

### Extraction de données carbone (Perfect pour votre stage!)

```bash
# Interface web optimisée
http://localhost:8000/frontend/enhanced_index.html
# Choisir: Smart Extract > Carbon Footprint Data

# API directe
curl -X POST "http://$(hostname):8000/smart-extract" \
  -F "extraction_type=carbon_footprint" \
  -F "task=Multi-page OCR" \
  -F "images=@rapport_carbone.pdf"
```

### Extraction personnalisée

```bash
# Avec instructions spécifiques
curl -X POST "http://$(hostname):8000/smart-extract" \
  -F "extraction_type=custom" \
  -F "custom_instructions=Extraire toutes les émissions CO2, consommations énergétiques et certifications environnementales" \
  -F "task=Multi-page OCR" \
  -F "images=@document.pdf"
```

---

🎉 **Votre système GOT-OCR 2.0 + SmolLM2 est prêt pour le cluster HPC !**

**Note**: La version **Reasoning** est recommandée pour votre stage car elle inclut l'extraction intelligente de données carbone avec SmolLM2.
