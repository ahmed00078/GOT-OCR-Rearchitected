#!/bin/bash
#SBATCH --job-name=got-ocr-reasoning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=got-ocr-reasoning-%j.out
#SBATCH --error=got-ocr-reasoning-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

echo "============================================================"
echo "GOT-OCR 2.0 + SmolLM2 Reasoning - Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "============================================================"

# === INFORMATIONS SYSTÈME ===
echo "=== System Information ==="
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Partition: $SLURM_JOB_PARTITION"

# === INFORMATIONS GPU ===
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
echo ""
nvidia-smi

# === VARIABLES D'ENVIRONNEMENT OPTIMISÉES ===

# Performance CPU
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true

# Optimisations CUDA
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Configuration OCR
export PDF_CONVERSION_DPI=300
export PDF_MAX_PAGES=100
export MULTIPAGE_BATCH_SIZE=8
export PDF_CONVERSION_TIMEOUT=600

# === CONFIGURATION SmolLM2 SPÉCIFIQUE CLUSTER ===
export REASONING_ENABLED=true
export ENABLE_QUANTIZATION=true
export REASONING_MAX_CONTEXT=2000
export UVICORN_TIMEOUT=1800

# Configuration mémoire aggressive pour cluster
export LOW_CPU_MEM_USAGE=true
export USE_GPU_IF_AVAILABLE=true

# Optimisations supplémentaires pour HPC
export TRANSFORMERS_CACHE=/tmp/transformers_cache_$SLURM_JOB_ID
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
export TMPDIR=/tmp/got_ocr_$SLURM_JOB_ID

# Créer les répertoires de cache temporaires
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p $TMPDIR

echo "=== Environment Variables Set ==="
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "UVICORN_TIMEOUT: $UVICORN_TIMEOUT"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"

# === VÉRIFICATIONS PRÉ-LANCEMENT ===
echo "=== Pre-launch Checks ==="

# Vérifier l'image Singularity
if [ ! -f "got-ocr-reasoning.sif" ]; then
    echo "ERREUR: Image got-ocr-reasoning.sif non trouvée!"
    exit 1
fi

echo "Image Singularity trouvée: $(ls -lh got-ocr-reasoning.sif)"

# === INFORMATIONS DE SERVICE ===
echo "=== Service Information ==="
echo "Service URL: http://$HOSTNAME:8000"
echo "Interface Web: http://$HOSTNAME:8000/frontend/index.html"
echo "API Standard: http://$HOSTNAME:8000/docs"
echo "Health Check: http://$HOSTNAME:8000/health"
echo ""
echo "SSH Tunnel pour accès externe:"
echo "ssh -L 8000:$HOSTNAME:8000 asidimoh@10.11.8.2"

# === FONCTION DE NETTOYAGE ===
cleanup() {
    echo "=== Nettoyage en cours ==="
    echo "Suppression des caches temporaires..."
    rm -rf $TRANSFORMERS_CACHE
    rm -rf $HF_HOME  
    rm -rf $TMPDIR
    echo "Nettoyage terminé"
}

# Programmer le nettoyage à la fin
trap cleanup EXIT

# === DÉMARRAGE DU SERVICE PRINCIPAL ===
echo "============================================================"
echo "DÉMARRAGE GOT-OCR 2.0 + SmolLM2 REASONING SERVICE"
echo "============================================================"

# Exécution avec gestion d'erreur
set -e
singularity run --nv \
    --bind /tmp:/tmp \
    --bind $PWD:/data \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env MKL_NUM_THREADS=$MKL_NUM_THREADS \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env REASONING_ENABLED=$REASONING_ENABLED \
    --env ENABLE_QUANTIZATION=$ENABLE_QUANTIZATION \
    --env UVICORN_TIMEOUT=$UVICORN_TIMEOUT \
    --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
    --env HF_HOME=$HF_HOME \
    got-ocr-reasoning.sif

# === POST-TRAITEMENT ===
echo "=== Service terminé ==="

echo "Job terminé à: $(date)"
echo "Durée totale: $SECONDS secondes"