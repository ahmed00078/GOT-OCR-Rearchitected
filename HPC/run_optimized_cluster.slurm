#!/bin/bash
#SBATCH --job-name=got-ocr-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=got-ocr-v2-%j.out
#SBATCH --error=got-ocr-v2-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "=================================================="
echo "GOT-OCR 2.0 Optimized - Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Date: $(date)"
echo "=================================================="

# Informations GPU
echo "=== GPU Information ==="
nvidia-smi

# Variables d'environnement pour performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Optimisations CUDA si disponible
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "=== Environment Variables ==="
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

echo "=== Service Information ==="
echo "Service URL: http://$HOSTNAME:8000/frontend/index.html"
echo "API Docs: http://$HOSTNAME:8000/docs"
echo "Health Check: http://$HOSTNAME:8000/health"
echo "SSH Tunnel: ssh -L 8000:$HOSTNAME:8000 asidimoh@10.11.8.2"

echo "=== Starting Container ==="
singularity run --nv got-ocr-optimized.sif