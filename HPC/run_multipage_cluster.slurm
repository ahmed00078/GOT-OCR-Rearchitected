#!/bin/bash
#SBATCH --job-name=got-ocr-v2-multipage
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=got-ocr-v2-multipage-%j.out
#SBATCH --error=got-ocr-v2-multipage-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "=================================================="
echo "GOT-OCR 2.0 Multi-Page - Job ID: $SLURM_JOB_ID"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# === NOUVELLES VARIABLES MULTI-PAGE ===
export PDF_CONVERSION_DPI=300
export PDF_MAX_PAGES=50
export MULTIPAGE_BATCH_SIZE=5
export UVICORN_TIMEOUT=600
export PDF_CONVERSION_TIMEOUT=300

echo "=== Environment Variables ==="
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "=== Multi-Page Settings ==="
echo "PDF_CONVERSION_DPI: $PDF_CONVERSION_DPI"
echo "PDF_MAX_PAGES: $PDF_MAX_PAGES"
echo "MULTIPAGE_BATCH_SIZE: $MULTIPAGE_BATCH_SIZE"

echo "=== Service Information ==="
echo "Service URL: http://$HOSTNAME:8000/frontend/index.html"
echo "API Docs: http://$HOSTNAME:8000/docs"
echo "Health Check: http://$HOSTNAME:8000/health"
echo "SSH Tunnel: ssh -L 8000:$HOSTNAME:8000 asidimoh@10.11.8.2"
echo ""
echo "=== Multi-Page Features ==="
echo "- PDF Processing: Enabled (max ${PDF_MAX_PAGES} pages)"
echo "- Image Processing: Enabled (JPEG, PNG, TIFF)"
echo "- Mixed Files: Supported"
echo "- Batch Processing: ${MULTIPAGE_BATCH_SIZE} pages per batch"

echo "=== Starting Container ==="
singularity run --nv got-ocr-multipage.sif