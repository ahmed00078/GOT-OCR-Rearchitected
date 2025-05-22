## Construire l'image Singularity

sudo singularity build HPC/got-ocr-optimized.sif HPC/got-ocr-optimized.def

## 

chmod +x HPC/run_optimized_cluster.slurm

# Transférer l'image et les scripts
scp HPC/got-ocr-optimized.sif asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/
scp HPC/run_optimized_cluster.slurm asidimoh@10.11.8.2:/data/asidimoh/GOT2/Try6/

# Se connecter au cluster
ssh asidimoh@10.11.8.2
cd /data/asidimoh/GOT2/Try6

# Soumettre le job
sbatch run_optimized_cluster.slurm

# Vérifier l'état
squeue -u $USER

# Suivre les logs en temps réel
tail -f got-ocr-v2-*.out

# Test du service sans SLURM (pour debug)
singularity run --nv got-ocr-optimized.sif

# Une fois sur le nœud, testez manuellement
singularity exec --nv got-ocr-optimized.sif /bin/bash

# Dans le conteneur, vérifiez la structure
ls -la /app

# Testez le runscript manuellement
cd /app
python main.py