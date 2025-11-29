#!/bin/bash
#SBATCH --job-name=cellmap_main
#SBATCH --time=24:00:00
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err

echo "Job started on $(hostname)"
echo "Time: $(date)"

# ===== Activate micromamba (using correct path) =====
micromamba activate csc

echo "Using Python:"
which python
python --version

# ===== Go to correct project directory =====
cd /home/chwa386g/chwa386g/cellmap-segmentation-challenge/data

# ===== Run your script =====
python main2.py

echo "Job finished at $(date)"
