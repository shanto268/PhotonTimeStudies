#!/bin/bash
#SBATCH --job-name=first_try
#SBATCH --output=quanah_logs/%x.o%j
#SBATCH --error=quanah_logs/%x.e%j
#SBATCH --partition quanah
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=5
#SBATCH --mail-user=<sadman-ahmed.shanto@ttu.edu>
#SBATCH --mail-type=ALL

# actual code to execute
conda activate apdl_ml
python ML_PTP.py
