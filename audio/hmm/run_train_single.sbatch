#!/bin/bash

#SBATCH --account=mscagpu
#SBATCH --job-name=robot_hmm
#SBATCH --output=%j_out.log
#SBATCH --error=%j_err.log
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=mscagpu
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=salerno@uchicago.edu
#SBATCH --mail-type=ALL

# Load modules:
module load Anaconda3/2019.03

# Activate conda environment:
source activate msca_gpu_env

# GPU-based executable:
python -u audio_hmm_single.py

