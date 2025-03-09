#!/bin/sh
#SBATCH --job-name=gpu
#SBATCH --partition=gpu-h100-killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=32000 # CPU memory (MB)
#SBATCH --gpus=1 # GPU cores per process
#SBATCH --cpus-per-task=8 # CPU cores per process
python3 /home/iscb/wolfson/omriyakir/cpp_deep/models/esm2/esm2_peptide_pretrained/train_from_scratch.py