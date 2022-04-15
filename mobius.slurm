#!/bin/bash
#SBATCH --job-name pytorch-example
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --partition smp,gpu

echo Starting... >&1

unset OMP_NUM_THREADS  #let pytorch & MKLDNN use all available CPU cores

module load anaconda3

python mobius.py
