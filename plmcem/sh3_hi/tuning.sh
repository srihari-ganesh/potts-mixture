#!/bin/bash

#SBATCH -c 6
#SBATCH -N 1

#SBATCH -t 0-08:00
#SBATCH -p short
#SBATCH	--mem=16G

#SBATCH -o tuning/job_info/%j.out
#SBATCH -e tuning/job_info/%j.err

module load miniconda3/4.10.3
module load gcc/9.2.0
module load cuda/11.2

source activate base
conda activate srihari_base

python ../plmcem.py $7 -m $4 -em 30 -n 6 -gt $2 -sgdt $5 -ei $6 -do tuning/outputs/tuning_run_$1 -t $3
