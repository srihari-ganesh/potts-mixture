#!/bin/bash

#SBATCH -c 20
#SBATCH -N 1

#SBATCH -t 0-00:10
#SBATCH -p short
#SBATCH	--mem=8G

#SBATCH -o %j.out
#SBATCH -e %j.err

module load miniconda3/4.10.3
module load gcc/9.2.0
module load hmmer/3.3.2
module load cuda/11.2

source /n/app/miniconda3/4.10.3/bin/activate srihari_base

python ../PLMC_EM.py sh3_pdz_combined.fa -m 5 -em 1000 -n 20 -g --fast -fc true_cluster_file_names.txt -fp ../../../plmc/bin/plmc -do itworks
