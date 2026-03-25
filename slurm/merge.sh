#!/bin/bash
#SBATCH --job-name=merge_csv
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/%j_merge_csv.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate tesis

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

python merge.py merge --config configs/merge.yaml

pwd; hostname; date
