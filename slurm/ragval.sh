#!/bin/bash
#SBATCH --job-name=llm_ragval
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/%j_ragval.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate tesis

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

python ragval.py generate --config configs/ragval.yaml

pwd; hostname; date
