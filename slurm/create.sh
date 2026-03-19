#!/bin/bash
#SBATCH --job-name=llm_data_small
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/%j_syntetic_small.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate tesis

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

python synthetic.py create --config configs/create.yaml

pwd; hostname; date
