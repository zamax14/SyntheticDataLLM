#!/bin/bash
#SBATCH --job-name=gen_refilter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_refilter.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate tesis

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Re-apply the quality gate"
echo "========================================"

python synthetic.py refilter --config configs/refilter.yaml

pwd; hostname; date
