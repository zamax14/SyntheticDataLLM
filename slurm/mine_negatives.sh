#!/bin/bash
#SBATCH --job-name=gen_mine
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=slurm/logs/%j_gen_mine.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx_large
set -e

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate tesis

echo "========================================"
echo "  Mine corpus hard negatives"
echo "========================================"

python synthetic.py mine_negatives --config configs/mine_negatives.yaml

pwd; hostname; date
