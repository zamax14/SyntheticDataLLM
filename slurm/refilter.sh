#!/bin/bash
#SBATCH --job-name=gen_refilter
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=slurm/logs/%j_gen_refilter.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate tesis

echo "========================================"
echo "  Re-apply the quality gate"
echo "========================================"

python synthetic.py refilter --config configs/refilter.yaml

pwd; hostname; date
