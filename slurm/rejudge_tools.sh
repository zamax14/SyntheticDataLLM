#!/bin/bash
#SBATCH --job-name=rejudge
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=slurm/logs/%j_rejudge.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Usage: sbatch slurm/rejudge_tools.sh [config]

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate datagen

echo "========================================"
echo "  Re-run the round-trip judge"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py rejudge_tools --config "${1:-configs/rejudge_train_tools.yaml}"

pwd; hostname; date
