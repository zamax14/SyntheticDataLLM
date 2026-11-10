#!/bin/bash
#SBATCH --job-name=gen_agent_testset
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=slurm/logs/%j_gen_agent_testset.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Usage: sbatch slurm/create_agent_testset.sh [--config overrides.yaml]

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate datagen

echo "========================================"
echo "  Agent evaluation test set"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py create_agent_testset --config configs/agent_testset.yaml "$@"

pwd; hostname; date
