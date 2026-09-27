#!/bin/bash
#SBATCH --job-name=restyle
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_restyle.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Restyle the E3 distractor tools"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py restyle_distractors --config configs/restyle_distractors.yaml

pwd; hostname; date
