#!/bin/bash
#SBATCH --job-name=gen_multitool
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=slurm/logs/%j_gen_multitool.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Usage: sbatch slurm/create_multi_tool_queries.sh [--config overrides.yaml ...]
#   smoke: --config configs/smoke_multi_tool_queries.yaml

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate datagen

echo "========================================"
echo "  Multi-tool requests (agent evaluation)"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /tmp/ollama_tags_$SLURM_JOB_ID.json \
  || { echo "ollama no responde en localhost:11434"; exit 1; }
# Tags on the node do not always match the weights: log the digests used.
python -c "import json,sys; [print('digest', m['name'], m['digest'][:12]) for m in json.load(open(sys.argv[1]))['models'] if m['name'] in ('gemma4:31b', 'deepseek-r1:70b')]" \
  /tmp/ollama_tags_$SLURM_JOB_ID.json
rm -f /tmp/ollama_tags_$SLURM_JOB_ID.json

python synthetic.py create_multi_tool_queries --config configs/multi_tool_queries.yaml "$@"

pwd; hostname; date
