#!/bin/bash
#SBATCH --job-name=gen_setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_setup.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# One-off: creates the 'datagen' env used by the generation job. Kept apart from
# 'tesis' so pip's resolver cannot touch its torch / sentence-transformers.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh

echo "========================================"
echo "  Create 'datagen' env (distilabel)"
echo "========================================"

conda env list | grep -q '^datagen ' || conda create -y -n datagen python=3.12
conda activate datagen
pip install -q "distilabel[openai]" jsonargparse pandas
python -c "import distilabel, jsonargparse, pandas; print('datagen ok, distilabel', distilabel.__version__)"

echo "========================================"
echo "  Check 'tesis' env (mining + export)"
echo "========================================"

conda activate tesis
python - <<'PY'
import importlib
for mod in ('pandas', 'jsonargparse', 'sentence_transformers', 'datasets'):
    try:
        importlib.import_module(mod)
        print(f'  ok      {mod}')
    except ImportError:
        print(f'  MISSING {mod}  <- pip install it in the tesis env')
PY

pwd; hostname; date
