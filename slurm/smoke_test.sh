#!/bin/bash
#SBATCH --job-name=gen_smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_smoke.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Mandatory before launching the full run: proves the answer is not empty (the
# reasoning-mode trap), shows sample triplets, and times one anchor so the
# ~13k-anchor run can be extrapolated. No --gres=gpu: Ollama owns its own GPU.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Smoke test: 2 documents"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

rm -rf smoke && mkdir -p smoke/md
ls output/*.md | head -2 | xargs -I{} cp {} smoke/md/

SECONDS=0
python synthetic.py create_embeddings \
    --config configs/create_embeddings.yaml \
    --data_path ./smoke/md \
    --output_path ./smoke/out
echo "tiempo total: ${SECONDS}s"

python - <<'PY'
import pandas as pd
df = pd.read_csv('smoke/out/embeddings_qa.csv')
print(f'\naceptadas: {len(df)}')
print('columnas:', list(df.columns))
assert df['query'].notna().all() and (df['query'].str.strip() != '').all(), \
    'consultas vacias: el modo thinking sigue activo'
for _, r in df.head(5).iterrows():
    print(f"\n  Q: {r['query']}\n  A: {str(r['answer'])[:120]}...\n  N: {str(r['hard_negative'])[:120]}...")
PY

pwd; hostname; date
