#!/bin/bash
# CPU job: build the irrelevance negatives and the training mixes with
# train/make_negatives.py, from train_59000.jsonl and val.jsonl.
# Formal runs use the SCF checkout on a clean, current main and write to
# $ROOT/data/negatives; SMOKE=1 writes to $ROOT/data/smoke/negatives_<job> instead.
#   sbatch slurm/make_negatives.sh
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/make_negatives_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/make_negatives_%j.err
set -euo pipefail

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
DATA=/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
PY=$DATA/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE

if [[ "${SMOKE:-0}" == 1 ]]; then
    OUT=$ROOT/data/smoke/negatives_${SLURM_JOB_ID}
else
    BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
    if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
        echo "formal runs need a clean main checkout (found $BRANCH)" >&2
        exit 1
    fi
    OUT=$ROOT/data/negatives
fi

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD) out $OUT"
"$PY" "$CODE/train/make_negatives.py" --revision "$MODEL_REVISION" \
    --train-file "$ROOT/data/processed/train_59000.jsonl" \
    --val-file "$ROOT/data/processed/val.jsonl" --out-dir "$OUT"
ls -l "$OUT"
echo "end $(date -Iseconds)"
