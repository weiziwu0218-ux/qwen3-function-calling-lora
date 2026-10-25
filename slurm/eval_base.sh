#!/bin/bash
# GPU job: formal validation-set scoring of the zero-shot base model.
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive \
#       --gres=gpu:H200:1 slurm/eval_base.sh
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_base_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_base_%j.err
set -euo pipefail

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
ROOT=/data/wenruixu/workspace/qwen3-function-calling-lora
PY=/data/wenruixu/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=/data/wenruixu/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
    echo "formal runs need a clean main checkout (found $BRANCH)" >&2
    exit 1
fi

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
"$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" \
    --val-file "$ROOT/data/processed/val.jsonl" \
    --output-dir "$ROOT/runs/baseline/zero_shot_base_${SLURM_JOB_ID}/val"
echo "end $(date -Iseconds)"
