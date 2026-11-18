#!/bin/bash
# GPU job: score the zero-shot base and already trained adapters on the
# irrelevance negatives, val_neg200.jsonl from slurm/make_negatives.sh. ADAPTERS
# is a colon-separated list of run directories under $ROOT/runs; each result goes
# to <run>/val_neg_<job>, the base's to runs/baseline/zero_shot_base_<job>/val_neg.
# Formal runs use the SCF checkout on a clean, current main.
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive --gres=gpu:H200:1 \
#       --export=ALL,ADAPTERS=expA/sft_n59000_r16_4110239 slurm/eval_neg.sh
# On another node (e.g. thidwick), add -w <node> and point the logs at horton:
#   --output=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_neg_%j.out
#   --error=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_neg_%j.err
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_neg_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/eval_neg_%j.err
set -euo pipefail
: "${ADAPTERS:?set ADAPTERS, e.g. expA/sft_n59000_r16_4110239}"

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
# Data, environment, and HF cache live on horton's /data; other nodes reach
# them through the /net/horton autofs path.
DATA=/data/wenruixu
[[ $(hostname -s) == horton ]] || DATA=/net/horton/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
PY=$DATA/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
VAL_NEG=$ROOT/data/negatives/val_neg200.jsonl
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
# SCF's /etc/profile.d/SITE-huggingface.sh points TRANSFORMERS_CACHE at an empty
# /scratch cache, which overrides HF_HUB_CACHE in transformers 4.x.
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE

BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
    echo "formal runs need a clean main checkout (found $BRANCH)" >&2
    exit 1
fi

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
"$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" --val-file "$VAL_NEG" \
    --output-dir "$ROOT/runs/baseline/zero_shot_base_${SLURM_JOB_ID}/val_neg"
for RUN in ${ADAPTERS//:/ }; do
    "$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" --val-file "$VAL_NEG" \
        --adapter "$ROOT/runs/$RUN/adapter" --output-dir "$ROOT/runs/$RUN/val_neg_${SLURM_JOB_ID}"
done
echo "end $(date -Iseconds)"
