#!/bin/bash
# GPU job: formal LoRA SFT runs plus their validation-set scoring. TRAIN_SIZES is
# a colon-separated list; the sizes run one after another on the same GPU.
# Formal runs use the SCF checkout on a clean, current main.
# Usage (flags repeated on the command line for the safe_submit quota check):
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive \
#       --gres=gpu:H200:1 --export=ALL,EXP=expA,TRAIN_SIZES=10000:5000,RANK=16,ALPHA=32 slurm/train_sft.sh
# On another node (e.g. thidwick), add -w <node> and point the logs at horton:
#   --output=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/train_sft_%j.out
#   --error=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/train_sft_%j.err
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/train_sft_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/train_sft_%j.err
set -euo pipefail
: "${EXP:?set EXP, e.g. expA}" "${TRAIN_SIZES:?set TRAIN_SIZES, e.g. 10000:5000}" "${RANK:?set RANK}" "${ALPHA:?set ALPHA}"

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
# Data, environment, and HF cache live on horton's /data; other nodes reach
# them through the /net/horton autofs path.
DATA=/data/wenruixu
[[ $(hostname -s) == horton ]] || DATA=/net/horton/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
PY=$DATA/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
    echo "formal runs need a clean main checkout (found $BRANCH)" >&2
    exit 1
fi

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

for TRAIN_SIZE in ${TRAIN_SIZES//:/ }; do
    RUN=$ROOT/runs/$EXP/sft_n${TRAIN_SIZE}_r${RANK}_${SLURM_JOB_ID}
    echo "run $RUN $(date -Iseconds)"
    "$PY" "$CODE/train/sft_lora.py" --revision "$MODEL_REVISION" \
        --train-file "$ROOT/data/processed/train_${TRAIN_SIZE}.jsonl" \
        --val-file "$ROOT/data/processed/val.jsonl" \
        --output-dir "$RUN" --rank "$RANK" --alpha "$ALPHA"
    "$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" \
        --adapter "$RUN/adapter" --val-file "$ROOT/data/processed/val.jsonl" \
        --output-dir "$RUN/val"
done
echo "end $(date -Iseconds)"
