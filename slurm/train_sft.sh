#!/bin/bash
# GPU job: formal LoRA SFT runs plus their validation-set scoring. TRAIN_SIZES and
# RANKS are colon-separated lists; every size x rank pair runs one after another on
# the same GPU. RSLORA=1 switches to rank-stabilized scaling alpha/sqrt(r).
# Formal runs use the SCF checkout on a clean, current main.
# Usage (flags repeated on the command line for the safe_submit quota check):
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive \
#       --gres=gpu:H200:1 --export=ALL,EXP=expA,TRAIN_SIZES=10000:5000,RANKS=16,ALPHA=32 slurm/train_sft.sh
#   experiment B: --export=ALL,EXP=expB,TRAIN_SIZES=59000,RANKS=4:8:32,ALPHA=8,RSLORA=1
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
: "${EXP:?set EXP, e.g. expA}" "${TRAIN_SIZES:?set TRAIN_SIZES, e.g. 10000:5000}" "${RANKS:?set RANKS, e.g. 4:8:32}" "${ALPHA:?set ALPHA}"

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
# SCF's /etc/profile.d/SITE-huggingface.sh points TRANSFORMERS_CACHE at an empty
# /scratch cache, which overrides HF_HUB_CACHE in transformers 4.x.
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE

BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
    echo "formal runs need a clean main checkout (found $BRANCH)" >&2
    exit 1
fi

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

SCALING=() TAG=""
if [[ "${RSLORA:-0}" == 1 ]]; then SCALING=(--rslora) TAG=rs; fi

for TRAIN_SIZE in ${TRAIN_SIZES//:/ }; do
    for RANK in ${RANKS//:/ }; do
        RUN=$ROOT/runs/$EXP/sft_n${TRAIN_SIZE}_r${RANK}${TAG}_${SLURM_JOB_ID}
        echo "run $RUN $(date -Iseconds)"
        "$PY" "$CODE/train/sft_lora.py" --revision "$MODEL_REVISION" \
            --train-file "$ROOT/data/processed/train_${TRAIN_SIZE}.jsonl" \
            --val-file "$ROOT/data/processed/val.jsonl" \
            --output-dir "$RUN" --rank "$RANK" --alpha "$ALPHA" "${SCALING[@]}"
        "$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" \
            --adapter "$RUN/adapter" --val-file "$ROOT/data/processed/val.jsonl" \
            --output-dir "$RUN/val"
    done
done
echo "end $(date -Iseconds)"
