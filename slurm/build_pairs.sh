#!/bin/bash
# GPU job: sample the base model's answers for DPO pairs. Every prompt of val.jsonl
# and of each train_N.jsonl in TRAIN_SIZES (colon-separated) is sampled into one
# shared file, $ROOT/data/dpo/base_samples.jsonl. Prompts already in it are
# skipped, so the nested subsets are sampled once (1000 first, then 59000 adds
# the rest), and a resubmitted job resumes where a preempted one stopped.
# Formal runs use the SCF checkout on a clean, current main.
# Usage (flags repeated on the command line for the safe_submit quota check):
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive \
#       --gres=gpu:H200:1 --export=ALL,TRAIN_SIZES=1000 slurm/build_pairs.sh
# On another node (e.g. thidwick), add -w <node> and point the logs at horton:
#   --output=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/build_pairs_%j.out
#   --error=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/build_pairs_%j.err
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/build_pairs_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/build_pairs_%j.err
set -euo pipefail
: "${TRAIN_SIZES:?set TRAIN_SIZES, e.g. 1000 or 1000:59000}"

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
# Data, environment, and HF cache live on horton's /data; other nodes reach
# them through the /net/horton autofs path.
DATA=/data/wenruixu
[[ $(hostname -s) == horton ]] || DATA=/net/horton/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
PY=$DATA/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SAMPLES=$ROOT/data/dpo/base_samples.jsonl
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

FILES=("$ROOT/data/processed/val.jsonl")
for TRAIN_SIZE in ${TRAIN_SIZES//:/ }; do
    FILES+=("$ROOT/data/processed/train_${TRAIN_SIZE}.jsonl")
done
for FILE in "${FILES[@]}"; do
    echo "sample $FILE $(date -Iseconds)"
    "$PY" "$CODE/train/build_pairs.py" --revision "$MODEL_REVISION" \
        --data-file "$FILE" --output "$SAMPLES"
done
echo "end $(date -Iseconds)"
