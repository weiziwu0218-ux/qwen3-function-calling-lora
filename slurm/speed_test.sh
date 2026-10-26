#!/bin/bash
# GPU job: LoRA SFT speed test on train_1000, then validation-set scoring of
# the trained adapter and of the zero-shot base model.
# Submit through the group wrapper. The wrapper reads only command-line flags,
# so repeat --qos and --gres there for its quota check:
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive \
#       --gres=gpu:H200:1 slurm/speed_test.sh
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/smoke/speed_test_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/smoke/speed_test_%j.err
set -euo pipefail

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
ROOT=/data/wenruixu/workspace/qwen3-function-calling-lora
PY=/data/wenruixu/venv/qwen3-fc/bin/python
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=/data/wenruixu/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
# SCF's /etc/profile.d/SITE-huggingface.sh points TRANSFORMERS_CACHE at an empty
# /scratch cache, which overrides HF_HUB_CACHE in transformers 4.x.
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE

RUN=$ROOT/runs/smoke/speed_sft_n1000_r16_${SLURM_JOB_ID}
echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

"$PY" "$CODE/train/sft_lora.py" --revision "$MODEL_REVISION" \
    --train-file "$ROOT/data/processed/train_1000.jsonl" \
    --val-file "$ROOT/data/processed/val.jsonl" \
    --output-dir "$RUN" --rank 16

"$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" \
    --adapter "$RUN/adapter" --val-file "$ROOT/data/processed/val.jsonl" \
    --output-dir "$RUN/val"

"$PY" "$CODE/train/eval_val.py" --revision "$MODEL_REVISION" \
    --val-file "$ROOT/data/processed/val.jsonl" \
    --output-dir "$ROOT/runs/smoke/zero_shot_base_${SLURM_JOB_ID}/val"
echo "end $(date -Iseconds)"
