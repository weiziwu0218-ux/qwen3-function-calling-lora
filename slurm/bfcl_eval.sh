#!/bin/bash
# GPU job: BFCL on merged models, one model after another on one GPU. MODELS and
# CATEGORIES are colon-separated lists (directories under $ROOT/merged, BFCL test
# categories). vLLM runs with the settings of the team's Experiment A BFCL runs
# (fp16, V0 engine, eager, one sequence at a time, 4096-token context), and
# bfcl/bfcl_nothink.py sends the non-thinking prompts to it.
# Formal runs use the SCF checkout on a clean, current main and write to
# $ROOT/bfcl/$EXP_<job>; SMOKE=1 writes to $ROOT/bfcl/smoke/<job> instead.
# Usage (flags repeated on the command line for the safe_submit quota check):
#   /accounts/projects/berkeleynlp/anyaji/safe_submit.sh sbatch --qos=preemptive --gres=gpu:H200:1 \
#       --export=ALL,EXP=expB,MODELS=sft_n59000_r4rs_4144596,CATEGORIES=simple_python:multiple slurm/bfcl_eval.sh
# On another node (e.g. feanor in the songmei partition, qos normal), add -p <partition>
# -w <node> and point the logs at horton:
#   --output=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/bfcl_eval_%j.out
#   --error=/net/horton/data/wenruixu/workspace/qwen3-function-calling-lora/logs/bfcl_eval_%j.err
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --no-requeue
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/bfcl_eval_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/bfcl_eval_%j.err
set -euo pipefail
: "${EXP:?set EXP, e.g. expB}" "${MODELS:?set MODELS}" "${CATEGORIES:?set CATEGORIES}"

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
# Data, environment, and HF cache live on horton's /data; other nodes reach
# them through the /net/horton autofs path. The venv's console scripts carry
# horton-local shebangs, so vllm, bfcl and pip run through python -m.
DATA=/data/wenruixu
[[ $(hostname -s) == horton ]] || DATA=/net/horton/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
PY=$DATA/venv/bfcl/bin/python
BFCL_MODEL=Qwen/Qwen3-1.7B-FC
CATS=${CATEGORIES//:/,}
PORT=$((20000 + SLURM_JOB_ID % 20000))
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE
export VLLM_USE_V1=0
export LOCAL_SERVER_PORT=$PORT

if [[ "${SMOKE:-0}" == 1 ]]; then
    OUT=$ROOT/bfcl/smoke/${EXP}_${SLURM_JOB_ID}
else
    BRANCH=$(git -C "$CODE" rev-parse --abbrev-ref HEAD)
    if [[ "$BRANCH" != main || -n "$(git -C "$CODE" status --porcelain)" ]]; then
        echo "formal runs need a clean main checkout (found $BRANCH)" >&2
        exit 1
    fi
    OUT=$ROOT/bfcl/${EXP}_${SLURM_JOB_ID}
fi
mkdir -p "$OUT"

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD) out $OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY" -m pip freeze | grep -iE "^(bfcl[-_]eval|vllm|torch|transformers)=="

SERVER=""
trap '[[ -n $SERVER ]] && kill $SERVER 2>/dev/null' EXIT

for NAME in ${MODELS//:/ }; do
    MODEL=$ROOT/merged/$NAME
    RUN=$OUT/$NAME
    mkdir -p "$RUN"
    echo "model $NAME $(date -Iseconds)"
    "$PY" -m vllm.entrypoints.cli.main serve "$MODEL" --port "$PORT" --dtype float16 --max-model-len 4096 \
        --gpu-memory-utilization 0.6 --swap-space 0 --enforce-eager --max-num-seqs 1 --seed 42 \
        > "$RUN/vllm_server.log" 2>&1 &
    SERVER=$!
    until curl -sf "localhost:$PORT/health" > /dev/null; do
        kill -0 "$SERVER"
        sleep 5
    done
    BFCL_PROJECT_ROOT=$RUN "$PY" "$CODE/bfcl/bfcl_nothink.py" generate \
        --model "$BFCL_MODEL" --test-category "$CATS" --skip-server-setup \
        --local-model-path "$MODEL" --temperature 0.001 --include-input-log
    BFCL_PROJECT_ROOT=$RUN "$PY" -m bfcl_eval evaluate --model "$BFCL_MODEL" --test-category "$CATS"
    kill "$SERVER"
    wait "$SERVER" || true
    SERVER=""
done
echo "end $(date -Iseconds)"
