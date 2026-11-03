#!/bin/bash
# CPU job: build the BFCL environment and merge the adapters that BFCL evaluates.
# BFCL serves models with vLLM 0.8.5, which needs its own torch, so it gets a
# separate venv; merging uses the training venv. MERGE is a colon-separated list
# of run directories under $ROOT/runs (e.g. expA/sft_n59000_r16_4110239); each
# merged model goes to $ROOT/merged/<run name>.
#   sbatch --export=ALL,MERGE=expA/sft_n59000_r16_4110239:expB/sft_n59000_r4rs_4144596 slurm/setup_bfcl.sh
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=work
#SBATCH --output=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/setup_bfcl_%j.out
#SBATCH --error=/data/wenruixu/workspace/qwen3-function-calling-lora/logs/setup_bfcl_%j.err
set -euo pipefail
: "${MERGE:?set MERGE, e.g. expA/sft_n59000_r16_4110239}"

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
DATA=/data/wenruixu
ROOT=$DATA/workspace/qwen3-function-calling-lora
VENV=$DATA/venv/bfcl
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
unset TRANSFORMERS_CACHE HF_DATASETS_CACHE

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"

if [[ ! -x $VENV/bin/python ]]; then
    /usr/bin/python3.12 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    # transformers is pinned to the release current with vLLM 0.8.5.
    "$VENV/bin/pip" install -q "bfcl-eval[oss-eval-vllm]==2026.3.23" "transformers==4.51.3"
fi
"$VENV/bin/pip" freeze > "$ROOT/env/pip_freeze_bfcl_${SLURM_JOB_ID}.txt"
"$VENV/bin/python" -c "import vllm, bfcl_eval, transformers, torch; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"

export HF_HUB_OFFLINE=1
for RUN in ${MERGE//:/ }; do
    "$DATA/venv/qwen3-fc/bin/python" "$CODE/bfcl/merge_adapter.py" --revision "$MODEL_REVISION" \
        --adapter "$ROOT/runs/$RUN/adapter" --output-dir "$ROOT/merged/$(basename "$RUN")"
done
du -sh "$VENV" "$ROOT/merged"
echo "end $(date -Iseconds)"
