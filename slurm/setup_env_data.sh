#!/bin/bash
# CPU job: build the Python environment, download Qwen3-1.7B and xLAM, and
# regenerate the converted data. The new manifest is compared with the
# committed data/processed/manifest.json.
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=work
set -euo pipefail

CODE=/accounts/projects/berkeleynlp/wenruixu/workspace/qwen3-function-calling-lora
ROOT=/data/wenruixu/workspace/qwen3-function-calling-lora
VENV=/data/wenruixu/venv/qwen3-fc
MODEL_REVISION=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export HF_HOME=/data/wenruixu/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub

echo "start $(date -Iseconds) host $(hostname) job ${SLURM_JOB_ID} commit $(git -C "$CODE" rev-parse HEAD)"
mkdir -p "$ROOT/data/raw" "$ROOT/data/processed" "$ROOT/env"

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "torch>=2.8,<2.10" --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/pip" install -r "$CODE/requirements.txt"
"$VENV/bin/pip" freeze > "$ROOT/env/pip_freeze_${SLURM_JOB_ID}.txt"

"$VENV/bin/hf" download Qwen/Qwen3-1.7B --revision "$MODEL_REVISION"
"$VENV/bin/hf" download Salesforce/xlam-function-calling-60k xlam_function_calling_60k.json \
    --repo-type dataset --local-dir "$ROOT/data/raw"

"$VENV/bin/python" "$CODE/scripts/prepare_xlam.py" \
    --raw "$ROOT/data/raw/xlam_function_calling_60k.json" --out "$ROOT/data/processed"

"$VENV/bin/python" - "$CODE/data/processed/manifest.json" "$ROOT/data/processed/manifest.json" <<'EOF'
import json, sys
committed, regenerated = (json.load(open(p)) for p in sys.argv[1:])
for key in ("raw_sha256", "sha256", "counts"):
    print(key, "MATCH" if committed[key] == regenerated[key] else "DIFFER")
if committed["sha256"] != regenerated["sha256"]:
    sys.exit("regenerated data differs from the committed manifest")
EOF
echo "end $(date -Iseconds)"
