#!/usr/bin/env python3
"""Merge a LoRA adapter into Qwen3-1.7B and save a full model directory.

vLLM (and so BFCL) serves a plain model directory. The merged weights are
saved in bf16, the training precision, together with the base tokenizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True, help="Hugging Face commit of the base model")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"merged {args.adapter} -> {args.output_dir}")


if __name__ == "__main__":
    main()
