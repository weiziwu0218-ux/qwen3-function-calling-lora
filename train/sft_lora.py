#!/usr/bin/env python3
"""LoRA SFT of Qwen3-1.7B on the converted xLAM data.

Reads the JSONL written by scripts/prepare_xlam.py and trains on the
``completion`` tokens only. The run directory receives the LoRA adapter,
``run_config.json`` (everything needed to reproduce or evaluate the run),
and ``metrics.json`` (validation loss and the cost record).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from datetime import datetime
from pathlib import Path

import peft
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True, help="Hugging Face commit of the base model")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8, help="per-device micro batch")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    return p.parse_args()


def tokenize_fn(tokenizer):
    # The converted completion ends with "<|im_end|>\n". Generation stops at
    # <|im_end|> (the tokenizer's eos), so the trailing newline is dropped and
    # the target sequence ends exactly where inference ends.
    def fn(row):
        completion = row["completion"].removesuffix("\n")
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    return fn


def collate(pad_id: int):
    def fn(batch):
        width = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, mask = [], [], []
        for row in batch:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            mask.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return fn


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = load_dataset(
        "json", data_files={"train": str(args.train_file), "val": str(args.val_file)}
    )
    data = data.map(tokenize_fn(tokenizer), remove_columns=data["train"].column_names)
    train_lengths = [len(ids) for ids in data["train"]["input_ids"]]
    train_tokens = sum(train_lengths)
    target_tokens = sum(sum(1 for t in labels if t != -100) for labels in data["train"]["labels"])

    run_config = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "algorithm": "sft",
        "base_model": args.model,
        "base_revision": args.revision,
        "train_file": str(args.train_file),
        "train_rows": len(data["train"]),
        "val_file": str(args.val_file),
        "val_rows": len(data["val"]),
        "seed": SEED,
        "lora": {"r": args.rank, "alpha": args.alpha, "dropout": args.dropout,
                 "target_modules": TARGET_MODULES},
        "optim": {"lr": args.lr, "epochs": args.epochs, "per_device_batch": args.batch_size,
                  "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio,
                  "scheduler": "cosine", "precision": "bf16"},
        "loss": "completion tokens only; trailing newline after <|im_end|> dropped",
        "thinking_mode": "non-thinking (target starts with an empty think block)",
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": transformers.__version__,
                     "peft": peft.__version__},
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        data_collator=collate(tokenizer.pad_token_id),
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - start
    peak_train = torch.cuda.max_memory_allocated()

    val = trainer.evaluate()
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")

    n_gpus = torch.cuda.device_count()
    metrics = {
        "val_loss": val["eval_loss"],
        "train_loss": train_result.training_loss,
        "cost": {
            "gpu_name": torch.cuda.get_device_name(0),
            "n_gpus": n_gpus,
            "train_wall_seconds": train_seconds,
            "gpu_hours": train_seconds * n_gpus / 3600,
            "peak_gpu_mem_gb": peak_train / 1e9,
            "trainable_params": trainable,
            "epoch_tokens": train_tokens,
            "epoch_target_tokens": target_tokens,
            "tokens_per_second": train_tokens * args.epochs / train_seconds,
            "optimizer_steps": train_result.global_step,
        },
        "length_stats": {
            "max": max(train_lengths),
            "mean": train_tokens / len(train_lengths),
            "p99": sorted(train_lengths)[math.ceil(0.99 * len(train_lengths)) - 1],
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
