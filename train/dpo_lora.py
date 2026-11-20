#!/usr/bin/env python3
"""LoRA DPO of Qwen3-1.7B on xLAM preference pairs.

A pair is a training prompt, its xLAM answer (chosen) and the first wrong
answer that build_pairs.py sampled for it (rejected). Prompts whose samples
were all correct give no pair, so a run on train_N.jsonl trains on fewer than
N pairs; run_config.json records both counts.

The policy is the base model plus a fresh LoRA adapter, and the reference is
the same model with the adapter switched off. The loss is the sigmoid DPO loss
on the summed log-probabilities of the answer tokens, with the prompt masked as
in sft_lora.py. The run directory receives the same files as an SFT run: the
LoRA adapter, ``run_config.json`` and ``metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import peft
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from eval_val import EMPTY_THINK
from sft_lora import SEED, TARGET_MODULES

IM_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True, help="Hugging Face commit of the base model")
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, required=True)
    p.add_argument("--samples-file", type=Path, required=True, help="output of build_pairs.py")
    p.add_argument("--negatives-file", type=Path,
                   help="make_negatives.py output (neg_<N>.jsonl); its first --num-negatives rows add refusal pairs")
    p.add_argument("--source-file", type=Path,
                   help="the converted file the negatives were drawn from, for their original calls")
    p.add_argument("--num-negatives", type=int, default=0)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--rslora", action="store_true",
                   help="rank-stabilized scaling alpha/sqrt(r) instead of alpha/r")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta: strength of the pull to the reference")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=2, help="per-device micro batch, in pairs")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    return p.parse_args()


def load_pairs(path: Path, samples: dict) -> tuple[int, list[dict]]:
    """Pair each row with its first sampled wrong answer; rows without one are dropped."""
    rows = [json.loads(line) for line in path.open()]
    missing = sum(row["id"] not in samples for row in rows)
    if missing:
        raise SystemExit(f"{missing} rows of {path} have no samples; run build_pairs.py on it")
    pairs = []
    for row in rows:
        wrong = samples[row["id"]]["wrong"]
        if not wrong:
            continue

        chosen = row["completion"].removesuffix("\n")
        if not (chosen.startswith(EMPTY_THINK) and chosen.endswith(IM_END)):
            raise SystemExit(f"unexpected completion format in {path}, id {row['id']}")
        pairs.append({"prompt": row["prompt"], "chosen": chosen,
                      "rejected": EMPTY_THINK + wrong[0] + IM_END})
    if not pairs:
        raise SystemExit(f"no wrong samples for any row of {path}")
    return len(rows), pairs


def load_negative_pairs(path: Path, source_path: Path, k: int) -> list[dict]:
    """Refusal pairs from the first k negatives: on the prompt whose called tools were
    removed, the refusal is chosen and the original xLAM call, which now names a tool
    that is not offered, is rejected."""
    negatives = [json.loads(line) for _, line in zip(range(k), path.open())]
    if len(negatives) < k:
        raise SystemExit(f"{path} has {len(negatives)} negatives, fewer than {k}")
    needed = {row["source_id"] for row in negatives}
    calls = {}
    for line in source_path.open():
        row = json.loads(line)
        if row["id"] in needed:
            calls[row["id"]] = row["completion"].removesuffix("\n")
    return [{"prompt": row["prompt"], "chosen": row["completion"].removesuffix("\n"),
             "rejected": calls[row["source_id"]]} for row in negatives]


def tokenize_fn(tokenizer):
    # Chosen and rejected share the prompt; as in sft_lora.py the prompt
    # tokens are labeled -100 and only the answer tokens are scored.
    def fn(row):
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        out = {}
        for key in ("chosen", "rejected"):
            answer_ids = tokenizer(row[key], add_special_tokens=False)["input_ids"]
            out[f"{key}_input_ids"] = prompt_ids + answer_ids
            out[f"{key}_labels"] = [-100] * len(prompt_ids) + answer_ids
        return out

    return fn


def collate(pad_id: int):
    # All chosen rows, then all rejected rows, padded into one batch so a single
    # forward pass scores both; compute_loss splits them back with chunk(2).
    def fn(batch):
        seqs = [(row[f"{key}_input_ids"], row[f"{key}_labels"])
                for key in ("chosen", "rejected") for row in batch]
        width = max(len(ids) for ids, _ in seqs)
        input_ids, labels, mask = [], [], []
        for ids, row_labels in seqs:
            pad = width - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            labels.append(row_labels + [-100] * pad)
            mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }

    return fn


def sequence_logps(model, batch) -> torch.Tensor:
    """Summed log-probability of the answer tokens of each row."""
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    picked = logits.gather(2, labels.clamp(min=0).unsqueeze(2)).squeeze(2)
    return ((picked - logits.logsumexp(2)) * mask).sum(1)


class DPOTrainer(Trainer):
    def __init__(self, *args, beta: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.model_accepts_loss_kwargs = False
        self.reward_stats = {"train": defaultdict(list), "eval": defaultdict(list)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        policy = sequence_logps(model, inputs)
        with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
            reference = sequence_logps(model, inputs)
        # Implicit rewards beta * log(pi / pi_ref) of the chosen and rejected answers.
        chosen, rejected = (self.beta * (policy - reference)).chunk(2)
        margin = chosen - rejected
        loss = -F.logsigmoid(margin).mean()

        stats = self.reward_stats["train" if model.training else "eval"]
        stats["rewards/chosen"].append(chosen.mean().item())
        stats["rewards/rejected"].append(rejected.mean().item())
        stats["rewards/margin"].append(margin.mean().item())
        stats["rewards/accuracy"].append((margin > 0).float().mean().item())
        return (loss, {}) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        # Trainer logs only the loss; add the reward averages since the last log.
        # Updating in place also puts them in the metrics that evaluate() returns.
        mode, prefix = ("eval", "eval_") if "eval_loss" in logs else ("train", "")
        logs.update({prefix + k: sum(v) / len(v) for k, v in self.reward_stats[mode].items()})
        self.reward_stats[mode].clear()
        super().log(logs, *args, **kwargs)


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    samples = {}
    for line in args.samples_file.open():
        row = json.loads(line)
        samples[row["id"]] = row
    sampler = json.loads(Path(f"{args.samples_file}.config.json").read_text())
    train_prompts, train_pairs = load_pairs(args.train_file, samples)
    val_prompts, val_pairs = load_pairs(args.val_file, samples)
    negative_pairs = []
    if args.num_negatives:
        negative_pairs = load_negative_pairs(args.negatives_file, args.source_file, args.num_negatives)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            use_rslora=args.rslora,
            lora_dropout=args.dropout,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = DatasetDict({"train": Dataset.from_list(train_pairs + negative_pairs),
                        "val": Dataset.from_list(val_pairs)})
    data = data.map(tokenize_fn(tokenizer), remove_columns=data["train"].column_names)
    train_lengths = [len(ids) for key in ("chosen", "rejected")
                     for ids in data["train"][f"{key}_input_ids"]]
    train_tokens = sum(train_lengths)
    target_tokens = sum(sum(1 for t in labels if t != -100)
                        for key in ("chosen", "rejected") for labels in data["train"][f"{key}_labels"])

    run_config = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "algorithm": "dpo",
        "base_model": args.model,
        "base_revision": args.revision,
        "train_file": str(args.train_file),
        "train_prompts": train_prompts,
        "train_pairs": len(train_pairs),
        "negatives_file": str(args.negatives_file) if negative_pairs else None,
        "negative_pairs": len(negative_pairs),
        "val_file": str(args.val_file),
        "val_prompts": val_prompts,
        "val_pairs": len(val_pairs),
        "samples_file": str(args.samples_file),
        "rejected_sampler": sampler,
        "seed": SEED,
        "beta": args.beta,
        "lora": {"r": args.rank, "alpha": args.alpha, "rslora": args.rslora,
                 "scaling": args.alpha / (args.rank ** 0.5 if args.rslora else args.rank),
                 "dropout": args.dropout, "target_modules": TARGET_MODULES},
        "optim": {"lr": args.lr, "epochs": args.epochs, "per_device_pairs": args.batch_size,
                  "grad_accum": args.grad_accum, "warmup_ratio": args.warmup_ratio,
                  "scheduler": "cosine", "precision": "bf16"},
        "loss": "sigmoid DPO on summed answer-token log-probs; reference is the base with the "
                "adapter off; rejected is the first distinct wrong sample, closed with <|im_end|>",
        "thinking_mode": "non-thinking (answers start with an empty think block)",
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
        prediction_loss_only=True,
        dataloader_num_workers=2,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        data_collator=collate(tokenizer.pad_token_id),
        beta=args.beta,
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
        "val_reward_accuracy": val["eval_rewards/accuracy"],
        "val_reward_margin": val["eval_rewards/margin"],
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
