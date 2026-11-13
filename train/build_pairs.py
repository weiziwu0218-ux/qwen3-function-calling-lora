#!/usr/bin/env python3
"""Sample answers from the model and keep the wrong ones as DPO rejected answers.

Every row of --data-file gets --num-samples sampled answers in non-thinking
mode, from the same prompt eval_val.py uses. Each answer is scored with
eval_val.py's exact match, and the distinct wrong answers that stopped at
<|im_end|> are kept; answers cut off at --max-new-tokens are only counted.

One JSON line per prompt is appended to --output, and prompts already there
are skipped. The train subsets are nested, so sampling train_1000 and then
train_59000 into the same file samples every prompt once, and a preempted job
resumes where it stopped. The sampling settings are saved next to the output
in <output>.config.json, and appending with different settings is refused.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from eval_val import EMPTY_THINK, parse_calls, reference_calls
from sft_lora import SEED

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]


def is_correct(text: str, ref) -> bool:
    pred = parse_calls(text)
    try:
        return pred is not None and Counter(pred) == Counter(ref)
    except TypeError:  # a sampled call can name a list or dict, which cannot be counted
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="samples JSONL, appended to")
    p.add_argument("--num-samples", type=int, default=8)
    # Qwen3's recommended sampling settings for non-thinking mode.
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="prompts per batch")
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()

    settings = {
        "model": args.model, "revision": args.revision, "thinking_mode": "non-thinking",
        "num_samples": args.num_samples, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
        "seed": SEED,
    }
    config_path = Path(f"{args.output}.config.json")
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != settings:
            raise SystemExit(f"{args.output} was sampled with different settings: {saved}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(settings, indent=2) + "\n")

    done = set()
    if args.output.exists():
        done = {json.loads(line)["id"] for line in args.output.open()}
    rows = [json.loads(line) for line in args.data_file.open()]
    todo = [row for row in rows if row["id"] not in done]
    print(f"{len(todo)} of {len(rows)} prompts to sample", flush=True)

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    model.eval()
    stop_ids = set(tokenizer.convert_tokens_to_ids(STOP_TOKENS))

    n = args.num_samples
    with args.output.open("a") as out:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            prompts = [row["prompt"] + EMPTY_THINK for row in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(
                    **enc.to(model.device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    num_return_sequences=n,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = gen[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j, row in enumerate(batch):
                ref = reference_calls(row)
                record = {"id": row["id"], "correct": 0, "truncated": 0, "wrong": []}
                # generate returns the n samples of each prompt next to each other.
                for k in range(j * n, (j + 1) * n):
                    if not stop_ids & set(gen[k].tolist()):
                        record["truncated"] += 1
                    elif is_correct(texts[k], ref):
                        record["correct"] += 1
                    elif texts[k] not in record["wrong"]:
                        record["wrong"].append(texts[k])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(f"sampled {i + len(batch)}/{len(todo)}", flush=True)

    ids = {row["id"] for row in rows}
    records = [r for r in (json.loads(line) for line in args.output.open()) if r["id"] in ids]
    with_wrong = sum(bool(r["wrong"]) for r in records)
    summary = {
        "data_file": str(args.data_file),
        "prompts": len(records),
        "prompts_with_wrong_sample": with_wrong,
        "pair_rate": with_wrong / len(records),
        "samples_correct": sum(r["correct"] for r in records) / (n * len(records)),
        "samples_truncated": sum(r["truncated"] for r in records) / (n * len(records)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
