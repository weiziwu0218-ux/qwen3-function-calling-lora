#!/usr/bin/env python3
"""Generate tool calls on val.jsonl and score them against the xLAM answers.

Prompts use Qwen3 non-thinking mode: the converted ``prompt`` plus an empty
think block, which is what ``apply_chat_template(..., enable_thinking=False)``
produces. Decoding is greedy, so the result is deterministic.

A prediction is an exact match when its multiset of (name, arguments) equals
the reference, ignoring call order. A parse failure is an output with no
<tool_call> block or with a block that is not valid JSON. A negative row (one
whose reference makes no call, from make_negatives.py) is correct when the
output makes no tool call; parse failures are counted on the other rows only.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

EMPTY_THINK = "<think>\n\n</think>\n\n"
CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_calls(text: str):
    """
    Returns:
        list: Valid tool calls, or [] for no tool call.
        None: Malformed tool-call output.
    """
    blocks = CALL_RE.findall(text)

    if not blocks:
        # A malformed tool-call tag is still a parse failure.
        if "<tool_call" in text or "</tool_call" in text:
            return None

        # No tool call was made.
        return []

    # Reject unbalanced or extra tool-call tags.
    if (
        text.count("<tool_call>") != len(blocks)
        or text.count("</tool_call>") != len(blocks)
    ):
        return None

    calls = []

    for block in blocks:
        try:
            call = json.loads(block)

            if not isinstance(call, dict):
                return None

            name = call.get("name")
            arguments = call.get("arguments", {})

            if not isinstance(name, str):
                return None

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if not isinstance(arguments, dict):
                return None

            calls.append(canonical(name, arguments))

        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    return calls


def canonical(name, arguments):
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def reference_calls(row: dict):
    calls = row["messages"][-1]["tool_calls"]
    return [canonical(c["function"]["name"], c["function"]["arguments"]) for c in calls]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True)
    p.add_argument("--adapter", type=Path, help="LoRA adapter dir; omit for the zero-shot base")
    p.add_argument("--val-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval()

    rows = [json.loads(line) for line in args.val_file.open()]
    stats = Counter()
    start = time.time()
    with (args.output_dir / "val_predictions.jsonl").open("w") as out:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i : i + args.batch_size]
            prompts = [row["prompt"] + EMPTY_THINK for row in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.no_grad():
                gen = model.generate(
                    **enc.to(model.device),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for row, text in zip(batch, texts):
                pred = parse_calls(text)
                ref = reference_calls(row)
                if ref:
                    exact = pred is not None and Counter(pred) == Counter(ref)
                else:
                    exact = CALL_RE.search(text) is None
                stats["n"] += 1
                stats["n_negative"] += not ref
                stats["parse_failure"] += bool(ref) and pred is None
                stats["exact_match"] += exact
                out.write(json.dumps({"id": row["id"], "exact_match": exact,
                                      "parse_failure": pred is None, "output": text},
                                     ensure_ascii=False) + "\n")
    seconds = time.time() - start
    n_positive = stats["n"] - stats["n_negative"]

    result = {
        "adapter": str(args.adapter) if args.adapter else None,
        "base_revision": args.revision,
        "thinking_mode": "non-thinking",
        "decoding": "greedy",
        "n": stats["n"],
        "n_negative": stats["n_negative"],
        "exact_match": stats["exact_match"] / stats["n"],
        "parse_failure_rate": stats["parse_failure"] / n_positive if n_positive else None,
        "generation_seconds": seconds,
    }
    (args.output_dir / "val_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
