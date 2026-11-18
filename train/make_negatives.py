#!/usr/bin/env python3
"""Build irrelevance negatives from the converted xLAM data.

xLAM has no negatives: every answer calls a function. A negative keeps a row's
user query but offers no tool that answers it, and its target is a fixed text
reply with no tool call. The functions the xLAM answer calls are removed from
the row's tools ("removed"); when no tool is left, the tools of another row of
the same file that shares no function name with this row are offered instead
("swapped").

Rows are rendered with the Qwen3-1.7B chat template from the tokenizer, the
template scripts/prepare_xlam.py rendered the positives with, and every source
row must reproduce its stored text first. Written to --out-dir, seed 42:
- neg_<N>.jsonl for each N in --sizes, drawn from --train-file and nested, so
  each file is a prefix of the next;
- train_<M>_neg<N>.jsonl: all M rows of --train-file plus neg_<N>, shuffled;
- val_neg<K>.jsonl: K negatives drawn from --val-file, for scoring refusals.
Negatives take id 100000 + the source row's id.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from eval_val import EMPTY_THINK
from sft_lora import SEED

REFUSAL = "None of the provided functions can be used for this request."
ID_OFFSET = 100000
MARKER = "<|im_start|>assistant\n"
# Keys of a converted row (scripts/prepare_xlam.py); the shuffled training mix
# keeps only these so that positives and negatives share one schema.
PUBLIC_KEYS = ("id", "source", "n_calls", "n_tools", "messages", "tools", "prompt", "completion", "text")


def render(tokenizer, messages, tools) -> tuple[str, str, str]:
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    idx = text.rfind(MARKER) + len(MARKER)
    return text[:idx], text[idx:], text


def tool_names(tools) -> set[str]:
    return {tool["function"]["name"] for tool in tools}


def make_negatives(tokenizer, rows: list[dict], k: int, kinds: Counter) -> list[dict]:
    """Turn k rows drawn from rows (seed 42) into negatives, in draw order."""
    rng = random.Random(SEED)
    negatives = []
    for row in rng.sample(rows, k):
        called = {call["function"]["name"] for call in row["messages"][-1]["tool_calls"]}
        tools = [tool for tool in row["tools"] if tool["function"]["name"] not in called]
        kind = "removed"
        if not tools:
            taken = tool_names(row["tools"]) | called
            donor = rng.choice(rows)
            while tool_names(donor["tools"]) & taken:
                donor = rng.choice(rows)
            tools, kind = donor["tools"], "swapped"
        kinds[kind] += 1
        messages = [row["messages"][0], {"role": "assistant", "content": REFUSAL, "tool_calls": []}]
        prompt, completion, text = render(tokenizer, messages, tools)
        if completion != EMPTY_THINK + REFUSAL + "<|im_end|>\n":
            raise SystemExit(f"row {row['id']}: unexpected negative target {completion!r}")
        negatives.append({
            "id": ID_OFFSET + row["id"], "source": row["source"], "n_calls": 0,
            "n_tools": len(tools), "messages": messages, "tools": tools, "prompt": prompt,
            "completion": completion, "text": text, "source_id": row["id"], "negative_kind": kind,
        })
    return negatives


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--revision", required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--val-file", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000])
    p.add_argument("--val-size", type=int, default=200)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    train = [json.loads(line) for line in args.train_file.open()]
    val = [json.loads(line) for line in args.val_file.open()]
    for row in train + val:
        if render(tokenizer, row["messages"], row["tools"])[2] != row["text"]:
            raise SystemExit(f"row {row['id']}: the tokenizer's chat template does not reproduce its text")

    summary = {"seed": SEED, "refusal": REFUSAL, "train_file": str(args.train_file),
               "val_file": str(args.val_file), "files": {}}
    kinds = Counter()
    negatives = make_negatives(tokenizer, train, max(args.sizes), kinds)
    for size in sorted(args.sizes):
        subset = negatives[:size]
        write_jsonl(args.out_dir / f"neg_{size}.jsonl", subset)
        mix = [{key: row[key] for key in PUBLIC_KEYS} for row in train + subset]
        random.Random(SEED).shuffle(mix)
        name = f"train_{len(train)}_neg{size}.jsonl"
        write_jsonl(args.out_dir / name, mix)
        summary["files"][f"neg_{size}.jsonl"] = dict(Counter(row["negative_kind"] for row in subset))
        summary["files"][name] = {"rows": len(mix), "negatives": size}

    val_kinds = Counter()
    val_neg = make_negatives(tokenizer, val, args.val_size, val_kinds)
    write_jsonl(args.out_dir / f"val_neg{args.val_size}.jsonl", val_neg)
    summary["files"][f"val_neg{args.val_size}.jsonl"] = dict(val_kinds)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
