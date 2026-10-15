# xLAM 60k in Qwen3 training format

Shared data for the Qwen3-1.7B LoRA experiments. The JSONL files in this folder are already converted and are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

`manifest.json` records the sha256 of each file. To rebuild from the original xLAM release instead, run `python3 scripts/prepare_xlam.py` from the repo root (seed 42). That downloads `Salesforce/xlam-function-calling-60k` and rewrites this folder.

## Format

Each JSONL row has:

- `messages`: user query, then an assistant turn whose `tool_calls` are the xLAM answers
- `tools`: the available functions as JSON schema (`int` → `integer`, `float` → `number`, `str` → `string`, `bool` → `boolean`, `list` / `tuple` / `set` → `array`, `dict` → `object`). Item types are kept, so `List[int]` becomes an array of integers. This release marks optional parameters inside the type string (`"int, optional"`) rather than with a `required` boolean; those parameters are left out of `parameters.required`. Defaults on the parameter are copied through.
- `text`: that conversation rendered by the Qwen3-1.7B chat template
- `prompt` and `completion`: the split inside `text`. `prompt + completion == text`

Train on `completion` only. The target is an empty `<think>` block followed by one `<tool_call>` block per answer, which is how this tokenizer renders a finished assistant turn:

```text
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "math_toolkit.product_of_primes", "arguments": {"count": 5}}
</tool_call><|im_end|>
```

`messages` and `tools` are what `tokenizer.apply_chat_template(..., tools=tools)` expects. There is no extra system message in `messages`; the template inserts the tool instructions itself.

## Splits

Validation is held out first and is not inside any training file. Use `val.jsonl` for every run, including the later LoRA-rank runs.

Training files are nested prefixes of one stratified pool: `train_1000` ⊂ `train_5000` ⊂ `train_10000` ⊂ `train_30000` ⊂ `train_59000`. `train_full.jsonl` points at the full pool. That full pool is the proposal's 60k condition; 1,000 rows are reserved for validation, so it has 59,000 rows.

Stratification is by generator (dataset card: ids 0–33658 are DeepSeek, the rest Mixtral) and by one call vs multiple calls, so each prefix has about the same mix. `split_ids.json` lists the row ids in validation and in training order.
