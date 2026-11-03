#!/usr/bin/env python3
"""The ``bfcl`` CLI with the Qwen3 FC handler in non-thinking mode.

Matches the team's Experiment A BFCL setup, whose logged prompts differ from
stock bfcl-eval 2026.3.23 in two ways:
- the assistant turn opens with an empty think block, as the Qwen3 chat
  template writes it for enable_thinking=False (training and val use the same);
- the context window is the 4096 tokens vLLM serves with, so max_tokens is
  4096 - prompt tokens - 2.
Usage is the same as ``bfcl``, e.g. ``python bfcl_nothink.py generate ...``.
"""

from bfcl_eval.__main__ import cli
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

MAX_MODEL_LEN = 4096
EMPTY_THINK = "<think>\n\n</think>\n\n"

_format_prompt = QwenFCHandler._format_prompt
_spin_up_local_server = QwenFCHandler.spin_up_local_server


def _format_prompt_no_think(self, messages, function):
    return _format_prompt(self, messages, function) + EMPTY_THINK


def _spin_up_local_server_4096(self, *args, **kwargs):
    _spin_up_local_server(self, *args, **kwargs)
    self.max_context_length = MAX_MODEL_LEN


QwenFCHandler._format_prompt = _format_prompt_no_think
QwenFCHandler.spin_up_local_server = _spin_up_local_server_4096

if __name__ == "__main__":
    cli()
