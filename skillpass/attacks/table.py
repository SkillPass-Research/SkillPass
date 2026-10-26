"""The public embedding table and tokenizer a relative gives the adversary (Section 2.3), fetched
from a checkpoint's shards without loading the model."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skillpass import registry


@dataclass(frozen=True)
class DecodeTable:
    """What an outsider holds: the table `(vocab, width)` as float32, text to ids (no special
    tokens) and ids to text."""

    table: np.ndarray
    tokenise: Callable[[str], Sequence[int]]
    detokenise: Callable[[Sequence[int]], str]


def _table_name(names: Sequence[str]) -> str:
    found = [n for n in names if n.endswith("embed_tokens.weight")]
    if not found:
        raise ValueError("the checkpoint has no embed_tokens.weight")
    in_language = [n for n in found if "language_model" in n]
    return (in_language or found)[0]


def table_for(entry: registry.Reader) -> np.ndarray:
    """`entry`'s token-embedding table as float32, read by partial safetensors loading at the
    pinned revision, in the coordinates the first block receives (scaled for Gemma)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    token = os.environ.get("HF_TOKEN") or None
    repo, revision = entry.reader_id, entry.revision
    try:
        index = hf_hub_download(
            repo, "model.safetensors.index.json", revision=revision, token=token
        )
        weight_map = json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"]
    except Exception:  # noqa: BLE001 - a single-file checkpoint has no index
        weight_map = None
    if weight_map is None:
        path = hf_hub_download(repo, "model.safetensors", revision=revision, token=token)
        with safe_open(path, "pt") as h:
            weight = h.get_tensor(_table_name(list(h.keys())))
    else:
        name = _table_name(list(weight_map))
        path = hf_hub_download(repo, weight_map[name], revision=revision, token=token)
        with safe_open(path, "pt") as h:
            weight = h.get_tensor(name)
    if entry.family in registry.SCALED_FAMILIES:
        config = json.loads(
            Path(hf_hub_download(repo, "config.json", revision=revision, token=token)).read_text()
        )
        width = (config.get("text_config") or config)["hidden_size"]
        weight = weight * torch.tensor(width**0.5).to(weight.dtype)
    return weight.float().numpy()


def decode_table_for(entry: registry.Reader) -> DecodeTable:
    """`entry`'s table and tokenizer: what the model-free adversary holds."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        entry.reader_id, revision=entry.revision, token=os.environ.get("HF_TOKEN") or None
    )
    return DecodeTable(
        table=table_for(entry),
        tokenise=lambda text: tokenizer(text, add_special_tokens=False).input_ids,
        detokenise=tokenizer.decode,
    )
