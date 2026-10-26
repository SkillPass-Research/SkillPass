"""Shared test helpers: a toy public table and tokenizer so the model-free pieces run on CPU."""

from __future__ import annotations

import string

import numpy as np
import pytest

from skillpass.attacks.table import DecodeTable

ALPHABET = string.printable  # 100 characters: digits, letters, punctuation, whitespace
UNKNOWN = ALPHABET.index("?")


def toy_tokenise(text: str) -> list[int]:
    return [ALPHABET.index(c) if c in ALPHABET else UNKNOWN for c in text]


def toy_detokenise(ids) -> str:
    return "".join(ALPHABET[i] if 0 <= i < len(ALPHABET) else "?" for i in ids)


def toy_table(width: int = 16, seed: int = 0) -> DecodeTable:
    rng = np.random.default_rng(seed)
    table = rng.standard_normal((len(ALPHABET), width)).astype(np.float32)
    return DecodeTable(table=table, tokenise=toy_tokenise, detokenise=toy_detokenise)


@pytest.fixture
def toy() -> DecodeTable:
    return toy_table()
