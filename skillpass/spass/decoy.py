"""The decoy: an unrelated skill document aligned to the private skill by position (Section 3.2).
Its token ids are cycled or cut to the skill's length m."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

Tokenise = Callable[[str], Sequence[int]]


def align(ids: Sequence[int], length: int) -> np.ndarray:
    """`ids` cycled or cut to `length`."""
    if length < 1:
        raise ValueError(f"a decoy needs a positive length, got {length}")
    array = np.asarray(list(ids), dtype=np.int64)
    if array.size == 0:
        raise ValueError("the decoy has no tokens")
    return np.tile(array, math.ceil(length / array.size))[:length]


def decoy_ids(tokenise: Tokenise, decoy_source: str, length: int) -> np.ndarray:
    """The decoy document's token ids under `tokenise`, aligned to a skill of `length` tokens."""
    return align(tokenise(decoy_source), length)
