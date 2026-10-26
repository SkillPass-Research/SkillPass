"""What the model-free adversary reads from an S-Pass with a public embedding table (Section 5.4).

`nearest_token` maps each vector to the nearest table row by L2: the decode. `strip` removes the
leading principal directions of the tensor first, in case what moved it off its tokens is
low-dimensional (subspace stripping). `decode_report` runs the decode at every stripping rank
and scores each readout against the skill (Decode RL, decode CER, token recovery) and against
the decoy; it also reports the floor, the decoy's own text scored against the skill, which is
what a readout no closer to the skill than an unrelated document scores. Everything is
`numpy`; the tokenizer is passed in as `detokenise`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from skillpass.metrics import cer, rouge_l
from skillpass.registry import STRIP_RANKS

Detokenise = Callable[[Sequence[int]], str]
NEAREST_CHUNK = 4096
_ROUNDING = 1e-9


def _as_matrix(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    return array


def nearest_token(table: ArrayLike, vectors: ArrayLike, chunk: int = NEAREST_CHUNK) -> np.ndarray:
    """The id of each vector's nearest row of `table` by L2, over the table in chunks, with a tie
    to the lowest row index."""
    table = _as_matrix(table, "table")
    vectors = _as_matrix(vectors, "vectors").astype(np.float64)
    if vectors.shape[1] != table.shape[1]:
        raise ValueError(f"vectors have width {vectors.shape[1]}, the table {table.shape[1]}")
    n = len(vectors)
    v_sq = (vectors**2).sum(axis=1)
    best_id = np.zeros(n, dtype=np.int64)
    best = np.full(n, np.inf)
    for start in range(0, len(table), chunk):
        rows = table[start : start + chunk].astype(np.float64)
        rows_sq = (rows**2).sum(axis=1)
        d2 = v_sq[:, None] - 2.0 * vectors @ rows.T + rows_sq[None, :]
        tolerance = _ROUNDING * (v_sq + rows_sq.max())
        near, col = np.nonzero(d2 <= d2.min(axis=1, keepdims=True) + tolerance[:, None])
        exact = ((rows[col] - vectors[near]) ** 2).sum(axis=1)
        for i, j, value in zip(near, col, exact, strict=True):
            if value < best[i]:
                best[i] = value
                best_id[i] = start + j
    return best_id


def available_directions(vectors: ArrayLike) -> int:
    s, d = _as_matrix(vectors, "vectors").shape
    return max(min(s - 1, d), 0)


def strip(vectors: ArrayLike, rank: int) -> np.ndarray:
    """`vectors` with the `rank` leading principal directions of the centred tensor removed;
    rank 0 returns them unchanged."""
    array = _as_matrix(vectors, "vectors").astype(np.float64)
    available = available_directions(array)
    if rank < 0 or (rank > 0 and rank >= available):
        raise ValueError(f"cannot strip {rank} of {available} directions")
    if rank == 0:
        return array.copy()
    mean = array.mean(axis=0, keepdims=True)
    centred = array - mean
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    leading = vh[:rank]
    return array - (centred @ leading.T) @ leading


def decode_report(
    table: ArrayLike,
    tensor: ArrayLike,
    ids: ArrayLike,
    decoy_ids: ArrayLike,
    *,
    detokenise: Detokenise,
    skill_text: str,
    ranks: Sequence[int] = STRIP_RANKS,
    chunk: int = NEAREST_CHUNK,
) -> dict[str, Any]:
    """Every model-free readout of one tensor, as raw numbers.

    `table` is the public embedding table `(vocab, d)`; `tensor` `(m, d)`; `ids` the skill's
    token ids and `decoy_ids` the aligned decoy's, both `(m,)`; `skill_text` the skill's repeat
    target. One row per rank in `ranks`: token recovery against the skill and the decoy, decode
    CER, ROUGE-L against the skill and against the decoy, and the readout as text. Beside the
    rows: `decoy_text`, `floor_rougeL` (the decoy's text against the skill) and
    `best_rougeL_skill`, the Decode RL of the tables (the best over the applicable ranks).
    """
    table = _as_matrix(table, "table")
    tensor = _as_matrix(tensor, "tensor").astype(np.float64)
    if tensor.shape[1] != table.shape[1]:
        raise ValueError(f"the tensor has width {tensor.shape[1]}, the table {table.shape[1]}")
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    decoy_ids = np.asarray(decoy_ids, dtype=np.int64).reshape(-1)
    if ids.shape != (len(tensor),) or decoy_ids.shape != (len(tensor),):
        raise ValueError("ids and decoy_ids must have one entry per tensor row")
    decoy_text = detokenise(decoy_ids.tolist())
    available = available_directions(tensor)
    rows = []
    for rank in ranks:
        if rank > 0 and rank >= available:
            rows.append({"rank": rank, "applicable": False})
            continue
        decoded = nearest_token(table, strip(tensor, rank), chunk)
        text = detokenise(decoded.tolist())
        rows.append(
            {
                "rank": rank,
                "applicable": True,
                "token_recovery": float(np.mean(decoded == ids)),
                "decoy_recovery": float(np.mean(decoded == decoy_ids)),
                "decode_cer": cer(skill_text, text),
                "decoy_cer": cer(decoy_text, text) if decoy_text.strip() else None,
                "rougeL_skill": rouge_l(skill_text, text),
                "rougeL_decoy": rouge_l(decoy_text, text) if decoy_text.strip() else None,
                "readout": text,
            }
        )
    applicable = [r for r in rows if r["applicable"]]
    return {
        "tokens": int(tensor.shape[0]),
        "width": int(tensor.shape[1]),
        "vocab": int(table.shape[0]),
        "available_directions": available,
        "decoy_text": decoy_text,
        "floor_rougeL": rouge_l(skill_text, decoy_text),
        "best_rougeL_skill": max(r["rougeL_skill"] for r in applicable),
        "lowest_decode_cer": min(r["decode_cer"] for r in applicable),
        "strip": rows,
    }
