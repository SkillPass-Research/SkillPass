"""A skill as the authorized model's token ids and embeddings e*, and each token's margin mu_i:
the distance from its row of the embedding table to the nearest other row (Equation 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NEAREST_CHUNK = 8192


@dataclass(frozen=True)
class Embedded:
    """Ids `(m,)`, embeddings `(m, d)` float32 and margins `(m,)`."""

    ids: np.ndarray
    embeddings: np.ndarray
    margins: np.ndarray


def table_of(embedding: Any) -> Any:
    """The rows the embedding module puts into the model, as float32: its weight, times the
    scale a Gemma embedding applies in its forward. Every other family gets its weight."""
    scale = getattr(embedding, "embed_scale", None)
    weight = embedding.weight.detach()
    if scale is None:
        return weight.float()
    return (weight * scale.to(weight.dtype)).float()


def margins_of(table: Any, ids: Any, chunk: int = NEAREST_CHUNK) -> Any:
    """For each id, the L2 distance from its row of `table` to the nearest other row; `torch`
    tensors on one device, computed in chunks so the full distance matrix is never held."""
    import torch

    table_sq = (table**2).sum(dim=1)
    out = torch.empty(len(ids), device=table.device, dtype=table.dtype)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        e = table[part]
        d2 = (e**2).sum(dim=1, keepdim=True) - 2 * e @ table.T + table_sq[None, :]
        d2[torch.arange(len(part), device=table.device), part] = float("inf")
        out[start : start + chunk] = d2.min(dim=1).values.clamp_min(0).sqrt()
    return out


def nearest_ids(table: Any, vectors: Any, chunk: int = NEAREST_CHUNK) -> Any:
    """For each row of `vectors`, the id of the nearest row of `table` in L2 (`torch`)."""
    import torch

    table_sq = (table**2).sum(dim=1)
    out = torch.empty(len(vectors), device=table.device, dtype=torch.long)
    for start in range(0, len(vectors), chunk):
        v = vectors[start : start + chunk]
        out[start : start + chunk] = (table_sq[None, :] - 2 * v @ table.T).argmin(dim=1)
    return out


def embed_skill(tokenizer: Any, embedding: Any, source: str) -> Embedded:
    """`source` tokenised with no special tokens, embedded through `embedding`, with margins."""
    import torch

    ids = tokenizer(source, add_special_tokens=False, return_tensors="pt").input_ids[0]
    ids = ids.to(embedding.weight.device)
    with torch.no_grad():
        table = table_of(embedding)
        vectors = table[ids]
        margins = margins_of(table, ids)
    return Embedded(
        ids=ids.cpu().numpy().astype(np.int64),
        embeddings=vectors.cpu().numpy().astype(np.float32),
        margins=margins.cpu().numpy().astype(np.float32),
    )
