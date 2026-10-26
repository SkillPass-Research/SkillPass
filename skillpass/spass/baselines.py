"""The two defence baselines of Section 5.1, made from the public embedding table alone.

- `offset`: the skill's own embeddings, each moved `OFFSET_MARGINS` (2.2) times its nearest-
  neighbour margin along a random direction. Section 4.2 says why this hides nothing: a
  random direction is almost orthogonal to the direction towards any token.
- `decoy`: the decoy's embeddings unchanged, the point the S-Pass starts from.

Each is written as an S-Pass is, with a sidecar naming the construction, so the runs read it
through the same path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from skillpass import registry
from skillpass.attacks.decode import nearest_token
from skillpass.attacks.table import DecodeTable
from skillpass.metrics import cer
from skillpass.registry import CONSTRUCTION_DECOY, CONSTRUCTION_OFFSET, OFFSET_MARGINS
from skillpass.skills import Skill
from skillpass.spass.decoy import decoy_ids as align_decoy
from skillpass.spass.file import source_sha256, write_plain, write_spass

BASELINES = (CONSTRUCTION_OFFSET, CONSTRUCTION_DECOY)
MARGIN_CHUNK = 1024


def margins(table: np.ndarray, ids: np.ndarray, chunk: int = MARGIN_CHUNK) -> np.ndarray:
    """Each id's distance to the nearest other row of `table`, in `numpy`."""
    table = np.asarray(table, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int64)
    squares = (table**2).sum(axis=1)
    out = np.empty(len(ids), dtype=np.float32)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        rows = table[part]
        d2 = (rows**2).sum(axis=1, keepdims=True) - 2.0 * rows @ table.T + squares[None, :]
        d2[np.arange(len(part)), part] = np.inf
        out[start : start + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def offset_tensor(
    embeddings: np.ndarray, margin: np.ndarray, scale: float, seed: int
) -> np.ndarray:
    """Each embedding moved `scale` margins along a unit direction drawn from `seed`."""
    E = np.asarray(embeddings, dtype=np.float32)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(E.shape).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return E + (scale * np.asarray(margin, dtype=np.float32))[:, None] * direction


def decoy_tensor(table: np.ndarray, decoy_ids: np.ndarray) -> np.ndarray:
    return np.asarray(table, dtype=np.float32)[np.asarray(decoy_ids, dtype=np.int64)]


def make_baseline(
    construction: str,
    public: DecodeTable,
    skill: Skill,
    decoy: Skill,
    out: Path,
    *,
    model: str,
    revision: str | None,
    seed: int = 0,
    offset_margins: float = OFFSET_MARGINS,
) -> dict[str, Any]:
    if construction not in BASELINES:
        raise ValueError(f"construction must be one of {BASELINES}, got {construction!r}")
    table = np.asarray(public.table, dtype=np.float32)
    ids = np.asarray(public.tokenise(skill.source), dtype=np.int64)
    if ids.size == 0:
        raise ValueError(f"{skill.name} has no tokens")
    decoy_ids = align_decoy(public.tokenise, decoy.source, int(ids.size))
    plain = table[ids]
    if construction == CONSTRUCTION_OFFSET:
        tensor = offset_tensor(plain, margins(table, ids), offset_margins, seed)
    else:
        tensor = decoy_tensor(table, decoy_ids)
    written = tensor.astype(np.float16).astype(np.float32)
    got = nearest_token(table, written)
    readout = public.detokenise(got.tolist())
    sidecar = {
        "construction": construction,
        "skill": skill.name,
        "model": model,
        "revision": revision,
        "tokens": int(ids.size),
        "width": int(table.shape[1]),
        "seed": seed,
        "decoy": decoy.name,
        "source_sha256": source_sha256(skill.source),
        "offset_margins": offset_margins if construction == CONSTRUCTION_OFFSET else None,
        "token_recovery": float((got == ids).mean()),
        "decode_cer": {
            "skill": cer(public.detokenise(ids.tolist()), readout),
            "decoy": cer(public.detokenise(decoy_ids.tolist()), readout),
        },
    }
    write_spass(Path(out), tensor, sidecar)
    write_plain(Path(out), plain)
    return sidecar


def make_many_baselines(
    pairs: Sequence[tuple[Skill, Skill]],
    seed: int,
    out: Path,
    *,
    construction: str,
    lineup: registry.Lineup | None = None,
    public: DecodeTable | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Each pair's `construction` baseline at `seed` under `out/<skill.name>`: the S-Pass maker's
    signature, so a run makes baselines where it makes S-Passes. No model is loaded."""
    from skillpass.attacks.table import decode_table_for

    lineup = lineup or registry.lineup(registry.DEFAULT_LINEUP)
    entry = lineup.authorized
    public = public or decode_table_for(entry)
    for skill, decoy in pairs:
        side = make_baseline(
            construction,
            public,
            skill,
            decoy,
            Path(out) / skill.name,
            model=entry.reader_id,
            revision=entry.revision,
            seed=seed,
        )
        log(
            f"baseline {construction}: wrote {Path(out) / skill.name}: token recovery "
            f"{side['token_recovery']:.3f}, decode CER vs skill {side['decode_cer']['skill']:.3f}"
        )
