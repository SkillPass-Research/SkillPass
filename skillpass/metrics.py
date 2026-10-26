"""The measures of Section 5.1, as pure functions over plain numbers and strings.

Utility: task success, uplift (Equation 1), the authorized model's loss rho and injection
cost iota (Equation 13), and a relative's residual uplift (Equation 14). Privacy: ROUGE-L and
the character error rate (CER) of a readout or a repetition against the skill, both over
normalised text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from statistics import mean
from typing import Any

import numpy as np

from skillpass.registry import USABLE_UPLIFT_MIN

_WHITESPACE_RUN = re.compile(r"\s+")
_WORD = re.compile(r"[^\W_]+")


def normalise(text: str) -> str:
    """NFC, every whitespace run collapsed to one space, ends stripped. No case folding."""
    return _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFC", text)).strip()


# utility


def accuracy(passed: Sequence[bool]) -> float:
    if len(passed) == 0:
        raise ValueError("accuracy of no tasks is undefined")
    return sum(1 for p in passed if p) / len(passed)


def uplift(with_skill: float, without_skill: float) -> float:
    """Delta_L(z) = success with the representation minus success with no skill (Equation 1)."""
    return with_skill - without_skill


def rho(uplift_text: float, uplift_spass: float) -> float:
    """The authorized model's utility loss, Delta(text) - Delta(S-Pass) (Equation 13)."""
    return uplift_text - uplift_spass


def iota(uplift_text: float, uplift_plain: float) -> float:
    """The injection cost, Delta(text) - Delta(plain) (Equation 13)."""
    return uplift_text - uplift_plain


def residual_uplift(
    uplift_spass: float, uplift_plain: float, floor: float = USABLE_UPLIFT_MIN
) -> float | None:
    """r_L = Delta_L(S-Pass) / Delta_L(plain), defined when the plain uplift is at least `floor`
    (Equation 14); `None` otherwise, shown as a dash in the tables."""
    if uplift_plain < floor:
        return None
    return uplift_spass / uplift_plain


def spread(values: Sequence[float | None]) -> dict[str, Any] | None:
    """The mean over seeds with the lowest and highest seed; `None` seeds are left out."""
    present = [float(v) for v in values if v is not None]
    if not present:
        return None
    return {
        "mean": mean(present),
        "min": min(present),
        "max": max(present),
        "per_seed": [None if v is None else float(v) for v in values],
    }


# text


def _levenshtein(a: str, b: str) -> int:
    """Edit distance with one numpy pass per character of `a`."""
    if not a or not b:
        return max(len(a), len(b))
    b_codes = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    positions = np.arange(len(b) + 1)
    previous = positions.copy()
    for i, ca in enumerate(a, start=1):
        cost = (b_codes != ord(ca)).astype(np.int64)
        best = np.empty(len(b) + 1, dtype=np.int64)
        best[0] = i
        best[1:] = np.minimum(previous[1:] + 1, previous[:-1] + cost)
        previous = np.minimum.accumulate(best - positions) + positions
    return int(previous[-1])


def cer(reference: str, hypothesis: str) -> float:
    """Edit distance over the reference's length, both normalised; can exceed one."""
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        raise ValueError("CER against an empty reference is undefined")
    return _levenshtein(ref, hyp) / len(ref)


def tokens(text: str) -> list[str]:
    """Words for ROUGE: normalised, lower-cased, split on anything but letters and digits."""
    return _WORD.findall(normalise(text).lower())


def _lcs_length(a: Sequence, b: Sequence) -> int:
    """Longest common subsequence, bit-parallel over `a` (Hyyro, 2004)."""
    if not a or not b:
        return 0
    masks: dict = {}
    for i, item in enumerate(a):
        masks[item] = masks.get(item, 0) | (1 << i)
    full = (1 << len(a)) - 1
    v = full
    for item in b:
        u = v & masks.get(item, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - bin(v).count("1")


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L as F1 over words: the whole text as one sequence."""
    ref, hyp = tokens(reference), tokens(hypothesis)
    if not ref:
        raise ValueError("ROUGE-L against a reference with no words is undefined")
    if not hyp:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(hyp), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
