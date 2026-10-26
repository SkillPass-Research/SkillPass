"""The terms of the objective (Section 3), in `numpy` where no model is needed.

The reconstruction term (Equation 4) is a hinge per position: zero once the vector is nearer the
decoy's token than the skill's by alpha times the position's margin, normalised by the margin.
The utility and transfer terms (Equations 2 and 5) are relative squared errors between hidden
states; `relative_error` is their shape, and `optimise` computes them on the model.
"""

from __future__ import annotations

import numpy as np


def hinge(
    vectors: np.ndarray,
    originals: np.ndarray,
    decoys: np.ndarray,
    margins: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per position: `max(0, ||z - e_decoy|| - ||z - e_skill|| + alpha * mu) / mu`."""
    to_decoy = np.linalg.norm(vectors - decoys, axis=1)
    to_skill = np.linalg.norm(vectors - originals, axis=1)
    return np.maximum(0.0, to_decoy - to_skill + alpha * margins) / margins


def reconstruction_loss(
    vectors: np.ndarray,
    originals: np.ndarray,
    decoys: np.ndarray,
    margins: np.ndarray,
    alpha: float,
) -> float:
    """L_recon: the mean of the hinge over positions (Equation 4)."""
    return float(hinge(vectors, originals, decoys, margins, alpha).mean())


def relative_error(hidden: np.ndarray, target: np.ndarray) -> float:
    """`||hidden - target||_F^2 / ||target||_F^2`, the shape of Equations 2 and 5."""
    hidden, target = np.asarray(hidden, dtype=np.float64), np.asarray(target, dtype=np.float64)
    return float(((hidden - target) ** 2).sum() / (target**2).sum())
