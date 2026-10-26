"""Adam on the S-Pass against Equation 6, with every model frozen (Section 3.4).

    L(z) = L_util(z) + lambda * L_recon(z) + beta * sum_{L in K} L_trans^L(z)

`L_util` is the relative squared error between the authorized model's hidden states after block
`k` on the S-Pass and on the skill, from the skill's first position to the end of the prompt,
averaged over the prompt contexts (Equation 2). `L_recon` is the hinge (Equation 4). Each
relative's `L_trans` is the same relative error between its hidden states on the S-Pass and on
its own embeddings of the decoy (Equation 5). The S-Pass starts at the decoy's embeddings plus
a random displacement of `START_OFFSET` margins, and Adam's step is `lr` times the per-entry
RMS of the skill's embeddings.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from skillpass.registry import ALPHA, BETA, LAMBDA, LR, MATCH_BLOCK, START_OFFSET, STEPS

LOG_EVERY = 25


@dataclass(frozen=True)
class Step:
    step: int
    util: float
    recon: float
    trans: tuple[float, ...]


@dataclass(frozen=True)
class Optimised:
    """The S-Pass `(m, d)` float32, the history, and the final terms."""

    vectors: np.ndarray
    history: tuple[Step, ...]
    util: float
    recon: float
    trans: tuple[float, ...]
    """Per relative, the final transfer term: zero would mean it reads the decoy exactly."""
    context_util: tuple[float, ...]
    """The final utility term under each context, the probes' first, then the conversations'."""


@dataclass(frozen=True)
class Relative:
    """A known relative L in K: its loaded model and, in its own embedding table, the decoy, the
    prompt prefix, each probe suffix and any further `(prefix, suffix)` contexts."""

    model: Any
    decoys: np.ndarray
    prefix: np.ndarray
    suffixes: Sequence[np.ndarray]
    contexts: Sequence[tuple[np.ndarray, np.ndarray]] = ()


class Blocks:
    """A language model's blocks, truncated to `k` for a forward pass and restored after."""

    def __init__(self, model: Any) -> None:
        import torch

        inner = model.model
        self.lm = getattr(inner, "language_model", inner)
        self.all = list(self.lm.layers)
        self.torch = torch
        self.device = model.device
        self.dtype = next(self.lm.parameters()).dtype

    def hidden(self, embeds: Any, k: int) -> Any:
        """The residual stream after block `k` for `(1, n, d)` embeddings, in float32, caught at
        block `k`'s own output: the truncated model applies its final norm after its last block,
        which block `k + 1` of the full model never sees."""
        torch = self.torch
        if k < 1 or k > len(self.all):
            raise ValueError(f"k must be in 1..{len(self.all)}, got {k}")
        caught: list[Any] = []

        def catch(_module: Any, _args: Any, output: Any) -> None:
            caught.append(output[0] if isinstance(output, tuple) else output)

        handle = self.all[k - 1].register_forward_hook(catch)
        self.lm.layers = torch.nn.ModuleList(self.all[:k])
        try:
            self.lm(inputs_embeds=embeds.to(self.dtype), use_cache=False)
        finally:
            handle.remove()
            self.lm.layers = torch.nn.ModuleList(self.all)
        return caught[-1].float()


def _contexts(torch: Any, prefix, suffixes, contexts, device) -> list[tuple[Any, Any]]:
    def t(x: np.ndarray) -> Any:
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=device)

    shared = t(prefix)
    return [(shared, t(s)) for s in suffixes] + [(t(p), t(s)) for p, s in contexts]


def _match(blocks: Blocks, span: Any, pairs, targets, k: int) -> list[Any]:
    """The relative squared error at the span's and suffix's positions, one per context."""
    terms = []
    for (P, s), target in zip(pairs, targets, strict=True):
        full = blocks.torch.cat([P, span.to(blocks.device), s], dim=0)[None]
        h = blocks.hidden(full, k)
        keep = slice(P.shape[0], full.shape[1])
        terms.append(((h[0, keep] - target[0, keep]) ** 2).sum() / (target[0, keep] ** 2).sum())
    return terms


def optimise(
    model: Any,
    originals: np.ndarray,
    decoys: np.ndarray,
    margins: np.ndarray,
    prefix: np.ndarray,
    suffixes: Sequence[np.ndarray],
    *,
    contexts: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    relatives: Sequence[Relative] = (),
    k: int = MATCH_BLOCK,
    steps: int = STEPS,
    lam: float = LAMBDA,
    alpha: float = ALPHA,
    beta: float = BETA,
    lr: float = LR,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> Optimised:
    """The S-Pass for one skill. `originals` (e*) and `decoys` are `(m, d)`, `margins` `(m,)`,
    `prefix` `(p, d)` and each suffix `(q_i, d)`, all the authorized model's embeddings in
    float32; `contexts` are further `(prefix, suffix)` pairs, the agent conversations. Every
    model is frozen. Deterministic for `seed` on one machine."""
    import torch

    blocks = Blocks(model)
    device = blocks.device
    E = torch.tensor(originals, dtype=torch.float32, device=device)
    T = torch.tensor(decoys, dtype=torch.float32, device=device)
    M = torch.tensor(margins, dtype=torch.float32, device=device)
    pairs = _contexts(torch, prefix, suffixes, contexts, device)
    if not pairs:
        raise ValueError("optimise needs at least one prompt context")
    with torch.no_grad():
        targets = [blocks.hidden(torch.cat([P, E, s], dim=0)[None], k) for P, s in pairs]
    kin = []
    for relative in relatives:
        rb = Blocks(relative.model)
        rT = torch.tensor(relative.decoys, dtype=torch.float32, device=rb.device)
        r_pairs = _contexts(torch, relative.prefix, relative.suffixes, relative.contexts, rb.device)
        with torch.no_grad():
            r_targets = [rb.hidden(torch.cat([P, rT, s], dim=0)[None], k) for P, s in r_pairs]
        kin.append((rb, r_pairs, r_targets))

    generator = torch.Generator(device=device).manual_seed(seed)
    direction = torch.randn(E.shape, generator=generator, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True)
    z = (T + direction * (START_OFFSET * M)[:, None]).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr * float(E.pow(2).mean().sqrt()))

    history: list[Step] = []
    util = recon = torch.zeros((), device=device)
    per_context: list[Any] = []
    trans: list[Any] = []
    for step in range(steps + 1):
        per_context = _match(blocks, z, pairs, targets, k)
        util = sum(per_context) / len(per_context)
        trans = [
            sum(_match(rb, z, r_pairs, r_targets, k)) / len(r_pairs)
            for rb, r_pairs, r_targets in kin
        ]
        d_skill = (z - E).norm(dim=1)
        d_decoy = (z - T).norm(dim=1)
        recon = (torch.clamp(d_decoy - d_skill + alpha * M, min=0.0) / M).mean()
        loss = util + lam * recon
        for term in trans:
            loss = loss + beta * term.to(device)
        if step % LOG_EVERY == 0 or step == steps:
            record = Step(
                step,
                float(util.detach()),
                float(recon.detach()),
                tuple(float(t.detach()) for t in trans),
            )
            history.append(record)
            if log is not None:
                kin_text = "".join(f" trans {t:.4f}" for t in record.trans)
                log(f"step {step}: util {record.util:.4f} recon {record.recon:.4f}{kin_text}")
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return Optimised(
        vectors=z.detach().cpu().numpy().astype(np.float32),
        history=tuple(history),
        util=float(util.detach()),
        recon=float(recon.detach()),
        trans=tuple(float(t.detach()) for t in trans),
        context_util=tuple(float(t.detach()) for t in per_context),
    )
