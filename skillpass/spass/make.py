"""Make one skill's S-Pass with the loaded authorized model and its relatives (Section 3.4).

`make_spass` embeds the skill and the aligned decoy through the authorized model's table, builds
the prompt prefix (the reading instruction, through the chat template) and one suffix per probe,
renders any agent conversations the S-Pass is also fitted under, does the same in each relative's
own table, runs `optimise`, and writes the tensor, the skill's own embeddings and the sidecar,
which records the settings, the final terms and the nearest-token readout of the float16 tensor
as written. `make_many` loads a lineup's models once for a batch of skills.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from skillpass import registry
from skillpass.metrics import cer
from skillpass.prompts import build_prompt, split_prompt
from skillpass.registry import ALPHA, BETA, LAMBDA, LR, MATCH_BLOCK, PROBES, STEPS, Presentation
from skillpass.skills import Skill
from skillpass.spass.decoy import decoy_ids as align_decoy
from skillpass.spass.embed import embed_skill, nearest_ids, table_of
from skillpass.spass.file import source_sha256, write_plain, write_spass
from skillpass.spass.optimise import Optimised, Relative, optimise

Conversation = Sequence[Mapping[str, str]]


@dataclass(frozen=True)
class Made:
    directory: Path
    optimised: Optimised
    sidecar: dict[str, Any]


def _pieces(reader: Any, skill_source: str, question: str) -> tuple[str, str]:
    """The templated `text` prompt's text before and after the skill, for one question."""
    prompt = build_prompt(Presentation.TEXT, question, skill_source)
    before, after = split_prompt(prompt, skill_source)
    text = reader.template(before + skill_source + after)
    if text.count(skill_source) != 1:
        raise ValueError("the chat template did not keep the skill text exactly once")
    pre, post = text.split(skill_source, 1)
    return pre, post


def _embed_text(table: Any, ids_of: Callable[[str], Sequence[int]], text: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        return (
            table[torch.tensor(list(ids_of(text)), device=table.device, dtype=torch.long)]
            .cpu()
            .numpy()
        )


def _contexts(
    reader: Any,
    table: Any,
    ids_of: Callable[[str], Sequence[int]],
    skill_source: str,
    probes: Sequence[str],
    conversations: Sequence[Conversation],
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """The shared prefix, one suffix per probe, and one `(prefix, suffix)` per conversation, in
    `reader`'s own embedding coordinates."""
    prefix = _embed_text(table, ids_of, _pieces(reader, skill_source, "")[0])
    suffixes = [_embed_text(table, ids_of, _pieces(reader, skill_source, q)[1]) for q in probes]
    contexts = []
    for messages in conversations:
        text = reader.render(messages)
        if text.count(skill_source) != 1:
            raise ValueError("a conversation must hold the skill text exactly once")
        pre, post = text.split(skill_source, 1)
        contexts.append((_embed_text(table, ids_of, pre), _embed_text(table, ids_of, post)))
    return prefix, suffixes, contexts


def trim_to_blocks(model: Any, k: int) -> None:
    """Keep the embedding table and the first `k` blocks of `model`, in place, and drop the rest:
    a relative is read only to block `k`, so this is all the transfer term needs."""
    import gc

    import torch

    inner = model.model
    lm = getattr(inner, "language_model", inner)
    if len(lm.layers) > k:
        lm.layers = torch.nn.ModuleList(list(lm.layers)[:k])
    for owner, name in (
        (model, "lm_head"),
        (inner, "visual"),
        (inner, "vision_tower"),
        (inner, "multi_modal_projector"),
        (inner, "audio_tower"),
        (inner, "embed_vision"),
        (inner, "embed_audio"),
    ):
        module = getattr(owner, name, None)
        if isinstance(module, torch.nn.Module) and not isinstance(module, torch.nn.Identity):
            setattr(owner, name, torch.nn.Identity())
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_spass(
    reader: Any,
    skill: Skill,
    decoy: Skill,
    out: Path,
    *,
    relatives: Sequence[Any] = (),
    conversations: Sequence[Conversation] = (),
    probes: Sequence[str] = PROBES,
    k: int = MATCH_BLOCK,
    steps: int = STEPS,
    lam: float = LAMBDA,
    alpha: float = ALPHA,
    beta: float = BETA,
    lr: float = LR,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> Made:
    """Make `skill`'s S-Pass under `out`. `reader` is the authorized model's wrapper and
    `relatives` the wrappers of the relatives K the transfer term is taken against, each of the
    authorized model's width; every model is loaded here if it is not, and a relative is then
    trimmed to its first `k` blocks. `conversations` are chat message lists holding the skill's
    text exactly once, rendered through each model's own chat template."""
    import torch

    if decoy.source == skill.source:
        raise ValueError("the decoy is the skill itself: no vector can be nearer a wrong token")
    reader.load()
    model, tokenizer = reader.model, reader.tokenizer
    embedding = model.get_input_embeddings()
    embedded = embed_skill(tokenizer, embedding, skill.source)
    length = int(embedded.ids.shape[0])
    ids_of = lambda text: tokenizer(text, add_special_tokens=False).input_ids  # noqa: E731
    decoy_ids = align_decoy(ids_of, decoy.source, length)
    collisions = int((decoy_ids == embedded.ids).sum())
    with torch.no_grad():
        table = table_of(embedding)
        decoys = table[torch.tensor(decoy_ids, device=table.device)].cpu().numpy()
    prefix, suffixes, contexts = _contexts(
        reader, table, ids_of, skill.source, probes, conversations
    )
    kin = []
    for other in relatives:
        other.load()
        other_embed = other.model.get_input_embeddings()
        if other_embed.weight.shape[1] != embedded.embeddings.shape[1]:
            raise ValueError(
                f"{other.entry.reader_id} has another width: it cannot consume the S-Pass"
            )
        trim_to_blocks(other.model, k)
        other_ids = lambda text, t=other.tokenizer: t(text, add_special_tokens=False).input_ids  # noqa: E731
        with torch.no_grad():
            table_o = table_of(other_embed)
            decoys_o = table_o[torch.tensor(decoy_ids, device=table_o.device)].cpu().numpy()
        prefix_o, suffixes_o, contexts_o = _contexts(
            other, table_o, other_ids, skill.source, probes, conversations
        )
        kin.append(Relative(other.model, decoys_o, prefix_o, suffixes_o, contexts_o))
    optimised = optimise(
        model,
        embedded.embeddings,
        decoys,
        embedded.margins,
        prefix,
        suffixes,
        contexts=contexts,
        relatives=kin,
        k=k,
        steps=steps,
        lam=lam,
        alpha=alpha,
        beta=beta,
        lr=lr,
        seed=seed,
        log=log,
    )
    written = optimised.vectors.astype(np.float16).astype(np.float32)
    with torch.no_grad():
        got = nearest_ids(table, torch.tensor(written, device=table.device)).cpu().tolist()
    readout = tokenizer.decode(got)
    sidecar = {
        "construction": registry.CONSTRUCTION_SPASS,
        "skill": skill.name,
        "model": reader.entry.reader_id,
        "revision": reader.entry.revision,
        "tokens": length,
        "width": int(embedded.embeddings.shape[1]),
        "seed": seed,
        "decoy": decoy.name,
        "decoy_collisions": collisions,
        "source_sha256": source_sha256(skill.source),
        "k": k,
        "steps": steps,
        "lambda": lam,
        "alpha": alpha,
        "beta": beta,
        "lr": lr,
        "probes": list(probes),
        "conversations": len(conversations),
        "relatives": [r.entry.reader_id for r in relatives],
        "util": optimised.util,
        "recon": optimised.recon,
        "trans": list(optimised.trans),
        "context_util": list(optimised.context_util),
        "decode_cer": {
            "skill": cer(tokenizer.decode(embedded.ids.tolist()), readout),
            "decoy": cer(tokenizer.decode(decoy_ids.tolist()), readout),
        },
        "history": [s.__dict__ for s in optimised.history],
    }
    write_spass(Path(out), optimised.vectors, sidecar)
    write_plain(Path(out), embedded.embeddings)
    return Made(Path(out), optimised, sidecar)


def make_many(
    pairs: Sequence[tuple[Skill, Skill]],
    seed: int,
    out: Path,
    *,
    lineup: registry.Lineup | None = None,
    conversations: Mapping[str, Sequence[Conversation]] | None = None,
    first_blocks_only: bool = True,
    log: Callable[[str], None] = print,
    **settings: Any,
) -> None:
    """Each `(skill, decoy)` pair's S-Pass at `seed` under `out/<skill.name>`, with the lineup's
    authorized model and every relative in K loaded once and released after. With
    `first_blocks_only`, every model is loaded with its table and first `k` blocks only.
    `conversations` maps a skill's name to the agent conversations its S-Pass is fitted under."""
    from skillpass.models import make_reader

    lineup = lineup or registry.lineup(registry.DEFAULT_LINEUP)
    k = settings.get("k", MATCH_BLOCK)
    blocks = k if first_blocks_only else None
    authorized = make_reader(lineup.authorized, blocks=blocks)
    relatives = [make_reader(r, blocks=blocks) for r in lineup.relatives]
    try:
        for skill, decoy in pairs:
            made = make_spass(
                authorized,
                skill,
                decoy,
                Path(out) / skill.name,
                relatives=relatives,
                conversations=(conversations or {}).get(skill.name, ()),
                seed=seed,
                log=lambda m, s=skill.name: log(f"spass {s}: {m}"),
                **settings,
            )
            log(
                f"spass: wrote {made.directory}: util {made.optimised.util:.4f}, decode CER "
                f"{made.sidecar['decode_cer']['skill']:.2f} vs skill, "
                f"{made.sidecar['decode_cer']['decoy']:.2f} vs decoy"
            )
    finally:
        authorized.close()
        for r in relatives:
            r.close()
