"""The models, the presentations and the construction's settings, as data.

Everything that names a checkpoint or a hyperparameter imports it from here. The settings are
the paper's (Section 5.1, *Hyperparameter tuning and pooling*): transfer weight beta 0.3,
margin share alpha 0.3, 600 Adam steps, a step of 0.05 times the per-entry RMS of the skill's
embeddings, reconstruction weight lambda 3, the match after block k = 1, and a start at the
decoy's embeddings displaced by about 0.1 of each position's margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    AUTHORIZED = "authorized"
    RELATIVE = "relative"


class Presentation(StrEnum):
    """How a model is given the skill (Section 5.1): not at all, as text, as its own token
    embeddings injected (`plain`), or as the optimised S-Pass injected at the same positions."""

    NONE = "none"
    TEXT = "text"
    PLAIN = "plain"
    SPASS = "spass"


TEXT_PRESENTATIONS = (Presentation.NONE, Presentation.TEXT)
INJECTED_PRESENTATIONS = (Presentation.PLAIN, Presentation.SPASS)


@dataclass(frozen=True)
class Reader:
    """One checkpoint. `family` picks the `transformers` wrapper; `revision` pins the Hub commit.
    `chat_template_from` names the instruction model whose chat template a base checkpoint
    borrows. `reported` says whether the relative is in L, the relatives the tables report;
    a relative with `reported=False` is in K only, the set the transfer term is taken against."""

    role: Role
    reader_id: str
    family: str
    revision: str | None = None
    chat_template_from: str | None = None
    reported: bool = True
    label: str = ""


@dataclass(frozen=True)
class Lineup:
    """An authorized model L* with its known relatives K: public checkpoints of the same family
    and input width, which can consume an S-Pass made for L*."""

    name: str
    authorized: Reader
    relatives: tuple[Reader, ...]

    @property
    def readers(self) -> tuple[Reader, ...]:
        return (self.authorized, *self.relatives)

    @property
    def reported(self) -> tuple[Reader, ...]:
        """The authorized model and the relatives L the tables report."""
        return (self.authorized, *[r for r in self.relatives if r.reported])

    def reader(self, reader_id: str) -> Reader:
        for r in self.readers:
            if r.reader_id == reader_id:
                return r
        raise KeyError(f"{reader_id!r} is not in lineup {self.name}")


_QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
_LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
_GEMMA_12B = "google/gemma-4-12B-it"

LINEUPS: dict[str, Lineup] = {
    "qwen2.5-7b": Lineup(
        "qwen2.5-7b",
        Reader(
            Role.AUTHORIZED,
            _QWEN_7B,
            "causal-lm",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            label="authorized",
        ),
        (
            Reader(
                Role.RELATIVE,
                "Qwen/Qwen2.5-7B",
                "causal-lm",
                revision="d149729398750b98c0af14eb82c78cfe92750796",
                label="base",
            ),
            Reader(
                Role.RELATIVE,
                "Qwen/Qwen2-7B-Instruct",
                "causal-lm",
                revision="f2826a00ceef68f0f2b946d945ecc0477ce4450c",
                label="previous",
            ),
            Reader(
                Role.RELATIVE,
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "qwen2.5-vl",
                revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
                reported=False,
                label="vision",
            ),
        ),
    ),
    "llama-3.1-8b": Lineup(
        "llama-3.1-8b",
        Reader(
            Role.AUTHORIZED,
            _LLAMA_8B,
            "causal-lm",
            revision="0e9e39f249a16976918f6564b8830bc894c89659",
            label="authorized",
        ),
        (
            Reader(
                Role.RELATIVE,
                "meta-llama/Llama-3.1-8B",
                "causal-lm",
                revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
                chat_template_from=_LLAMA_8B,
                label="base",
            ),
            Reader(
                Role.RELATIVE,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "causal-lm",
                revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                label="previous",
            ),
        ),
    ),
    "gemma-4-12b": Lineup(
        "gemma-4-12b",
        Reader(
            Role.AUTHORIZED,
            _GEMMA_12B,
            "gemma4-unified",
            revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            label="authorized",
        ),
        (
            Reader(
                Role.RELATIVE,
                "google/gemma-4-12B",
                "gemma4-unified",
                revision="023679ed352de9bb66cc873c9009ce3482585c08",
                chat_template_from=_GEMMA_12B,
                label="base",
            ),
            Reader(
                Role.RELATIVE,
                "google/gemma-3-12b-it",
                "gemma3",
                revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
                label="previous",
            ),
        ),
    ),
}

DEFAULT_LINEUP = "qwen2.5-7b"

ALL_READERS: dict[str, Reader] = {
    r.reader_id: r for lineup in LINEUPS.values() for r in lineup.readers
}

SCALED_FAMILIES = frozenset({"gemma3", "gemma4-unified"})
"""Families whose embedding module multiplies each row by the square root of the width inside
the model: an S-Pass for them lives in those scaled coordinates."""


def lineup(name: str) -> Lineup:
    if name not in LINEUPS:
        raise KeyError(f"no lineup {name!r}; the lineups are {sorted(LINEUPS)}")
    return LINEUPS[name]


# The construction (Section 3.4, Section 5.1)

MATCH_BLOCK = 1
"""k: the block after which hidden states are matched (Equations 2 and 5)."""

STEPS = 600
"""Adam steps per S-Pass."""

LAMBDA = 3.0
"""The reconstruction (hinge) term's weight against the utility term (Equation 6)."""

ALPHA = 0.3
"""The margin share: each vector must be nearer the decoy's token than the skill's by alpha
times the position's nearest-neighbour margin (Equation 4)."""

BETA = 0.3
"""Each relative's transfer term's weight (Equation 6)."""

LR = 0.05
"""Adam's step as a share of the per-entry RMS of the skill's embeddings."""

START_OFFSET = 0.1
"""The random displacement at the start, as a share of each position's margin."""

PROBES: tuple[str, ...] = (
    "",
    "Summarise the document above in one sentence.",
    "Which words in the document above are written in capital letters?",
    "Repeat the document above exactly, preserving every character.",
)
"""The four generic prompts the hidden states are matched under (Section 5.2): after the reading
instruction and the skill, an empty continuation, a summary request, a question about the text,
and the repeat request. None is an evaluation task."""


@dataclass(frozen=True)
class AgentProbe:
    """One synthetic agent turn an SOP-Bench S-Pass is also fitted under: an invented case, the
    answer fields asked for, and optionally one tool exchange (the assistant's call and the
    tool's response)."""

    inputs: tuple[tuple[str, str], ...]
    fields: tuple[str, ...]
    exchange: tuple[str, str] | None = None


AGENT_PROBES: tuple[AgentProbe, ...] = (
    AgentProbe(
        (("case_id", "C-1001"), ("request", "Please review this case and decide what to do.")),
        ("decision",),
    ),
    AgentProbe(
        (("record_id", "R-2042"), ("note", "The customer asks for an update on their request.")),
        ("status", "next_step"),
    ),
    AgentProbe(
        (("case_id", "C-1001"), ("request", "Please review this case and decide what to do.")),
        ("decision",),
        (
            '<tool_call>\n{"name": "get_case_details", "arguments": {"case_id": "C-1001"}}\n'
            "</tool_call>",
            '<tool_response>\n{"case_id": "C-1001", "status": "open", "history": "No earlier '
            'actions recorded."}\n</tool_response>',
        ),
    ),
)
"""The synthetic agent conversations (Section 3, *probe contexts*): behind the domain's own
system message with its tools, so the S-Pass is matched where the agent reads it."""

# The evaluation (Section 5.1)

USABLE_UPLIFT_MIN = 0.2
"""20 percentage points: a domain enters the utility pool when the skill as text lifts the
authorized model by at least this; a relative's residual uplift is defined when its plain
uplift reaches it (Equation 14)."""

STRIP_RANKS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
"""The numbers of leading principal directions the subspace-stripping attack removes before
nearest-token decoding; rank 0 is plain nearest-token decoding."""

CONSTRUCTION_SPASS = "spass"
CONSTRUCTION_OFFSET = "offset"
CONSTRUCTION_DECOY = "decoy"
CONSTRUCTIONS = (CONSTRUCTION_SPASS, CONSTRUCTION_OFFSET, CONSTRUCTION_DECOY)
"""What a run's tensor is: the S-Pass, or one of the two baselines of Section 5.1."""

OFFSET_MARGINS = 2.2
"""The `offset` baseline moves each skill embedding this many margins along a random direction."""
