"""One SOP-Bench domain: its procedure, its tools, its cases, and how an answer is checked.

A domain is a directory of the benchmark's data (`src/amazon_sop_bench/benchmarks/data/<name>/`
in its repository) holding `sop.txt`, `toolspecs.json`, `tools.py`, `metadata.json` and
`test_set_with_outputs.csv`. The data is not in this repository: a checkout at the pinned commit
is read in place. The procedure is the skill; the case arrives as text.

The tools are the domain's own Python, loaded as SOP-Bench loads them. Arguments a tool's
description advertises but its function does not take are dropped and counted. A domain's
inputs are the columns its `metadata.json` lists; where it lists none, the columns of the
benchmark's answer-free test file, or every column but the outputs. An output column is never
an input.

The agent is asked for its answer in `<final_output>` tags; some procedures prescribe their own
tag (`<final_decision>`, `<final_response>`), and a model following the procedure uses it, so an
answer is read from the last of any of these tags.
"""

from __future__ import annotations

import collections
import csv
import importlib.util
import inspect
import json
import logging
import random
import re
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from skillpass.skills import Skill, normalise_answer, skill_from_text

DATA_SUBDIR = Path("src/amazon_sop_bench/benchmarks/data")
DOMAIN_FILES = (
    "sop.txt",
    "toolspecs.json",
    "tools.py",
    "metadata.json",
    "test_set_with_outputs.csv",
)
ANSWER_FREE_FILE = "test_set_without_outputs.csv"

TOOL_CALL_TAG = "<tool_call>"
TOOL_RESPONSE_TAG = "<tool_response>"
ANSWER_TAG = "final_output"
_PRESCRIBED_TAG = re.compile(r"<(final_[a-z_]+)>")
_DECODER = json.JSONDecoder()


class DomainError(ValueError):
    """A domain directory that cannot be read as one."""


@dataclass(frozen=True)
class Task:
    """One test case: its inputs, and the fields and values the answer must carry."""

    id: str
    inputs: dict[str, str]
    expected: dict[str, str]


@dataclass
class Domain:
    name: str
    path: Path
    sop: str
    tools: list[dict[str, Any]]
    input_columns: list[str]
    output_columns: list[str]
    rows: list[dict[str, str]] = field(repr=False)
    input_source: str = "metadata"
    excluded_inputs: tuple[str, ...] = ()

    @property
    def answer_tags(self) -> tuple[str, ...]:
        """`final_output`, then every answer tag the procedure prescribes."""
        return tuple(dict.fromkeys([ANSWER_TAG, *_PRESCRIBED_TAG.findall(self.sop)]))

    def skill(self) -> Skill:
        return skill_from_text(skill_name(self.name), self.sop, self.path)

    def tasks(self, limit: int | None = None, seed: int = 0) -> list[Task]:
        """Every case in file order, or `limit` of them drawn round-robin over the expected
        outcomes so each outcome is represented."""
        indexed = list(enumerate(self.rows))
        if limit is not None and limit < len(indexed):
            by_outcome = collections.defaultdict(list)
            for i, row in indexed:
                by_outcome[tuple(row[c] for c in self.output_columns)].append((i, row))
            rng = random.Random(seed)
            for pool in by_outcome.values():
                rng.shuffle(pool)
            picked = []
            while len(picked) < limit:
                for outcome in sorted(by_outcome):
                    if by_outcome[outcome] and len(picked) < limit:
                        picked.append(by_outcome[outcome].pop())
            indexed = sorted(picked)
        return [
            Task(
                id=f"{self.name}-{i}",
                inputs={c: row[c] for c in self.input_columns},
                expected={c: row[c] for c in self.output_columns},
            )
            for i, row in indexed
        ]


def skill_name(domain: str) -> str:
    return "sop-" + domain.replace("_", "-")


def list_domains(data_dir: Path) -> list[str]:
    return sorted(
        p.name
        for p in Path(data_dir).iterdir()
        if p.is_dir() and all((p / f).is_file() for f in DOMAIN_FILES)
    )


def load_domain(data_dir: Path, name: str, exclude: Sequence[str] = ()) -> Domain:
    path = Path(data_dir) / name
    missing = [f for f in DOMAIN_FILES if not (path / f).is_file()]
    if missing:
        raise DomainError(f"{path} lacks {', '.join(missing)}")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    outputs = meta.get("output_columns")
    if not isinstance(outputs, list) or not outputs:
        raise DomainError(f"{path / 'metadata.json'} has no output_columns")
    with (path / "test_set_with_outputs.csv").open(
        newline="", encoding="utf-8", errors="replace"
    ) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    inputs, source = _input_columns(path, meta, columns, outputs)
    unknown = [c for c in exclude if c not in inputs]
    if unknown:
        raise DomainError(f"{name}: cannot exclude {unknown}, which are not its input columns")
    inputs = [c for c in inputs if c not in exclude]
    if not inputs:
        raise DomainError(f"{name}: the exclusions leave no input columns")
    absent = [c for c in inputs + outputs if c not in columns]
    if absent:
        raise DomainError(f"{path} test set lacks the columns {absent}")
    tools = [
        {
            "name": spec["toolSpec"]["name"],
            "description": spec["toolSpec"].get("description", ""),
            "parameters": spec["toolSpec"]["inputSchema"]["json"],
        }
        for spec in json.loads((path / "toolspecs.json").read_text(encoding="utf-8"))
    ]
    return Domain(
        name=name,
        path=path,
        sop=(path / "sop.txt").read_text(encoding="utf-8"),
        tools=tools,
        input_columns=inputs,
        output_columns=list(outputs),
        rows=rows,
        input_source=source,
        excluded_inputs=tuple(exclude),
    )


def _input_columns(
    path: Path, meta: dict, columns: list[str], outputs: list[str]
) -> tuple[list[str], str]:
    listed = meta.get("input_columns")
    if listed:
        inputs, source = list(listed), "metadata"
    elif (path / ANSWER_FREE_FILE).is_file():
        with (path / ANSWER_FREE_FILE).open(newline="", encoding="utf-8", errors="replace") as fh:
            inputs = [c for c in next(csv.reader(fh), []) if c]
        source = "test set without outputs"
    else:
        inputs, source = [c for c in columns if c not in outputs], "every column but the outputs"
    leaked = [c for c in inputs if c in outputs]
    if leaked:
        raise DomainError(f"{path}: the output columns {leaked} would be shown as inputs")
    if not inputs:
        raise DomainError(f"{path}: no input columns")
    return inputs, source


# the tools


def _provide_logging_helper() -> None:
    for name in ("amazon_sop_bench", "amazon_sop_bench.utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "amazon_sop_bench.utils.logging" not in sys.modules:
        helper = types.ModuleType("amazon_sop_bench.utils.logging")
        helper.get_logger = logging.getLogger
        sys.modules["amazon_sop_bench.utils.logging"] = helper


class ToolBox:
    """A domain's tool class, called with the arguments its methods take."""

    def __init__(self, domain: Domain) -> None:
        _provide_logging_helper()
        module_name = f"sopbench_tools_{domain.name}"
        spec = importlib.util.spec_from_file_location(module_name, domain.path / "tools.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes = [
            v
            for v in vars(module).values()
            if isinstance(v, type)
            and v.__module__ == module_name
            and hasattr(v, "process_tool_call")
        ]
        if len(classes) != 1:
            raise DomainError(
                f"{domain.path / 'tools.py'} has {len(classes)} tool classes, not one"
            )
        self.manager = classes[0]()
        self.dropped: collections.Counter = collections.Counter()

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        method = getattr(self.manager, name, None)
        if method is not None:
            accepted = set(inspect.signature(method).parameters)
            for key in [k for k in arguments if k not in accepted]:
                self.dropped[f"{name}.{key}"] += 1
            arguments = {k: v for k, v in arguments.items() if k in accepted}
        return self.manager.process_tool_call(name, arguments)


# reading the agent's replies


def tool_calls(reply: str) -> list[dict[str, Any]]:
    """Every call in a reply: the first JSON object after each `<tool_call>` tag, whether or not
    the tag was closed. A call that is not valid JSON is `{"_unparsed": ...}`."""
    calls = []
    for chunk in reply.split(TOOL_CALL_TAG)[1:]:
        start = chunk.find("{")
        if start < 0:
            continue
        try:
            obj, _ = _DECODER.raw_decode(chunk[start:])
        except json.JSONDecodeError:
            calls.append({"_unparsed": chunk[:200]})
            continue
        calls.append(obj if isinstance(obj, dict) else {"_unparsed": chunk[:200]})
    return calls


def task_text(task: Task) -> str:
    """SOP-Bench's case text: the inputs and the answer's format, which names the output fields
    but never their values."""
    fields = "\n".join(f"{k}: {v}" for k, v in task.inputs.items())
    template = json.dumps(dict.fromkeys(task.expected, "..."))
    return (
        "Process this case, using the available tools to gather what you need.\n\n"
        f"{fields}\n\nWhen you have finished, give your final answer as a JSON object with the "
        f"fields {', '.join(task.expected)}, inside <final_output></final_output> tags, for "
        f"example <final_output>{template}</final_output>."
    )


def final_answer(
    reply: str, tags: Sequence[str] = (ANSWER_TAG,), fields: Sequence[str] = ()
) -> dict[str, Any] | None:
    """The reply's last answer under any of `tags`: a JSON object between the tags or after the
    tag's name and `=`, or a bare value between the tags where the case has one output field."""
    names = "|".join(re.escape(t) for t in tags)
    found: list[tuple[int, dict[str, Any] | None]] = []
    for match in re.finditer(r"<(" + names + r")>(.*?)</\1>", reply, re.S):
        found.append((match.start(), _answer_in(match.group(2).strip(), fields)))
    for match in re.finditer(r"(?<![<\w/])(?:" + names + r")\s*=\s*(?=\{)", reply):
        try:
            obj, _ = _DECODER.raw_decode(reply[match.end() :])
        except json.JSONDecodeError:
            obj = None
        found.append((match.start(), obj if isinstance(obj, dict) else None))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def _answer_in(body: str, fields: Sequence[str]) -> dict[str, Any] | None:
    try:
        obj, _ = _DECODER.raw_decode(body)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        return obj
    if len(fields) == 1 and body and not isinstance(obj, list):
        return {fields[0]: body}
    return None


def answer_key(value: Any) -> str:
    return normalise_answer(str(value)).casefold()


def strict_match(answer: dict[str, Any] | None, expected: dict[str, str]) -> bool:
    """Success: every expected field present and equal to its value after the answer
    normaliser, ignoring case."""
    if not answer:
        return False
    return all(k in answer and answer_key(answer[k]) == answer_key(v) for k, v in expected.items())


def lenient_match(answer: dict[str, Any] | None, expected: dict[str, str]) -> bool:
    """SOP-Bench's own looser rule, recorded beside success and never used as it."""
    if not answer:
        return False

    def one(a: Any, e: Any) -> bool:
        a = str(a).lower().strip().replace("_", " ").replace("-", " ")
        e = str(e).lower().strip().replace("_", " ").replace("-", " ")
        return bool(a) and (a == e or a in e or e in a)

    return all(k in answer and one(answer[k], v) for k, v in expected.items())
