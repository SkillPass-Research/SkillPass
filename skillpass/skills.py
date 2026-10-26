"""Skills: the synthetic suite's on-disk format, its loader and the task checkers.

A synthetic skill is a directory under `skills/synthetic/<name>/` holding `SKILL.md` (the
document, plain Markdown), `meta.yaml` (`name`, `kind`, `identifiers`: invented strings that
appear verbatim in the document) and `tasks.yaml` (a list of `{id, text, checker, expected}`,
a `regex` task also carrying an `example`). Every answer names an invented identifier, which
is what makes a task unanswerable without the document. An SOP-Bench procedure is also a
skill (`skill_from_text`), with no tasks of its own: its cases are judged by the agent loop.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from skillpass.metrics import normalise

SUITE_ROOT = Path(__file__).resolve().parents[1] / "skills" / "synthetic"
SKILL_FILE, META_FILE, TASKS_FILE = "SKILL.md", "meta.yaml", "tasks.yaml"
CHECKERS = ("exact", "regex", "contains_all")
EMPHASIS_DELIMITERS = ("**", "__", "`", "*", "_")
_FENCED = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)


class SuiteError(ValueError):
    """A skill directory the loader refuses."""


@dataclass(frozen=True)
class Task:
    id: str
    text: str
    checker: str
    expected: str | tuple[str, ...]
    example: str | None = None


@dataclass(frozen=True)
class Skill:
    """A skill document with its tasks. `transcription` is the repeat target repetitions and
    readouts are scored against: the source after `normalise`."""

    name: str
    source: str
    tasks: tuple[Task, ...] = ()
    identifiers: tuple[str, ...] = ()
    directory: Path | None = field(default=None, compare=False)

    @property
    def transcription(self) -> str:
        return normalise(self.source)

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def skill_from_text(name: str, source: str, directory: Path | None = None) -> Skill:
    return Skill(name, source, (), (), directory)


def _strip_emphasis(text: str) -> str:
    stripped = True
    while stripped:
        stripped = False
        for delimiter in EMPHASIS_DELIMITERS:
            width = len(delimiter)
            if len(text) > 2 * width and text.startswith(delimiter) and text.endswith(delimiter):
                text = text[width:-width].strip()
                stripped = True
                break
    return text


def normalise_answer(text: str) -> str:
    """An answer with its formatting taken off: an enclosing code fence, enclosing emphasis or
    backticks, one trailing full stop; then `normalise`. Content is never changed."""
    text = text.strip()
    fenced = _FENCED.fullmatch(text)
    if fenced is not None and "```" not in fenced.group(1):
        text = fenced.group(1).strip()
    text = _strip_emphasis(text)
    if text.endswith("."):
        text = _strip_emphasis(text[:-1].strip())
    return normalise(text)


def check(task: Task, answer: str) -> bool:
    """Whether `answer` passes `task`: exact match, a full regex match, or every listed string
    present, on the normalised answer."""
    answer = normalise_answer(answer)
    if task.checker == "exact":
        return answer == task.expected
    if task.checker == "regex":
        return re.fullmatch(task.expected, answer) is not None
    if task.checker == "contains_all":
        return all(part in answer for part in task.expected)
    raise ValueError(f"unknown checker {task.checker!r}")


def _parse_task(record: object, label: str) -> Task:
    if not isinstance(record, dict) or not {"id", "text", "checker", "expected"} <= set(record):
        raise SuiteError(f"{label}: a task needs id, text, checker and expected")
    checker = record["checker"]
    if checker not in CHECKERS:
        raise SuiteError(f"{label}: unknown checker {checker!r}")
    expected = record["expected"]
    if checker == "contains_all":
        if not isinstance(expected, list) or not expected:
            raise SuiteError(f"{label}: contains_all needs a non-empty list")
        expected = tuple(str(e) for e in expected)
    else:
        expected = str(expected)
    example = record.get("example")
    if checker == "regex":
        if not isinstance(example, str):
            raise SuiteError(f"{label}: a regex task needs an example")
        try:
            re.compile(expected)
        except re.error as error:
            raise SuiteError(f"{label}: the pattern does not compile: {error}") from error
    task = Task(str(record["id"]), str(record["text"]), checker, expected, example)
    if example is not None and not check(task, example):
        raise SuiteError(f"{label}: the pattern does not match its example")
    return task


def load_skill(directory: Path) -> Skill:
    """Load one skill directory, or raise `SuiteError`."""
    directory = Path(directory)
    for filename in (SKILL_FILE, META_FILE, TASKS_FILE):
        if not (directory / filename).is_file():
            raise SuiteError(f"{directory}: {filename} is missing")
    meta = yaml.safe_load((directory / META_FILE).read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("name") != directory.name:
        raise SuiteError(f"{directory}: meta.yaml must name the directory")
    source = (directory / SKILL_FILE).read_text(encoding="utf-8")
    identifiers = tuple(str(i) for i in meta.get("identifiers", []))
    missing = [i for i in identifiers if i not in source]
    if missing:
        raise SuiteError(f"{directory}: identifiers not in SKILL.md: {missing}")
    records = yaml.safe_load((directory / TASKS_FILE).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise SuiteError(f"{directory}: tasks.yaml must be a non-empty list")
    tasks = tuple(_parse_task(r, f"{directory.name} task {i}") for i, r in enumerate(records))
    if len({t.id for t in tasks}) != len(tasks):
        raise SuiteError(f"{directory}: task ids repeat")
    return Skill(directory.name, source, tasks, identifiers, directory)


def load_suite(root: Path = SUITE_ROOT) -> tuple[Skill, ...]:
    """Every skill directory under `root`, sorted by name."""
    root = Path(root)
    if not root.is_dir():
        raise SuiteError(f"{root} is not a directory")
    return tuple(
        load_skill(p) for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")
    )
