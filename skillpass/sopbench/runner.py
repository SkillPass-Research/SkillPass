"""The SOP-Bench run: the S-Passes, the episodes and the report, each a phase that resumes.

    passes    one S-Pass per domain and seed (or a baseline's tensor), fitted under the generic
              probes and the domain's agent conversations, turned against every relative in K
    episodes  every reader, every domain: `none`, `text` and `plain` once, `spass` once per seed,
              each case an agent episode; then the repeat from `plain` and from each S-Pass
    report    each S-Pass decoded with the public table, then `report.json` and `table.md`

A phase can be limited to some domains and readers, so several jobs, one a GPU, run side by
side over the same `--out`; every phase skips what the directory holds, and jobs sharing a
directory take its lock to write. Evidence goes to `--out` (`manifest.json`, `answers.jsonl`
with every trajectory, each sidecar, each decode, the report); tensors go to `--passes`.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import json
import re
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from skillpass import registry
from skillpass.models import make_reader
from skillpass.prompts import REPEAT_MAX_NEW_TOKENS, REPEAT_PROMPT, build_prompt
from skillpass.registry import Presentation
from skillpass.sopbench.agent import (
    BATCH,
    MAX_TURNS,
    TURN_TOKENS,
    agent_conversations,
    run_episodes,
)
from skillpass.sopbench.domain import (
    DATA_SUBDIR,
    Domain,
    DomainError,
    ToolBox,
    list_domains,
    load_domain,
    skill_name,
)
from skillpass.spass.decoy import decoy_ids as align_decoy
from skillpass.spass.file import (
    PLAIN_FILE,
    SIDECAR_FILE,
    SPASS_FILE,
    file_sha256,
    read_plain,
    read_spass,
)

REPOSITORY = "https://github.com/amazon-science/sop-bench"
LOG_FILE, MANIFEST_FILE, LOCK_FILE = "answers.jsonl", "manifest.json", ".lock"
CONFIG_FIELDS = {
    "sopbench_dir",
    "sopbench_commit",
    "lineup",
    "domains",
    "exclude",
    "decoys",
    "input_exclusions",
    "construction",
    "seeds",
    "tasks_per_domain",
    "readers",
    "max_turns",
    "turn_tokens",
    "batch",
}
PHASES = ("all", "passes", "episodes", "report")
JOB_ANSWER, JOB_REPEAT = "answer", "repeat"
SEEDLESS = (Presentation.NONE, Presentation.TEXT, Presentation.PLAIN)

PassMaker = Callable[[Sequence[tuple[Any, Any]], int, Path], None]


class SopBenchError(ValueError):
    """A config, a data checkout or an output directory the run refuses."""


@dataclass
class SopBenchConfig:
    path: Path
    sha256: str
    sopbench_dir: Path
    sopbench_commit: str
    lineup: registry.Lineup
    domains: list[str]
    decoys: dict[str, str]
    seeds: tuple[int, ...]
    tasks_per_domain: int | None
    readers: tuple[registry.Reader, ...]
    construction: str = registry.CONSTRUCTION_SPASS
    max_turns: int = MAX_TURNS
    turn_tokens: int = TURN_TOKENS
    batch: int = BATCH
    input_exclusions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.sopbench_dir / DATA_SUBDIR

    @property
    def authorized(self) -> registry.Reader:
        return self.lineup.authorized

    def domain(self, name: str) -> Domain:
        try:
            return load_domain(self.data_dir, name, self.input_exclusions.get(name, ()))
        except DomainError as error:
            raise SopBenchError(str(error)) from error


def procedure_family(domain: str) -> str:
    """A domain's name without a version suffix: `_v1` and `_v2` are one procedure, and one must
    never be the other's decoy, since its text would disclose the procedure."""
    return re.sub(r"_v\d+$", "", domain)


def default_decoys(domains: Sequence[str]) -> dict[str, str]:
    """Each domain's decoy where the config names none: the next domain in the list whose
    procedure family differs."""
    out = {}
    for i, d in enumerate(domains):
        for step in range(1, len(domains)):
            candidate = domains[(i + step) % len(domains)]
            if procedure_family(candidate) != procedure_family(d):
                out[d] = candidate
                break
    return out


def _positive(data: Mapping, key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SopBenchError(f"{key} must be a positive integer, got {value!r}")
    return value


def parse_config(data: Any, path: Path, sha256: str) -> SopBenchConfig:
    if not isinstance(data, dict):
        raise SopBenchError("the config must be a mapping")
    unknown = sorted(set(data) - CONFIG_FIELDS)
    if unknown:
        raise SopBenchError(f"unknown field {unknown[0]!r}; the fields are {sorted(CONFIG_FIELDS)}")
    for key in ("sopbench_dir", "sopbench_commit", "seeds"):
        if key not in data:
            raise SopBenchError(f"field {key!r} is required")
    sopbench_dir = Path(str(data["sopbench_dir"])).expanduser()
    commit = data["sopbench_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise SopBenchError("sopbench_commit must be the full 40-character commit")
    seeds = data["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise SopBenchError("seeds must be a non-empty list of distinct non-negative integers")
    data_dir = sopbench_dir / DATA_SUBDIR
    if not data_dir.is_dir():
        raise SopBenchError(f"{data_dir} does not exist; fetch the data with --fetch")
    available = list_domains(data_dir)
    wanted = data.get("domains", "all")
    if wanted == "all":
        domains = list(available)
    elif isinstance(wanted, list) and wanted and all(isinstance(d, str) for d in wanted):
        absent = [d for d in wanted if d not in available]
        if absent:
            raise SopBenchError(f"domains: no such domain {absent}")
        domains = list(wanted)
    else:
        raise SopBenchError("domains must be 'all' or a list of domain names")
    exclude = data.get("exclude") or []
    if not isinstance(exclude, list) or any(d not in available for d in exclude):
        raise SopBenchError(f"exclude must list domains of the checkout, got {exclude!r}")
    domains = [d for d in domains if d not in exclude]
    if len(domains) < 2:
        raise SopBenchError("at least two domains are needed, so each can be another's decoy")
    decoys = dict(data.get("decoys") or {})
    for domain, decoy in default_decoys(domains).items():
        decoys.setdefault(domain, decoy)
    for domain, decoy in decoys.items():
        if decoy not in available:
            raise SopBenchError(f"decoys: {domain}'s decoy {decoy!r} is not a domain")
        if decoy == domain or procedure_family(decoy) == procedure_family(domain):
            raise SopBenchError(
                f"decoys: {domain}'s decoy {decoy} is a version of the same procedure"
            )
    construction = data.get("construction", registry.CONSTRUCTION_SPASS)
    if construction not in registry.CONSTRUCTIONS:
        raise SopBenchError(f"construction must be one of {list(registry.CONSTRUCTIONS)}")
    exclusions = data.get("input_exclusions") or {}
    if not isinstance(exclusions, dict) or any(
        d not in domains or not isinstance(c, list) for d, c in exclusions.items()
    ):
        raise SopBenchError("input_exclusions must map run domains to lists of input columns")
    limit = data.get("tasks_per_domain")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise SopBenchError("tasks_per_domain must be a positive integer or null")
    name = data.get("lineup", registry.DEFAULT_LINEUP)
    if name not in registry.LINEUPS:
        raise SopBenchError(f"lineup must be one of {sorted(registry.LINEUPS)}, got {name!r}")
    lineup = registry.LINEUPS[name]
    wanted = data.get("readers", "all")
    if wanted == "all":
        readers = lineup.reported
    elif isinstance(wanted, list) and all(isinstance(r, str) for r in wanted):
        ids = {r.reader_id for r in lineup.readers}
        if any(r not in ids for r in wanted):
            raise SopBenchError(f"readers must be among lineup {name}'s: {sorted(ids)}")
        readers = tuple(r for r in lineup.readers if r.reader_id in wanted)
    else:
        raise SopBenchError("readers must be 'all' or a list of the lineup's reader ids")
    return SopBenchConfig(
        path=path,
        sha256=sha256,
        sopbench_dir=sopbench_dir,
        sopbench_commit=commit,
        lineup=lineup,
        domains=domains,
        decoys={d: decoys[d] for d in domains},
        seeds=tuple(seeds),
        tasks_per_domain=limit,
        readers=tuple(readers),
        construction=construction,
        max_turns=_positive(data, "max_turns", MAX_TURNS),
        turn_tokens=_positive(data, "turn_tokens", TURN_TOKENS),
        batch=_positive(data, "batch", BATCH),
        input_exclusions={d: tuple(c) for d, c in exclusions.items()},
    )


def load_config(path: Path) -> SopBenchConfig:
    raw = Path(path).read_bytes()
    return parse_config(yaml.safe_load(raw), Path(path), hashlib.sha256(raw).hexdigest())


# the checkout


def checkout_commit(sopbench_dir: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=sopbench_dir,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def check_checkout(config: SopBenchConfig) -> None:
    found = checkout_commit(config.sopbench_dir)
    if found != config.sopbench_commit:
        raise SopBenchError(
            f"{config.sopbench_dir} is at {found or 'no git commit'}, the config pins "
            f"{config.sopbench_commit}; fetch it with --fetch"
        )


def fetch(sopbench_dir: Path, commit: str, repository: str = REPOSITORY) -> None:
    """Clone the benchmark into `sopbench_dir` and check out `commit`."""
    sopbench_dir = Path(sopbench_dir)
    if not (sopbench_dir / ".git").is_dir():
        subprocess.run(["git", "clone", "--quiet", repository, str(sopbench_dir)], check=True)
    subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=sopbench_dir, check=False)
    subprocess.run(["git", "checkout", "--quiet", commit], cwd=sopbench_dir, check=True)


# the output directory


def read_log(out: Path) -> list[dict[str, Any]]:
    path = Path(out) / LOG_FILE
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@contextlib.contextmanager
def locked(out: Path):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / LOCK_FILE).open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append_unlocked(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with (Path(out) / LOG_FILE).open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(row, default=str) + "\n" for row in rows))


def _append(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with locked(out):
        _append_unlocked(out, rows)


def read_manifest(out: Path) -> dict[str, Any]:
    path = Path(out) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_manifest(out: Path, manifest: Mapping[str, Any]) -> None:
    path = Path(out) / MANIFEST_FILE
    staged = path.with_suffix(".json.writing")
    staged.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def update_manifest(out: Path, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    with locked(out):
        manifest = read_manifest(out)
        change(manifest)
        write_manifest(out, manifest)
    return manifest


def log_key(row: Mapping[str, Any]) -> tuple:
    return (
        row["reader"],
        row["domain"],
        row["presentation"],
        row["seed"],
        row["job"],
        row.get("task"),
    )


def settings_record() -> dict[str, Any]:
    return {
        "k": registry.MATCH_BLOCK,
        "steps": registry.STEPS,
        "lambda": registry.LAMBDA,
        "alpha": registry.ALPHA,
        "beta": registry.BETA,
        "lr": registry.LR,
    }


def open_run(config: SopBenchConfig, out: Path, phase: str) -> dict[str, Any]:
    """Create or resume `out`: the manifest must name the same config and data."""
    inputs = {
        d: {
            "columns": config.domain(d).input_columns,
            "excluded": list(config.input_exclusions.get(d, ())),
        }
        for d in config.domains
    }
    harness = {
        "max_turns": config.max_turns,
        "turn_tokens": config.turn_tokens,
        "batch": config.batch,
    }
    with locked(out):
        manifest = read_manifest(out)
        if manifest:
            if manifest.get("config", {}).get("sha256") != config.sha256:
                raise SopBenchError(f"{out} was made from another config; use another --out")
            if manifest.get("sopbench_commit") != config.sopbench_commit:
                raise SopBenchError(f"{out} was made from another SOP-Bench commit")
        else:
            manifest = {
                "config": {"path": str(config.path), "sha256": config.sha256},
                "sopbench_commit": config.sopbench_commit,
                "lineup": config.lineup.name,
                "construction": config.construction,
                "settings": settings_record(),
                "domains": config.domains,
                "decoys": config.decoys,
                "inputs": inputs,
                "harness": harness,
                "seeds": list(config.seeds),
                "passes": {},
                "readers": {},
                "sessions": [],
            }
        manifest["sessions"].append(
            {
                "id": uuid.uuid4().hex,
                "phase": phase,
                "started": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            }
        )
        write_manifest(out, manifest)
    return manifest


# the passes


def pass_dir(root: Path, domain: str, seed: int) -> Path:
    return Path(root) / str(seed) / skill_name(domain)


def conversations_for(
    config: SopBenchConfig, domains: Sequence[str] | None = None
) -> dict[str, list]:
    """Each domain's agent conversations its S-Pass is fitted under, by the domain's skill name."""
    out = {}
    for name in domains or config.domains:
        domain = config.domain(name)
        out[domain.skill().name] = agent_conversations(domain)
    return out


def make_passes(
    config: SopBenchConfig,
    out: Path,
    passes_root: Path,
    *,
    make: PassMaker,
    domains: Sequence[str] | None = None,
    log=print,
) -> None:
    """Each domain's tensor at each seed that is not yet made, then each recorded in the manifest
    with its sha256, its sidecar copied beside the results."""
    names = list(domains or config.domains)
    loaded = {d: config.domain(d) for d in {*names, *(config.decoys[d] for d in names)}}
    for seed in config.seeds:
        todo = [
            (loaded[d].skill(), loaded[config.decoys[d]].skill())
            for d in names
            if not (pass_dir(passes_root, d, seed) / SPASS_FILE).is_file()
        ]
        if todo:
            log(f"sopbench: seed {seed}: making {', '.join(s.name for s, _ in todo)}")
            make(todo, seed, Path(passes_root) / str(seed))
        update_manifest(
            out, lambda m, seed=seed: _record_passes(m, config, out, passes_root, names, seed)
        )


def _record_passes(manifest, config, out, passes_root, domains, seed) -> None:
    for d in domains:
        directory = pass_dir(passes_root, d, seed)
        if not (directory / SPASS_FILE).is_file():
            raise SopBenchError(f"the maker did not write {directory / SPASS_FILE}")
        side = json.loads((directory / SIDECAR_FILE).read_text(encoding="utf-8"))
        if side.get("construction") != config.construction:
            raise SopBenchError(
                f"{directory} was made by {side.get('construction')!r}, the run is "
                f"{config.construction!r}; "
                "use another --passes"
            )
        record = {
            "sha256": file_sha256(directory / SPASS_FILE),
            "plain_sha256": file_sha256(directory / PLAIN_FILE),
            "construction": side["construction"],
            "decoy": side["decoy"],
        }
        old = manifest["passes"].get(d, {}).get(str(seed))
        if old is not None and old["sha256"] != record["sha256"]:
            raise SopBenchError(f"{directory / SPASS_FILE} differs from the tensor {out} recorded")
        manifest["passes"].setdefault(d, {})[str(seed)] = record
        target = Path(out) / "passes" / str(seed) / directory.name / SIDECAR_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((directory / SIDECAR_FILE).read_bytes())


def _tensors(manifest: Mapping[str, Any], passes_root: Path, domain: str, seeds) -> dict:
    """The `plain` tensor and each seed's tensor, checked against the manifest."""
    out: dict[Any, Any] = {}
    for seed in seeds:
        directory = pass_dir(passes_root, domain, seed)
        record = manifest.get("passes", {}).get(domain, {}).get(str(seed))
        if record is None or not (directory / SPASS_FILE).is_file():
            raise SopBenchError(f"{domain}: seed {seed} has no tensor; run the passes phase first")
        if file_sha256(directory / SPASS_FILE) != record["sha256"]:
            raise SopBenchError(f"{directory / SPASS_FILE} is not the tensor the manifest recorded")
        out[seed] = read_spass(directory).tensor.astype(np.float32)
        if Presentation.PLAIN not in out:
            out[Presentation.PLAIN] = read_plain(directory).astype(np.float32)
    return out


# the episodes


SEEDLESS_SHARED_KEYS = ("sopbench_commit", "lineup", "domains", "inputs", "harness")


def reuse_seedless(config: SopBenchConfig, out: Path, other: Path) -> int:
    """Copy into `out` the `none`, `text` and `plain` episodes and repeats of `other`, a run of the
    same lineup that no tensor enters: a baseline run reuses them from the S-Pass run, since
    greedy decoding makes them the same episodes."""
    out, other = Path(out), Path(other)
    theirs = read_manifest(other)
    if not theirs:
        raise SopBenchError(f"{other} holds no run to reuse")
    seedless = {p.value for p in SEEDLESS}
    with locked(out):
        ours = read_manifest(out)
        if not ours:
            raise SopBenchError(f"{out} holds no run; open it with a phase first")
        for key in SEEDLESS_SHARED_KEYS:
            if ours.get(key) != theirs.get(key):
                raise SopBenchError(f"{other}'s {key} differs from {out}'s: not the same episodes")
        for reader_id, identity in theirs.get("readers", {}).items():
            ours["readers"].setdefault(reader_id, identity)
        done = {log_key(r) for r in read_log(out)}
        rows = []
        for row in read_log(other):
            if row["presentation"] in seedless and row["seed"] is None and log_key(row) not in done:
                done.add(log_key(row))
                rows.append(row)
        _append_unlocked(out, rows)
        ours.setdefault("reused_seedless", []).append({"from": str(other), "rows": len(rows)})
        write_manifest(out, ours)
    return len(rows)


def run_readers(
    config: SopBenchConfig,
    out: Path,
    passes_root: Path,
    *,
    domains: Sequence[str] | None = None,
    readers: Sequence[str] | None = None,
    reader_factory: Callable[[registry.Reader], Any] = make_reader,
    log=print,
) -> None:
    """Every chosen reader through every chosen domain, skipping what the log holds."""
    manifest = read_manifest(out)
    names = list(domains or config.domains)
    entries = [e for e in config.readers if readers is None or e.reader_id in readers]
    for entry in entries:
        done = {log_key(r) for r in read_log(out)}
        reader = reader_factory(entry)
        try:
            identity = reader.identity()
            manifest = update_manifest(
                out, lambda m, i=identity, r=entry.reader_id: m["readers"].__setitem__(r, i)
            )
            for name in names:
                domain = config.domain(name)
                tasks = domain.tasks(config.tasks_per_domain)
                tensors = _tensors(manifest, passes_root, name, config.seeds)
                toolbox = ToolBox(domain)
                runs = [(p, None) for p in SEEDLESS] + [
                    (Presentation.SPASS, s) for s in config.seeds
                ]
                for presentation, seed in runs:
                    pending = [
                        t
                        for t in tasks
                        if (entry.reader_id, name, presentation.value, seed, JOB_ANSWER, t.id)
                        not in done
                    ]
                    injected = presentation in (Presentation.PLAIN, Presentation.SPASS)
                    tensor = None
                    if injected:
                        tensor = (
                            tensors[seed]
                            if presentation is Presentation.SPASS
                            else tensors[Presentation.PLAIN]
                        )
                    for start in range(0, len(pending), config.batch):
                        chunk = pending[start : start + config.batch]
                        records = run_episodes(
                            reader,
                            presentation,
                            chunk,
                            domain,
                            toolbox,
                            tensor,
                            max_turns=config.max_turns,
                            turn_tokens=config.turn_tokens,
                        )
                        _append(
                            out,
                            [
                                {
                                    "reader": entry.reader_id,
                                    "domain": name,
                                    "presentation": presentation.value,
                                    "seed": seed,
                                    "job": JOB_ANSWER,
                                    **r,
                                }
                                for r in records
                            ],
                        )
                    if pending:
                        log(
                            f"sopbench: {entry.reader_id}: {name}: {presentation.value}"
                            + (f" seed {seed}" if seed is not None else "")
                            + f": {len(pending)} episodes"
                        )
                repeats = [(Presentation.PLAIN, None)] + [
                    (Presentation.SPASS, s) for s in config.seeds
                ]
                todo = [
                    (p, s)
                    for p, s in repeats
                    if (entry.reader_id, name, p.value, s, JOB_REPEAT, None) not in done
                ]
                if todo:
                    prompt = build_prompt(Presentation.TEXT, REPEAT_PROMPT, domain.sop)
                    items = [
                        (
                            tensors[s] if p is Presentation.SPASS else tensors[Presentation.PLAIN],
                            prompt,
                            domain.sop,
                        )
                        for p, s in todo
                    ]
                    answers = reader.read_embeds_many(items, max_new_tokens=REPEAT_MAX_NEW_TOKENS)
                    _append(
                        out,
                        [
                            {
                                "reader": entry.reader_id,
                                "domain": name,
                                "presentation": p.value,
                                "seed": s,
                                "job": JOB_REPEAT,
                                "answer": a,
                            }
                            for (p, s), a in zip(todo, answers, strict=True)
                        ],
                    )
                if toolbox.dropped:
                    update_manifest(
                        out,
                        lambda m, n=name, t=toolbox: (
                            m.setdefault("dropped_tool_arguments", {})
                            .setdefault(n, {})
                            .update(
                                {
                                    k: m["dropped_tool_arguments"][n].get(k, 0) + v
                                    for k, v in t.dropped.items()
                                }
                            )
                        ),
                    )
        finally:
            reader.close()


# the decode


def decode_all(
    config: SopBenchConfig,
    out: Path,
    passes_root: Path,
    decode_table: Callable[[], Any],
    *,
    domains: Sequence[str] | None = None,
    log=print,
) -> None:
    """Each tensor decoded with the authorized model's public table (Section 5.4), written to
    `decode/<domain>/<seed>.json` with every readout's text."""
    from skillpass.attacks.decode import decode_report

    manifest = read_manifest(out)
    table = None
    for name in domains or config.domains:
        domain = config.domain(name)
        decoy = config.domain(config.decoys[name])
        for seed in config.seeds:
            path = Path(out) / "decode" / name / f"{seed}.json"
            if path.is_file():
                continue
            tensors = _tensors(manifest, passes_root, name, [seed])
            table = table or decode_table()
            ids = np.asarray(table.tokenise(domain.sop), dtype=np.int64)
            decoy_ids = align_decoy(table.tokenise, decoy.sop, len(ids))
            report = decode_report(
                table.table,
                tensors[seed],
                ids,
                decoy_ids,
                detokenise=table.detokenise,
                skill_text=domain.skill().transcription,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            log(
                f"sopbench: decoded {name} seed {seed}: Decode RL "
                f"{report['best_rougeL_skill']:.3f}, "
                f"Floor RL {report['floor_rougeL']:.3f}"
            )
