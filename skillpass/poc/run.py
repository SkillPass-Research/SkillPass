"""The proof of concept (Section 5.2): one S-Pass per synthetic skill and seed, read by the
authorized model and its relatives under `none`, `text`, `plain` and `spass`, repeated from the
tensor, and decoded by the model-free adversary. Four phases, each resumable:

    passes   one S-Pass per skill and seed, the lineup's models loaded once per seed
    answers  every reported reader answers every task under every presentation, then repeats
             the skill from `plain` and from each seed's S-Pass
    decode   each S-Pass decoded with the authorized model's public table at every strip rank
    report   Table 1: per skill and pooled, success under the four presentations, rho for the
             authorized model, residual uplift for each relative, Repeat RL, Decode RL and
             Floor RL

Evidence goes to `--out`: `manifest.json`, `answers.jsonl`, `decode/<skill>/<seed>.json`,
`report.json` and `table.md`; the tensors go to `--passes/<seed>/<skill>/`.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml

from skillpass import registry
from skillpass.metrics import cer, iota, residual_uplift, rho, rouge_l, spread, uplift
from skillpass.models import FAKE_FAMILY, make_reader
from skillpass.prompts import (
    ANSWER_MAX_NEW_TOKENS,
    REPEAT_MAX_NEW_TOKENS,
    REPEAT_PROMPT,
    build_prompt,
)
from skillpass.registry import TEXT_PRESENTATIONS, Presentation
from skillpass.skills import SUITE_ROOT, Skill, check, load_suite
from skillpass.spass.decoy import decoy_ids as align_decoy
from skillpass.spass.file import SPASS_FILE, file_sha256, read_plain, read_spass

LOG_FILE, MANIFEST_FILE = "answers.jsonl", "manifest.json"
REPORT_FILE, TABLE_FILE = "report.json", "table.md"
JOB_ANSWER, JOB_REPEAT = "answer", "repeat"
READ_BATCH = 32
SEEDLESS = (Presentation.NONE, Presentation.TEXT, Presentation.PLAIN)
PHASES = ("all", "passes", "answers", "decode", "report")

PassMaker = Callable[[Sequence[tuple[Skill, Skill]], int, Path], None]
ReaderFactory = Callable[[registry.Reader], Any]
DecodeTableFactory = Callable[[registry.Reader], Any]


class PocError(ValueError):
    """A config or an output directory the run refuses."""


@dataclass(frozen=True)
class PocConfig:
    path: Path
    sha256: str
    lineup: registry.Lineup
    skills: tuple[Skill, ...]
    decoys: dict[str, Skill]
    seeds: tuple[int, ...]

    @property
    def authorized(self) -> registry.Reader:
        return self.lineup.authorized

    @property
    def readers(self) -> tuple[registry.Reader, ...]:
        return self.lineup.reported


def load_config(path: Path, *, suite_root: Path = SUITE_ROOT) -> PocConfig:
    """`skills` (names), `decoys` (skill to decoy skill), `seeds`, optional `lineup`."""
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise PocError("the config must be a mapping")
    unknown = sorted(set(data) - {"skills", "decoys", "seeds", "lineup"})
    if unknown:
        raise PocError(f"unknown field {unknown[0]!r}")
    suite = {s.name: s for s in load_suite(suite_root)}
    names = data.get("skills")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise PocError("skills must be a non-empty list of distinct skill names")
    absent = [n for n in names if n not in suite]
    if absent:
        raise PocError(f"no such skill: {absent}; the suite has {sorted(suite)}")
    decoys = data.get("decoys") or {}
    for name in names:
        decoy = decoys.get(name)
        if decoy not in suite:
            raise PocError(f"decoys: {name} needs a decoy skill from the suite")
        if decoy == name:
            raise PocError(f"decoys: {name} cannot be its own decoy")
    seeds = data.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise PocError("seeds must be a non-empty list of distinct non-negative integers")
    lineup_name = data.get("lineup", registry.DEFAULT_LINEUP)
    if lineup_name not in registry.LINEUPS:
        raise PocError(f"lineup must be one of {sorted(registry.LINEUPS)}")
    return PocConfig(
        path=Path(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        lineup=registry.LINEUPS[lineup_name],
        skills=tuple(suite[n] for n in names),
        decoys={n: suite[decoys[n]] for n in names},
        seeds=tuple(seeds),
    )


# the output directory


def pass_dir(root: Path, seed: int, skill: str) -> Path:
    return Path(root) / str(seed) / skill


def read_log(out: Path) -> list[dict[str, Any]]:
    path = Path(out) / LOG_FILE
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _append(out: Path, rows: Sequence[dict[str, Any]]) -> None:
    with (Path(out) / LOG_FILE).open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def row_key(row: dict[str, Any]) -> tuple:
    return (
        row["reader"],
        row["skill"],
        row.get("task"),
        row["presentation"],
        row["seed"],
        row["job"],
    )


def read_manifest(out: Path) -> dict[str, Any]:
    path = Path(out) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_manifest(out: Path, manifest: dict[str, Any]) -> None:
    (Path(out) / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def open_run(config: PocConfig, out: Path) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(out)
    if manifest and manifest.get("config", {}).get("sha256") != config.sha256:
        raise PocError(f"{out} was made from another config; use another --out")
    if not manifest:
        manifest = {
            "config": {"path": str(config.path), "sha256": config.sha256},
            "lineup": config.lineup.name,
            "settings": settings_record(),
            "skills": [s.name for s in config.skills],
            "decoys": {k: v.name for k, v in config.decoys.items()},
            "seeds": list(config.seeds),
            "fake": config.authorized.family == FAKE_FAMILY,
            "passes": {},
            "readers": {},
        }
        write_manifest(out, manifest)
    return manifest


def settings_record() -> dict[str, Any]:
    return {
        "k": registry.MATCH_BLOCK,
        "steps": registry.STEPS,
        "lambda": registry.LAMBDA,
        "alpha": registry.ALPHA,
        "beta": registry.BETA,
        "lr": registry.LR,
    }


# the passes


def make_passes(
    config: PocConfig, out: Path, passes_root: Path, *, make: PassMaker, log=print
) -> None:
    """Each skill's S-Pass at each seed that is not yet on disk, then every tensor's sha256
    recorded in the manifest and its sidecar copied beside the results."""
    for seed in config.seeds:
        todo = [
            (s, config.decoys[s.name])
            for s in config.skills
            if not (pass_dir(passes_root, seed, s.name) / SPASS_FILE).is_file()
        ]
        if todo:
            log(f"poc: seed {seed}: making {', '.join(s.name for s, _ in todo)}")
            make(todo, seed, Path(passes_root) / str(seed))
        manifest = read_manifest(out)
        for skill in config.skills:
            directory = pass_dir(passes_root, seed, skill.name)
            if not (directory / SPASS_FILE).is_file():
                raise PocError(f"the maker did not write {directory / SPASS_FILE}")
            digest = file_sha256(directory / SPASS_FILE)
            old = manifest["passes"].get(skill.name, {}).get(str(seed))
            if old is not None and old != digest:
                raise PocError(f"{directory} differs from the S-Pass {out} recorded")
            manifest["passes"].setdefault(skill.name, {})[str(seed)] = digest
            target = Path(out) / "passes" / str(seed) / skill.name / "spass.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory / "spass.json").read_bytes())
        write_manifest(out, manifest)


def tensors_for(config: PocConfig, passes_root: Path, skill: Skill) -> dict[Any, np.ndarray]:
    """The `plain` tensor and each seed's S-Pass for `skill`, checked against the manifest's
    sidecar expectations: the skill, its source and the authorized model."""
    out: dict[Any, np.ndarray] = {}
    for seed in config.seeds:
        directory = pass_dir(passes_root, seed, skill.name)
        try:
            sp = read_spass(directory)
        except FileNotFoundError as error:
            raise PocError(
                f"{skill.name}: seed {seed}: {error}; run the passes phase first"
            ) from error
        side = sp.sidecar
        if side["skill"] != skill.name or side["source_sha256"] != skill.source_sha256:
            raise PocError(f"{directory} was made for another skill or another version of it")
        if config.authorized.family != FAKE_FAMILY and side["model"] != config.authorized.reader_id:
            raise PocError(
                f"{directory} was made for {side['model']}, not the lineup's authorized model"
            )
        out[seed] = sp.tensor.astype(np.float32)
        if Presentation.PLAIN not in out:
            out[Presentation.PLAIN] = read_plain(directory).astype(np.float32)
    return out


# the answers


def _ask(reader, presentation: Presentation, tensors, asks, seed, budget: int):
    """Answer `asks`, `(skill, question)` pairs, under one presentation as one batch."""
    if presentation in TEXT_PRESENTATIONS:
        prompts = [
            build_prompt(presentation, q, s.source if presentation is Presentation.TEXT else None)
            for s, q in asks
        ]
        answers = reader.read_many(prompts, max_new_tokens=budget)
    else:
        key = Presentation.PLAIN if presentation is Presentation.PLAIN else seed
        items = [
            (tensors[s.name][key], build_prompt(Presentation.TEXT, q, s.source), s.source)
            for s, q in asks
        ]
        answers = reader.read_embeds_many(items, max_new_tokens=budget)
    truncated = [r == "length" for r in reader.last_finish_reasons]
    return answers, truncated


def run_readers(
    config: PocConfig,
    out: Path,
    passes_root: Path,
    *,
    reader_factory: ReaderFactory = make_reader,
    readers: Sequence[str] | None = None,
    log=print,
) -> None:
    """Every chosen reader through every skill, skipping what the log already holds."""
    tensors = {s.name: tensors_for(config, passes_root, s) for s in config.skills}
    for entry in config.readers:
        if readers is not None and entry.reader_id not in readers:
            continue
        reader = reader_factory(entry)
        try:
            manifest = read_manifest(out)
            manifest["readers"][entry.reader_id] = reader.identity()
            write_manifest(out, manifest)
            done = {row_key(r) for r in read_log(out)}
            runs = [(p, None) for p in SEEDLESS] + [(Presentation.SPASS, s) for s in config.seeds]
            for presentation, seed in runs:
                pending = [
                    (s, t)
                    for s in config.skills
                    for t in s.tasks
                    if (entry.reader_id, s.name, t.id, presentation.value, seed, JOB_ANSWER)
                    not in done
                ]
                for start in range(0, len(pending), READ_BATCH):
                    batch = pending[start : start + READ_BATCH]
                    answers, truncated = _ask(
                        reader,
                        presentation,
                        tensors,
                        [(s, t.text) for s, t in batch],
                        seed,
                        ANSWER_MAX_NEW_TOKENS,
                    )
                    _append(
                        out,
                        [
                            {
                                "reader": entry.reader_id,
                                "skill": s.name,
                                "task": t.id,
                                "presentation": presentation.value,
                                "seed": seed,
                                "job": JOB_ANSWER,
                                "answer": a,
                                "passed": check(t, a),
                                "truncated": cut,
                            }
                            for (s, t), a, cut in zip(batch, answers, truncated, strict=True)
                        ],
                    )
                if pending:
                    log(
                        f"poc: {entry.reader_id}: {presentation.value}"
                        + (f" seed {seed}" if seed is not None else "")
                        + f": {len(pending)} answers"
                    )
            repeats = [(Presentation.PLAIN, None)] + [(Presentation.SPASS, s) for s in config.seeds]
            for presentation, seed in repeats:
                todo = [
                    s
                    for s in config.skills
                    if (entry.reader_id, s.name, None, presentation.value, seed, JOB_REPEAT)
                    not in done
                ]
                if not todo:
                    continue
                texts, truncated = _ask(
                    reader,
                    presentation,
                    tensors,
                    [(s, REPEAT_PROMPT) for s in todo],
                    seed,
                    REPEAT_MAX_NEW_TOKENS,
                )
                _append(
                    out,
                    [
                        {
                            "reader": entry.reader_id,
                            "skill": s.name,
                            "task": None,
                            "presentation": presentation.value,
                            "seed": seed,
                            "job": JOB_REPEAT,
                            "answer": text,
                            "passed": None,
                            "truncated": cut,
                        }
                        for s, text, cut in zip(todo, texts, truncated, strict=True)
                    ],
                )
        finally:
            reader.close()


# the decode


def decode_all(
    config: PocConfig, out: Path, passes_root: Path, *, decode_table: DecodeTableFactory, log=print
) -> None:
    """Each S-Pass decoded with the authorized model's public table, written to
    `decode/<skill>/<seed>.json`."""
    from skillpass.attacks.decode import decode_report

    table = None
    for skill in config.skills:
        for seed in config.seeds:
            path = Path(out) / "decode" / skill.name / f"{seed}.json"
            if path.is_file():
                continue
            table = table or decode_table(config.authorized)
            ids = np.asarray(table.tokenise(skill.source), dtype=np.int64)
            decoy_ids = align_decoy(table.tokenise, config.decoys[skill.name].source, len(ids))
            tensor = read_spass(pass_dir(passes_root, seed, skill.name)).tensor.astype(np.float32)
            report = decode_report(
                table.table,
                tensor,
                ids,
                decoy_ids,
                detokenise=table.detokenise,
                skill_text=skill.transcription,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
            log(
                f"poc: decoded {skill.name} seed {seed}: Decode RL "
                f"{report['best_rougeL_skill']:.3f}, "
                f"Floor RL {report['floor_rougeL']:.3f}"
            )


# the report


def _m(figure: dict[str, Any] | None) -> float | None:
    return None if figure is None else figure["mean"]


def build_report(config: PocConfig, out: Path) -> dict[str, Any]:
    """Table 1 from the answer log and the decodes: per skill and pooled over every task."""
    rows = read_log(out)
    seen: set[tuple] = set()
    for r in rows:
        if row_key(r) in seen:
            raise PocError(f"{out}/{LOG_FILE} holds {row_key(r)} twice")
        seen.add(row_key(r))
    seeds = list(config.seeds)
    authorized = config.authorized.reader_id
    passed: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    repeats: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["job"] == JOB_ANSWER:
            passed[r["reader"]][r["skill"]][(r["presentation"], r["seed"])].append(
                bool(r["passed"])
            )
        else:
            repeats[r["reader"]][r["skill"]][(r["presentation"], r["seed"])] = r["answer"]
    n_tasks = {s.name: len(s.tasks) for s in config.skills}

    def success(reader: str, names: Sequence[str], presentation: str, seed: int | None):
        values: list[bool] = []
        for name in names:
            got = passed[reader][name][(presentation, seed)]
            if len(got) != n_tasks[name]:
                return None
            values += got
        return mean(values)

    def figures(names: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for entry in config.readers:
            reader = entry.reader_id
            none = success(reader, names, "none", None)
            text = success(reader, names, "text", None)
            plain = success(reader, names, "plain", None)
            spass = [success(reader, names, "spass", s) for s in seeds]
            if None in (none, text, plain) or None in spass:
                result[reader] = None
                continue
            u_text, u_plain = uplift(text, none), uplift(plain, none)
            u_spass = [uplift(c, none) for c in spass]
            mine: dict[str, Any] = {
                "role": entry.role.value,
                "label": entry.label,
                "success": {"none": none, "text": text, "plain": plain, "spass": spread(spass)},
                "uplift": {"text": u_text, "plain": u_plain, "spass": spread(u_spass)},
            }
            if reader == authorized:
                mine["rho"] = spread([rho(u_text, u) for u in u_spass])
                mine["iota"] = iota(u_text, u_plain)
            else:
                mine["residual_uplift"] = spread([residual_uplift(u, u_plain) for u in u_spass])
            result[reader] = mine
        return result

    report: dict[str, Any] = {
        "config": {"path": str(config.path), "sha256": config.sha256},
        "lineup": config.lineup.name,
        "settings": settings_record(),
        "authorized": authorized,
        "readers": [
            {"reader": e.reader_id, "role": e.role.value, "label": e.label} for e in config.readers
        ],
        "seeds": seeds,
        "fake": config.authorized.family == FAKE_FAMILY,
        "skills": {},
    }
    complete = []
    for skill in config.skills:
        name = skill.name
        utility = figures([name])
        if any(v is None for v in utility.values()):
            report["skills"][name] = {"complete": False}
            continue
        complete.append(name)
        target = skill.transcription
        for reader in utility:
            rep = repeats[reader][name]
            utility[reader]["repeat_rougeL"] = {
                "plain": rouge_l(target, rep[("plain", None)]) if ("plain", None) in rep else None,
                "spass": spread(
                    [rouge_l(target, rep[("spass", s)]) for s in seeds if ("spass", s) in rep]
                ),
            }
            utility[reader]["repeat_cer"] = spread(
                [cer(target, rep[("spass", s)]) for s in seeds if ("spass", s) in rep]
            )
        report["skills"][name] = {
            "complete": True,
            "decoy": config.decoys[name].name,
            "tasks": n_tasks[name],
            "utility": utility,
            "reconstruction": _reconstruction(out, name, seeds),
        }
    report["incomplete"] = [s.name for s in config.skills if s.name not in complete]
    pooled = figures(complete) if complete else None
    if pooled:
        for reader, mine in pooled.items():
            if mine is None:
                continue
            per = [
                report["skills"][d]["utility"][reader]["repeat_rougeL"]["spass"] for d in complete
            ]
            per = [x["mean"] for x in per if x]
            mine["repeat_rougeL"] = {"spass": min(per) if per else None, "rule": "lowest skill"}
    report["pooled"] = {
        "skills": complete,
        "tasks": sum(n_tasks[d] for d in complete),
        "utility": pooled,
        "reconstruction": {
            "rule": "per skill; the worst skill shown",
            **_worst(report["skills"], complete),
        },
    }
    return report


def _reconstruction(out: Path, name: str, seeds: Sequence[int]) -> dict[str, Any] | None:
    decodes = []
    for seed in seeds:
        path = Path(out) / "decode" / name / f"{seed}.json"
        if not path.is_file():
            return None
        decodes.append(json.loads(path.read_text(encoding="utf-8")))
    rank0 = [next(r for r in d["strip"] if r["rank"] == 0) for d in decodes]
    return {
        "decode_rougeL": spread([d["best_rougeL_skill"] for d in decodes]),
        "floor_rougeL": spread([d["floor_rougeL"] for d in decodes]),
        "lowest_decode_cer": spread([d["lowest_decode_cer"] for d in decodes]),
        "rank0_token_recovery": spread([r["token_recovery"] for r in rank0]),
        "rank0_rougeL_decoy": spread([r["rougeL_decoy"] for r in rank0]),
    }


def _worst(skills: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    recon = [skills[n]["reconstruction"] for n in names if skills[n].get("reconstruction")]
    if not recon:
        return {"decode_rougeL": None, "floor_rougeL": None}
    return {
        "decode_rougeL": max(_m(r["decode_rougeL"]) for r in recon),
        "floor_rougeL": max(_m(r["floor_rougeL"]) for r in recon),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def _cell_pct(figure: dict[str, Any] | None) -> str:
    if figure is None:
        return "n/a"
    text = _pct(figure["mean"])
    if figure["min"] != figure["max"]:
        text += f" [{_pct(figure['min'])}, {_pct(figure['max'])}]"
    return text


def _cell(figure: dict[str, Any] | None, digits: int = 3) -> str:
    if figure is None:
        return "—"
    text = f"{figure['mean']:.{digits}f}"
    if figure["min"] != figure["max"]:
        text += f" [{figure['min']:.{digits}f}, {figure['max']:.{digits}f}]"
    return text


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _cell_pp(figure: dict[str, Any] | None) -> str:
    """A figure in percentage points: the mean with the seed range."""
    if figure is None:
        return "n/a"
    return _cell({k: 100 * v for k, v in figure.items() if k in ("mean", "min", "max")}, 1)


def _utility_rows(utility: dict[str, Any], authorized: str) -> list[str]:
    out = [
        "| Reader | None | Text | Plain | S-Pass | rho (pp) / r_L | Repeat RL |",
        "|---|---|---|---|---|---|---|",
    ]
    for reader, mine in utility.items():
        if mine is None:
            out.append(f"| `{reader}` | n/a | | | | | |")
            continue
        own = (
            f"rho {_cell_pp(mine['rho'])}"
            if reader == authorized
            else f"r_L {_cell(mine['residual_uplift'], 2)}"
        )
        rep = mine.get("repeat_rougeL") or {}
        repeat = rep.get("spass")
        repeat_text = (
            _cell(repeat)
            if isinstance(repeat, dict)
            else (f"{repeat:.3f}" if repeat is not None else "n/a")
        )
        s = mine["success"]
        out.append(
            f"| `{reader}` ({mine['label']}) | {_pct(s['none'])} | {_pct(s['text'])} | "
            f"{_pct(s['plain'])} | {_cell_pct(s['spass'])} | {own} | {repeat_text} |"
        )
    return out


def table_markdown(report: dict[str, Any]) -> str:
    settings = ", ".join(f"{k} {v:g}" for k, v in report["settings"].items())
    out = [
        "# Proof of concept on synthetic skills (Section 5.2, Table 1)",
        "",
        f"Lineup `{report['lineup']}`, authorized model `{report['authorized']}`; seeds "
        f"{', '.join(map(str, report['seeds']))}; settings: {settings}.",
        "",
        "Success is in %. `none`, `text` and `plain` are read once (greedy decoding); S-Pass is "
        "the "
        "mean over seeds with the lowest and highest seed in brackets. rho is the authorized "
        "model's "
        "utility loss, Delta(text) - Delta(S-Pass), in percentage points; r_L is a relative's "
        "residual "
        "uplift, Delta(S-Pass) / Delta(plain), a dash where its plain uplift is under 20 points. "
        "Repeat RL is the ROUGE-L of the reader's repetition of the skill from the S-Pass. Decode "
        "RL "
        "is the best nearest-token ROUGE-L against the skill over the stripping ranks; Floor RL is "
        "the "
        "decoy's own overlap with the skill.",
        "",
    ]
    if report.get("fake"):
        out += ["> **FAKE READERS.** Nothing in this table is evidence.", ""]
    if report.get("incomplete"):
        out += [f"Not yet complete: {', '.join(report['incomplete'])}.", ""]
    for name, section in report["skills"].items():
        if not section.get("complete"):
            continue
        out += [f"## {name}", "", f"{section['tasks']} tasks; decoy `{section['decoy']}`.", ""]
        out += _utility_rows(section["utility"], report["authorized"])
        recon = section["reconstruction"]
        if recon is None:
            out += ["", "Not yet decoded.", ""]
        else:
            out += [
                "",
                f"Reconstruction: Decode RL {_cell(recon['decode_rougeL'])}; Floor RL "
                f"{_cell(recon['floor_rougeL'])}; token recovery at rank 0 "
                f"{_cell(recon['rank0_token_recovery'])}; lowest decode CER "
                f"{_cell(recon['lowest_decode_cer'])}.",
                "",
            ]
    pooled = report["pooled"]
    if pooled.get("utility"):
        out += [f"## Pooled ({pooled['tasks']} tasks)", ""]
        out += _utility_rows(pooled["utility"], report["authorized"])
        out += [
            "",
            "Pooled Repeat RL is the lowest skill's. Reconstruction is per skill; the worst skill: "
            f"Decode RL {_fmt(pooled['reconstruction']['decode_rougeL'])}, Floor RL "
            f"{_fmt(pooled['reconstruction']['floor_rougeL'])}.",
            "",
        ]
    return "\n".join(out)


def write_report(config: PocConfig, out: Path) -> dict[str, Any]:
    report = build_report(config, out)
    (Path(out) / REPORT_FILE).write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (Path(out) / TABLE_FILE).write_text(table_markdown(report), encoding="utf-8")
    return report


def run_poc(
    config: PocConfig,
    out: Path,
    passes_root: Path,
    *,
    make: PassMaker,
    decode_table: DecodeTableFactory,
    reader_factory: ReaderFactory = make_reader,
    phase: str = "all",
    readers: Sequence[str] | None = None,
    log=print,
) -> dict[str, Any] | None:
    """Run the phases named by `phase` into `out`, resuming what is there."""
    if phase not in PHASES:
        raise PocError(f"phase must be one of {PHASES}")
    open_run(config, out)
    if phase in ("all", "passes"):
        make_passes(config, out, passes_root, make=make, log=log)
    if phase in ("all", "answers"):
        run_readers(
            config, out, passes_root, reader_factory=reader_factory, readers=readers, log=log
        )
    if phase in ("all", "decode"):
        decode_all(config, out, passes_root, decode_table=decode_table, log=log)
    if phase in ("all", "report"):
        return write_report(config, out)
    return None
