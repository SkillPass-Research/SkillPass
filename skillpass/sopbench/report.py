"""The SOP-Bench tables (Sections 5.3 to 5.5) from one run's answer log and decodes.

Utility (Table 2) pools the domains where the procedure as text lifts the authorized model by
at least 20 percentage points, tasks concatenated: success under the four presentations, the
loss rho, the injection cost iota and Repeat RL. Reconstruction (Table 3) gives Decode RL and
Floor RL with every domain at equal weight. Transfer (Table 4) gives each reported relative's
success, its uplift from `plain` and from the S-Pass, its residual uplift r_L and Repeat RL,
over the same pool. A pool over every domain and every domain's own figures sit beside them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from skillpass.metrics import cer, iota, residual_uplift, rho, rouge_l, spread, uplift
from skillpass.registry import USABLE_UPLIFT_MIN, Role

REPORT_FILE, TABLE_FILE = "report.json", "table.md"


def _m(figure: dict[str, Any] | None) -> float | None:
    return None if figure is None else figure["mean"]


def build_report(config, out: Path) -> dict[str, Any]:
    from skillpass.sopbench.runner import (
        JOB_ANSWER,
        JOB_REPEAT,
        SopBenchError,
        log_key,
        read_log,
        read_manifest,
    )

    out = Path(out)
    manifest = read_manifest(out)
    rows = read_log(out)
    seen: set[tuple] = set()
    for r in rows:
        if log_key(r) in seen:
            raise SopBenchError(f"{out}/answers.jsonl holds {log_key(r)} twice")
        seen.add(log_key(r))
    seeds = list(config.seeds)
    authorized = config.authorized.reader_id
    passed: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    no_final: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    repeats: dict = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["presentation"], r["seed"])
        if r["job"] == JOB_ANSWER:
            passed[r["reader"]][r["domain"]][key].append(bool(r["passed"]))
            no_final[r["reader"]][r["domain"]][key] += r.get("final") is None
        elif r["job"] == JOB_REPEAT:
            repeats[r["reader"]][r["domain"]][key] = r["answer"]
    n_tasks = {d: len(config.domain(d).tasks(config.tasks_per_domain)) for d in config.domains}

    def success(reader: str, domains: Sequence[str], presentation: str, seed: int | None):
        values: list[bool] = []
        for d in domains:
            got = passed[reader][d][(presentation, seed)]
            if len(got) != n_tasks[d]:
                return None
            values += got
        return mean(values)

    def figures(domains: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for entry in config.readers:
            reader = entry.reader_id
            none = success(reader, domains, "none", None)
            text = success(reader, domains, "text", None)
            plain = success(reader, domains, "plain", None)
            spass = [success(reader, domains, "spass", s) for s in seeds]
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
            if entry.role is Role.AUTHORIZED:
                mine["rho"] = spread([rho(u_text, u) for u in u_spass])
                mine["iota"] = iota(u_text, u_plain)
            else:
                mine["residual_uplift"] = spread([residual_uplift(u, u_plain) for u in u_spass])
            result[reader] = mine
        return result

    def repeat_scores(domain: str, target: str) -> dict[str, Any]:
        out_: dict[str, Any] = {}
        for entry in config.readers:
            rep = repeats[entry.reader_id][domain]
            out_[entry.reader_id] = {
                "plain": rouge_l(target, rep[("plain", None)]) if ("plain", None) in rep else None,
                "spass": spread(
                    [rouge_l(target, rep[("spass", s)]) for s in seeds if ("spass", s) in rep]
                ),
                "spass_cer": spread(
                    [cer(target, rep[("spass", s)]) for s in seeds if ("spass", s) in rep]
                ),
            }
        return out_

    report: dict[str, Any] = {
        "config": manifest.get("config"),
        "sopbench_commit": manifest.get("sopbench_commit"),
        "lineup": config.lineup.name,
        "construction": config.construction,
        "settings": manifest.get("settings"),
        "seeds": seeds,
        "authorized": authorized,
        "readers": [
            {"reader": e.reader_id, "role": e.role.value, "label": e.label} for e in config.readers
        ],
        "usable_uplift_min": USABLE_UPLIFT_MIN,
        "domains": {},
    }
    complete: list[str] = []
    for name in config.domains:
        utility = figures([name])
        if any(v is None for v in utility.values()):
            report["domains"][name] = {"complete": False}
            continue
        complete.append(name)
        domain = config.domain(name)
        text_uplift = utility[authorized]["uplift"]["text"]
        report["domains"][name] = {
            "complete": True,
            "decoy": config.decoys[name],
            "tasks": n_tasks[name],
            "text_uplift": text_uplift,
            "in_pool": text_uplift >= USABLE_UPLIFT_MIN,
            "utility": utility,
            "repeat_rougeL": repeat_scores(name, domain.skill().transcription),
            "no_final_answer": {
                f"{p}:{s}" if s is not None else p: v
                for (p, s), v in no_final[authorized][name].items()
            },
            "reconstruction": _reconstruction(out, name, seeds),
        }
    pool = [d for d in complete if report["domains"][d]["in_pool"]]
    report["incomplete"] = [d for d in config.domains if d not in complete]
    report["excluded_from_pool"] = [d for d in complete if d not in pool]
    report["pool"] = _pool(report, pool, figures)
    report["all"] = _pool(report, complete, figures)
    report["reconstruction"] = _reconstruction_pooled(report, complete)
    return report


def _pool(report: dict[str, Any], domains: Sequence[str], figures) -> dict[str, Any]:
    utility = figures(domains) if domains else None
    repeat: dict[str, Any] = {}
    if utility:
        for reader in utility:
            per = [report["domains"][d]["repeat_rougeL"][reader]["spass"] for d in domains]
            means = [x["mean"] for x in per if x]
            repeat[reader] = {
                "spass_mean_over_domains": mean(means) if means else None,
                "spass_lowest_domain": min(means) if means else None,
            }
    return {
        "domains": list(domains),
        "tasks": sum(report["domains"][d]["tasks"] for d in domains),
        "utility": utility,
        "repeat_rougeL": repeat,
    }


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
        "floor_rougeL": decodes[0]["floor_rougeL"],
        "lowest_decode_cer": spread([d["lowest_decode_cer"] for d in decodes]),
        "rank0_token_recovery": spread([r["token_recovery"] for r in rank0]),
        "rank0_rougeL_decoy": spread([r["rougeL_decoy"] for r in rank0]),
    }


def _reconstruction_pooled(report: dict[str, Any], domains: Sequence[str]) -> dict[str, Any]:
    """Every domain at equal weight (Table 3)."""
    recon = [report["domains"][d]["reconstruction"] for d in domains]
    recon = [r for r in recon if r is not None]
    if not recon:
        return {"domains": 0, "decode_rougeL": None, "floor_rougeL": None}
    return {
        "domains": len(recon),
        "decode_rougeL": mean(_m(r["decode_rougeL"]) for r in recon),
        "floor_rougeL": mean(r["floor_rougeL"] for r in recon),
        "lowest_decode_cer": mean(_m(r["lowest_decode_cer"]) for r in recon),
        "rank0_token_recovery": mean(_m(r["rank0_token_recovery"]) for r in recon),
    }


# rendering


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}"


def _pp(figure: dict[str, Any] | None) -> str:
    if figure is None:
        return "n/a"
    text = f"{100 * figure['mean']:.1f}"
    if figure["min"] != figure["max"]:
        text += f" [{100 * figure['min']:.1f}, {100 * figure['max']:.1f}]"
    return text


def _num(figure: dict[str, Any] | float | None, digits: int = 3) -> str:
    if figure is None:
        return "—"
    if isinstance(figure, dict):
        text = f"{figure['mean']:.{digits}f}"
        if figure["min"] != figure["max"]:
            text += f" [{figure['min']:.{digits}f}, {figure['max']:.{digits}f}]"
        return text
    return f"{figure:.{digits}f}"


def _authorized_row(label: str, pool: dict[str, Any], authorized: str) -> str:
    mine = (pool.get("utility") or {}).get(authorized)
    if not mine:
        return f"| {label} | n/a | | | | | | | |"
    s = mine["success"]
    rep = pool["repeat_rougeL"].get(authorized, {}).get("spass_mean_over_domains")
    return (
        f"| {label} | {len(pool['domains'])} / {pool['tasks']} | {_pct(s['none'])} | "
        f"{_pct(s['text'])} | "
        f"{_pct(s['plain'])} | {_pp(s['spass'])} | {_pp(mine['rho'])} | {100 * mine['iota']:.1f} | "
        f"{_num(rep, 2)} |"
    )


def table_markdown(report: dict[str, Any]) -> str:
    authorized = report["authorized"]
    construction = report["construction"]
    settings = ", ".join(f"{k} {v:g}" for k, v in (report.get("settings") or {}).items())
    out = [
        f"# SOP-Bench: lineup `{report['lineup']}`, construction `{construction}`",
        "",
        f"SOP-Bench at `{report['sopbench_commit']}`; seeds "
        f"{', '.join(map(str, report['seeds']))}; "
        f"settings: {settings}.",
        "",
        "Success is in %: every output field equal to the expected value after the answer "
        "normaliser, "
        "ignoring case. `none`, `text` and `plain` are read once; the constructed tensor is the "
        "mean "
        "over seeds with the lowest and highest seed in brackets. The utility pool holds the "
        "domains "
        f"where the procedure as text lifts the authorized model by at least "
        f"{100 * report['usable_uplift_min']:.0f} points. rho is Delta(text) - Delta(tensor) and "
        f"iota is "
        "Delta(text) - Delta(plain), both in points. Repeat RL is the mean over the pool's domains "
        "of "
        "the ROUGE-L of the reader's repetition of the procedure from the tensor.",
        "",
    ]
    if construction != "spass":
        out += [
            f"**Baseline `{construction}`**: the `S-Pass` column holds the baseline's tensor, not "
            f"an S-Pass.",
            "",
        ]
    if report.get("incomplete"):
        out += [f"Not yet complete: {', '.join(report['incomplete'])}.", ""]
    if report.get("excluded_from_pool"):
        out += [
            f"Outside the utility pool (text uplift under the floor): "
            f"{', '.join(report['excluded_from_pool'])}.",
            "",
        ]
    out += [
        "## Authorized model (Table 2)",
        "",
        "| Pool | Domains / tasks | None | Text | Plain | S-Pass | rho (pp) | iota (pp) | Repeat "
        "RL |",
        "|---|---|---|---|---|---|---|---|---|",
        _authorized_row("utility pool", report["pool"], authorized),
        _authorized_row("every domain", report["all"], authorized),
        "",
        "## Reconstruction by the model-free adversary (Table 3)",
        "",
        "| Construction | Domains | Decode RL | Floor RL | Token recovery (rank 0) |",
        "|---|---|---|---|---|",
    ]
    recon = report["reconstruction"]
    out += [
        f"| {construction} | {recon['domains']} | {_num(recon.get('decode_rougeL'))} | "
        f"{_num(recon.get('floor_rougeL'))} | {_num(recon.get('rank0_token_recovery'))} |",
        "",
        "Decode RL is the best nearest-token ROUGE-L against the procedure over the stripping "
        "ranks, "
        "using the authorized model's embedding layer; Floor RL is the decoy's own overlap with "
        "the "
        "procedure. Every complete domain has equal weight.",
        "",
        "## Relatives (Table 4)",
        "",
        "| Relative | None | Text | Plain | S-Pass | Delta(plain) (pp) | Delta(S-Pass) (pp) | r_L "
        "| Repeat RL |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    pool = report["pool"]
    for entry in report["readers"]:
        reader = entry["reader"]
        if reader == authorized:
            continue
        mine = (pool.get("utility") or {}).get(reader)
        if not mine:
            out.append(f"| `{reader}` ({entry['label']}) | n/a | | | | | | | |")
            continue
        s, u = mine["success"], mine["uplift"]
        rep = pool["repeat_rougeL"].get(reader, {}).get("spass_mean_over_domains")
        out.append(
            f"| `{reader}` ({entry['label']}) | {_pct(s['none'])} | {_pct(s['text'])} | "
            f"{_pct(s['plain'])} | "
            f"{_pp(s['spass'])} | {100 * u['plain']:.1f} | {_pp(u['spass'])} | "
            f"{_num(mine['residual_uplift'])} | {_num(rep, 2)} |"
        )
    out += [
        "",
        "Pooled over the utility pool. r_L is Delta(S-Pass) / Delta(plain), a dash where the "
        "relative's "
        "plain uplift is under the floor.",
        "",
        "## Per domain",
        "",
        "| Domain | Tasks | In pool | Text uplift (pp) | Auth. None | Text | Plain | S-Pass | rho "
        "(pp) | "
        "Decode RL | Floor RL |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in report["domains"].items():
        if not d.get("complete"):
            out.append(f"| {name} | | incomplete | | | | | | | | |")
            continue
        a = d["utility"][authorized]
        r = d["reconstruction"]
        out.append(
            f"| {name} | {d['tasks']} | {'yes' if d['in_pool'] else 'no'} | "
            f"{100 * d['text_uplift']:.1f} | "
            f"{_pct(a['success']['none'])} | {_pct(a['success']['text'])} | "
            f"{_pct(a['success']['plain'])} | "
            f"{_pp(a['success']['spass'])} | {_pp(a['rho'])} | "
            f"{_num(r['decode_rougeL']) if r else 'n/a'} | "
            f"{_num(r['floor_rougeL']) if r else 'n/a'} |"
        )
    out.append("")
    return "\n".join(out)


def write_report(config, out: Path) -> dict[str, Any]:
    report = build_report(config, out)
    (Path(out) / REPORT_FILE).write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (Path(out) / TABLE_FILE).write_text(table_markdown(report), encoding="utf-8")
    return report
