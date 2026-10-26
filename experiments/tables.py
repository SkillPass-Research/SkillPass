"""Assemble the paper's Tables 2, 3 and 4 from the finished SOP-Bench runs.

    uv run python experiments/tables.py --runs results/sopbench --out results/sopbench/tables.md

Reads `<runs>/<lineup>/report.json` for each lineup, and `<runs>/<lineup>-offset/report.json`
and `<runs>/<lineup>-decoy/report.json` where they exist, and writes one Markdown file. A run
that is missing is left out of its row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from skillpass import registry


def _load(path: Path) -> dict[str, Any] | None:
    file = path / "report.json"
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def _pp(figure) -> str:
    return "—" if not figure else f"{100 * figure['mean']:.1f}"


def _num(figure, digits: int = 3) -> str:
    if figure is None:
        return "—"
    return f"{figure['mean']:.{digits}f}" if isinstance(figure, dict) else f"{figure:.{digits}f}"


def assemble(runs: Path, lineups: list[str]) -> str:
    out = [
        "# SOP-Bench results (Sections 5.3 to 5.5)",
        "",
        "## Table 2: authorized-model performance, pooled over domains with text uplift of at "
        "least 20 points",
        "",
        "| Lineup | Domains / tasks | None | Text | Plain | S-Pass | rho (pp) | iota (pp) | Repeat "
        "RL | Offset | Decoy |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    table3 = ["| Lineup | Construction | Decode RL | Floor RL |", "|---|---|---|---|"]
    table4 = [
        "| Lineup | Relative | None | Text | Plain | S-Pass | Delta(plain) (pp) | Delta(S-Pass) "
        "(pp) | r_L | Repeat RL |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in lineups:
        main = _load(runs / name)
        offset = _load(runs / f"{name}-offset")
        decoy = _load(runs / f"{name}-decoy")
        authorized = registry.LINEUPS[name].authorized.reader_id
        if main is None:
            out.append(f"| {authorized} | no run | | | | | | | | | |")
            continue
        pool = main["pool"]
        mine = (pool.get("utility") or {}).get(authorized)
        if mine:
            s = mine["success"]
            rep = pool["repeat_rougeL"].get(authorized, {}).get("spass_mean_over_domains")

            def baseline(run, authorized=authorized):
                m = run and (run["pool"].get("utility") or {}).get(authorized)
                return _pp(m["success"]["spass"]) if m else "—"

            out.append(
                f"| {authorized} | {len(pool['domains'])} / {pool['tasks']} | {_pct(s['none'])} | "
                f"{_pct(s['text'])} | {_pct(s['plain'])} | {_pp(s['spass'])} | {_pp(mine['rho'])} "
                f"| "
                f"{100 * mine['iota']:.1f} | {_num(rep, 2)} | {baseline(offset)} | "
                f"{baseline(decoy)} |"
            )
        else:
            out.append(f"| {authorized} | incomplete | | | | | | | | | |")
        for label, run in (("SkillPass", main), ("Offset", offset), ("Decoy", decoy)):
            if run is None:
                continue
            r = run["reconstruction"]
            table3.append(
                f"| {authorized} | {label} | {_num(r.get('decode_rougeL'))} | "
                f"{_num(r.get('floor_rougeL'))} |"
            )
        for entry in main["readers"]:
            reader = entry["reader"]
            if reader == authorized:
                continue
            m = (pool.get("utility") or {}).get(reader)
            if not m:
                table4.append(
                    f"| {authorized} | {reader} ({entry['label']}) | incomplete | | | | | | | |"
                )
                continue
            s, u = m["success"], m["uplift"]
            rep = pool["repeat_rougeL"].get(reader, {}).get("spass_mean_over_domains")
            table4.append(
                f"| {authorized} | {reader} ({entry['label']}) | {_pct(s['none'])} | "
                f"{_pct(s['text'])} | "
                f"{_pct(s['plain'])} | {_pp(s['spass'])} | {100 * u['plain']:.1f} | "
                f"{_pp(u['spass'])} | "
                f"{_num(m['residual_uplift'])} | {_num(rep, 2)} |"
            )
    out += [
        "",
        "## Table 3: nearest-token reconstruction, all domains at equal weight",
        "",
        *table3,
    ]
    out += [
        "",
        "## Table 4: reuse of S-Passes by relative models, pooled over each lineup's utility pool",
        "",
        *table4,
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, default=Path("results/sopbench"))
    parser.add_argument("--out", type=Path, default=None, help="default <runs>/tables.md")
    parser.add_argument("--lineups", default=",".join(registry.LINEUPS))
    args = parser.parse_args(argv)
    text = assemble(args.runs, args.lineups.split(","))
    out = args.out or (args.runs / "tables.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"tables: wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
