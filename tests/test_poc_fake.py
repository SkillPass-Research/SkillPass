"""The proof-of-concept pipeline end to end on fake readers and a toy table: the passes, the
answers, the decode and the table, with resume. Never evidence; it checks the plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skillpass import registry
from skillpass.models import FAKE_FAMILY, FakeReader
from skillpass.poc.run import LOG_FILE, PocConfig, read_log, run_poc
from skillpass.skills import SUITE_ROOT, load_suite
from skillpass.spass.decoy import decoy_ids
from skillpass.spass.file import write_plain, write_spass
from tests.conftest import toy_table

AUTH = registry.Reader(registry.Role.AUTHORIZED, "fake/authorized", FAKE_FAMILY, label="authorized")
BASE = registry.Reader(registry.Role.RELATIVE, "fake/base", FAKE_FAMILY, label="base")
LINEUP = registry.Lineup("fake", AUTH, (BASE,))

SKILL_OPENING = "The Thistlewane Ferry Relay is the internal service"
SCRIPTS = {
    # the authorized model answers `twf_greet` whenever it is given a skill, by text or by an
    # injected tensor, and nothing right without one; its repetition opens as the skill does
    "fake/authorized": {
        "Repeat the skill document above": "# Thistlewane Ferry Relay\n\n" + SKILL_OPENING,
        "<spass>": "twf_greet",
        SKILL_OPENING: "twf_greet",
        "default": "?",
    },
    "fake/base": {"default": "no idea"},
}


def toy_maker(table):
    """Writes the decoy baseline as if it were an S-Pass: the plain tensor beside it."""

    def make(pairs, seed, out: Path):
        for skill, decoy in pairs:
            ids = np.asarray(table.tokenise(skill.source))
            dec = decoy_ids(table.tokenise, decoy.source, len(ids))
            sidecar = {
                "construction": "spass",
                "skill": skill.name,
                "model": AUTH.reader_id,
                "revision": None,
                "tokens": len(ids),
                "width": table.table.shape[1],
                "seed": seed,
                "decoy": decoy.name,
                "source_sha256": skill.source_sha256,
            }
            write_spass(out / skill.name, table.table[dec], sidecar)
            write_plain(out / skill.name, table.table[ids])

    return make


def test_poc_runs_end_to_end_and_resumes(tmp_path):
    table = toy_table()
    suite = {s.name: s for s in load_suite(SUITE_ROOT)}
    config = PocConfig(
        path=Path("poc.yaml"),
        sha256="0" * 64,
        lineup=LINEUP,
        skills=(suite["fictional-api"], suite["vell-units"]),
        decoys={"fictional-api": suite["vell-units"], "vell-units": suite["fictional-api"]},
        seeds=(1, 2),
    )
    out, passes = tmp_path / "out", tmp_path / "passes"
    factory = lambda entry: FakeReader(entry, SCRIPTS[entry.reader_id])  # noqa: E731
    report = run_poc(
        config,
        out,
        passes,
        make=toy_maker(table),
        decode_table=lambda entry: table,
        reader_factory=factory,
        log=lambda m: None,
    )
    rows = read_log(out)
    # 2 readers x 2 skills x 10 tasks x (3 seedless + 2 seeds) answers,
    # and 2 readers x 2 skills x 3 repeats
    assert sum(r["job"] == "answer" for r in rows) == 2 * 2 * 10 * 5
    assert sum(r["job"] == "repeat" for r in rows) == 2 * 2 * 3
    auth = report["skills"]["fictional-api"]["utility"]["fake/authorized"]
    assert auth["success"]["none"] == 0.0
    assert auth["success"]["text"] == 0.1 and auth["success"]["plain"] == 0.1
    assert auth["success"]["spass"]["mean"] == 0.1
    assert auth["rho"]["mean"] == 0.0
    assert auth["repeat_rougeL"]["spass"]["mean"] > 0
    base = report["skills"]["fictional-api"]["utility"]["fake/base"]
    assert base["residual_uplift"] is None  # plain uplift under the floor
    recon = report["skills"]["fictional-api"]["reconstruction"]
    assert recon["decode_rougeL"]["mean"] >= recon["floor_rougeL"]["mean"]
    assert report["pooled"]["tasks"] == 20
    assert (out / "table.md").read_text().startswith("# Proof of concept")
    assert json.loads((out / "report.json").read_text())["fake"] is True
    # a second run over the same directory adds nothing
    run_poc(
        config,
        out,
        passes,
        make=toy_maker(table),
        decode_table=lambda entry: table,
        reader_factory=factory,
        log=lambda m: None,
    )
    assert len(read_log(out)) == len(rows)
    assert (out / LOG_FILE).is_file()
