"""The proof of concept on synthetic skills (Section 5.2, Table 1).

    uv run python experiments/poc.py --config experiments/poc.yaml \
        --out results/poc --passes passes/poc
        [--phase all|passes|answers|decode|report] [--readers id,id]

Makes one S-Pass per held-out skill and seed with the lineup's authorized model (turned against
every relative in K), reads each skill with every reported reader under `none`, `text`, `plain`
and `spass` with the repeat, decodes each S-Pass with the authorized model's public embedding
table, and writes `report.json` and `table.md` under `--out`. Resumes over the same `--out`.
Needs the gpu group and a CUDA GPU for the `passes` and `answers` phases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skillpass.env import load_dotenv_from_repo_root
from skillpass.poc.run import PHASES, PocError, load_config, run_poc


def main(argv=None) -> int:
    load_dotenv_from_repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the results directory")
    parser.add_argument("--passes", type=Path, required=True, help="where the tensors go")
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--readers", help="comma-separated reader ids, a subset of the lineup's")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except PocError as error:
        print(f"poc: {error}", file=sys.stderr)
        return 2
    from skillpass.attacks.table import decode_table_for
    from skillpass.spass.make import make_many

    log = lambda m: print(m, flush=True)  # noqa: E731
    make = lambda pairs, seed, out: make_many(pairs, seed, out, lineup=config.lineup, log=log)  # noqa: E731
    try:
        report = run_poc(
            config,
            args.out,
            args.passes,
            make=make,
            decode_table=decode_table_for,
            phase=args.phase,
            readers=args.readers.split(",") if args.readers else None,
            log=log,
        )
    except PocError as error:
        print(f"poc: {error}", file=sys.stderr)
        return 2
    if report is not None:
        print(f"poc: wrote {args.out}/report.json and table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
