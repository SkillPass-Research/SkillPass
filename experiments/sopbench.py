"""The SOP-Bench evaluation (Sections 5.3 to 5.5, Tables 2 to 4).

    uv run python experiments/sopbench.py --config experiments/sopbench-<lineup>.yaml --fetch
    uv run python experiments/sopbench.py --config experiments/sopbench-<lineup>.yaml \
        --out results/sopbench/<lineup> --passes passes/sopbench/<lineup> \
        [--phase all|passes|episodes|report] [--domains a,b] [--readers id,id] [--reuse SPASS_OUT]

One config per lineup (`qwen2.5-7b`, `llama-3.1-8b`, `gemma-4-12b`) and, beside each, a
`-offset.yaml` and a `-decoy.yaml` for the two baselines (`construction: offset|decoy`).
`--fetch` clones the benchmark into the config's `sopbench_dir` at its pinned commit and stops.
`passes` makes one S-Pass per domain and seed (the lineup's models loaded to their first block
only), or a baseline's tensors from the public table alone; `episodes` runs each reader through
each domain under the four presentations and the repeats; `report` decodes the tensors and
writes `report.json` and `table.md`. `--reuse SPASS_OUT` copies the S-Pass run's `none`, `text`
and `plain` episodes into a baseline run before its episodes. `--domains` and `--readers` split
a phase into jobs over the same `--out`, one a GPU; every phase skips what is done.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skillpass.env import load_dotenv_from_repo_root
from skillpass.sopbench.runner import PHASES, SopBenchError, fetch


def main(argv=None) -> int:
    load_dotenv_from_repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--fetch", action="store_true", help="fetch the benchmark at the pinned commit"
    )
    parser.add_argument("--out", type=Path, help="the results directory")
    parser.add_argument("--passes", type=Path, help="where the tensors go")
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--domains", help="comma-separated domains, a subset of the config's")
    parser.add_argument("--readers", help="comma-separated reader ids, a subset of the config's")
    parser.add_argument(
        "--reuse",
        type=Path,
        metavar="SPASS_OUT",
        help="copy the none, text and plain episodes of SPASS_OUT into --out first",
    )
    args = parser.parse_args(argv)
    import yaml

    if args.fetch:
        data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        target = Path(str(data["sopbench_dir"])).expanduser()
        fetch(target, data["sopbench_commit"])
        print(f"sopbench: {target} is at {data['sopbench_commit']}")
        return 0
    if args.out is None or args.passes is None:
        parser.error("--out and --passes are required unless --fetch")
    from skillpass.sopbench import runner

    try:
        config = runner.load_config(args.config)
        runner.check_checkout(config)
        domains = args.domains.split(",") if args.domains else None
        readers = args.readers.split(",") if args.readers else None
        if domains and any(d not in config.domains for d in domains):
            raise SopBenchError(f"--domains must be among {config.domains}")
        if readers and any(r not in [e.reader_id for e in config.readers] for r in readers):
            raise SopBenchError("--readers must be among the config's readers")
        runner.open_run(config, args.out, args.phase)
        log = lambda m: print(m, flush=True)  # noqa: E731
        if args.phase in ("all", "passes"):
            if config.construction == "spass":
                from skillpass.spass.make import make_many

                conversations = runner.conversations_for(config, domains)
                make = lambda pairs, seed, out: make_many(  # noqa: E731
                    pairs,
                    seed,
                    out,
                    lineup=config.lineup,
                    conversations=conversations,
                    first_blocks_only=True,
                    log=log,
                )
            else:
                from skillpass.spass.baselines import make_many_baselines

                make = lambda pairs, seed, out: make_many_baselines(  # noqa: E731
                    pairs,
                    seed,
                    out,
                    construction=config.construction,
                    lineup=config.lineup,
                    log=log,
                )
            runner.make_passes(config, args.out, args.passes, make=make, domains=domains, log=log)
        if args.reuse is not None and args.phase in ("all", "episodes"):
            copied = runner.reuse_seedless(config, args.out, args.reuse)
            print(f"sopbench: reused {copied} rows from {args.reuse}", flush=True)
        if args.phase in ("all", "episodes"):
            runner.run_readers(
                config, args.out, args.passes, domains=domains, readers=readers, log=log
            )
        if args.phase in ("all", "report"):
            from skillpass.attacks.table import decode_table_for
            from skillpass.sopbench.report import write_report

            runner.decode_all(
                config,
                args.out,
                args.passes,
                lambda: decode_table_for(config.authorized),
                domains=domains,
                log=log,
            )
            write_report(config, args.out)
            print(f"sopbench: wrote {args.out}/report.json and table.md")
    except SopBenchError as error:
        print(f"sopbench: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
