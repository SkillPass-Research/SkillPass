# SkillPass

Confidential sharing of agent skills: a skill document is released as an **S-Pass**, a tensor
in one authorized language model's input space that takes the place of the skill's tokens.
The authorized model follows the skill from it; nearest-token decoding reads an unrelated
decoy document; the model's public relatives read the decoy too. This repository is a focused
implementation of the paper's approach (Section 3) and of its main experiments: the proof of
concept on synthetic skills (Section 5.2), and the utility, reconstruction and transfer
evaluation on Amazon's SOP-Bench (Sections 5.3 to 5.5) with the two defence baselines.

## The construction

For a skill of `m` tokens with embeddings `e*` in the authorized model `L*`, and a decoy skill
aligned to the same length, the S-Pass `z` is optimised with Adam, every model frozen, against

```
L(z) = L_util(z) + lambda * L_recon(z) + beta * sum over relatives L of L_trans^L(z)
```

- **Workflow fidelity, `L_util`** (Equation 2): the relative squared error between `L*`'s
  hidden states after block `k` on `z` and on the skill, from the skill's position to the end of
  the prompt, averaged over a set of prompt contexts: the reading instruction before the skill
  and, after it, four generic probes (an empty continuation, a summary request, a question about
  the text, the repeat request); for SOP-Bench, also the agent's own system message with its
  tools and three synthetic agent conversations.
- **Content confidentiality, `L_recon`** (Equation 4): a hinge per position, zero once the
  vector is nearer the decoy's token than the skill's by `alpha` times the position's
  nearest-neighbour margin `mu_i`, normalised by `mu_i`.
- **Model specificity, `L_trans`** (Equation 5): for each known relative `L` of the same
  family and width, the same relative error between `L`'s hidden states on `z` and on its own
  embeddings of the decoy.

Optimisation starts at the decoy's embeddings displaced by a random vector of `0.1 mu_i`.
The settings are the paper's and live in [skillpass/registry.py](skillpass/registry.py):

| Setting | Value | Constant |
|---|---|---|
| matched block `k` | 1 | `MATCH_BLOCK` |
| Adam steps | 600 | `STEPS` |
| reconstruction weight `lambda` | 3 | `LAMBDA` |
| margin share `alpha` | 0.3 | `ALPHA` |
| transfer weight `beta` | 0.3 | `BETA` |
| Adam step, as a share of the per-entry RMS of `e*` | 0.05 | `LR` |
| start displacement, in margins | 0.1 | `START_OFFSET` |

At deployment the S-Pass replaces the skill text at the same input positions; the task and
every later turn arrive as text through the ordinary interface (`plain` and `spass` use exactly
the `text` presentation's prompt with the tensor spliced in where the skill stood).

## Layout

```
skillpass/
  registry.py        the three lineups (authorized model + relatives), presentations, settings
  prompts.py         the reading instruction, the repeat prompt, the token budgets
  metrics.py         uplift, rho, iota, residual uplift r_L, ROUGE-L, CER
  skills.py          the synthetic skill suite's loader and task checkers
  models.py          every model behind one interface: token path, injection path, chat path
  spass/             the construction: embed, decoy, objective, optimise, make, file, baselines
  attacks/           the model-free adversary: the public table (table) and the decode (decode)
  poc/run.py         Section 5.2: passes, answers, decode, report (Table 1)
  sopbench/          Sections 5.3 to 5.5: domain, agent, runner, report (Tables 2 to 4)
experiments/         the entry points and one YAML config per run
skills/synthetic/    the three held-out synthetic skills with their tasks
tests/               CPU tests on toy tables and fake readers
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev          # the CPU pieces and the tests
uv sync --group dev --group gpu   # torch, transformers: making S-Passes and running models
```

The Llama and Gemma checkpoints are gated: accept each licence on the Hub and put a read token
in `.env` (`cp .env.example .env`) or export `HF_TOKEN`. Every model goes to `cuda` in
bfloat16 by default; `SKILLPASS_DEVICE_MAP=auto` spreads a large model over the visible GPUs
and `SKILLPASS_DEVICE_MAP=cpu` runs a small model in float32 for a smoke test. Checks:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

The tests cover the hinge, the nearest-token and subspace-stripping attacks on toy tables
(including the Section 4.2 result that the `offset` baseline decodes in full), the file format,
the skill checkers, the SOP-Bench answer parsing, and the whole proof-of-concept pipeline on
fake readers. Nothing in them is evidence.

## Models

Each lineup is an authorized model `L*` with its known relatives `K`, public checkpoints of
the same family and width; the tables report the base and previous-generation relatives `L`.

| Lineup | Authorized | Base | Previous generation | In K only |
|---|---|---|---|---|
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-7B | Qwen/Qwen2-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct |
| `llama-3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B | meta-llama/Meta-Llama-3-8B-Instruct | |
| `gemma-4-12b` | google/gemma-4-12B-it | google/gemma-4-12B | google/gemma-3-12b-it | |

Every checkpoint is pinned to a Hub revision in the registry. A base checkpoint borrows the
instruction model's chat template. Gemma scales its embeddings inside the model, so its
S-Passes, `plain` tensors and decode table are all in those scaled coordinates.

## Running the experiments

Every command resumes over the same `--out`: a phase skips what is already there. Tensors go
to `--passes` and are never committed (`passes/` is gitignored); they regenerate from the seed.

### 1. Proof of concept on synthetic skills (Section 5.2, Table 1)

Three held-out synthetic skills (`fictional-api`, `vell-units`, `cloudloom-stilling`), each
the next one's decoy, seeds 1, 2 and 3, with the `qwen2.5-7b` lineup
([experiments/poc.yaml](experiments/poc.yaml)). One command on a machine with one GPU:

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc
```

Or phase by phase, so the readers can be split over GPUs:

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc --phase passes
```

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc --phase answers --readers Qwen/Qwen2.5-7B-Instruct
```

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc --phase answers --readers Qwen/Qwen2.5-7B,Qwen/Qwen2-7B-Instruct
```

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc --phase decode
```

```bash
uv run python experiments/poc.py --config experiments/poc.yaml --out results/poc --passes passes/poc --phase report
```

`passes` makes one S-Pass per skill and seed, the authorized model and every relative in `K`
loaded once per seed (their first block only). `answers` has every reported reader answer the
ten tasks of each skill under `none`, `text`, `plain` and `spass` (one S-Pass per seed), then
repeat the skill from `plain` and from each S-Pass. `decode` reads each S-Pass with the
authorized model's public embedding table by nearest token at every stripping rank. `report`
writes `results/poc/table.md` and `report.json`: per skill and pooled over the 30 tasks, each
reader's success under the four presentations, `rho` for the authorized model, `r_L` for each
relative, Repeat RL, Decode RL and Floor RL.

### 2. SOP-Bench (Sections 5.3 to 5.5, Tables 2 to 4)

The benchmark's data is not in this repository. Fetch it once at the pinned commit into the
configs' `sopbench_dir` (`../sop-bench`, beside this checkout):

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --fetch
```

Then one run per lineup, each with its own config, `--out` and `--passes`. The whole run in one
command (the passes, every reader's episodes and repeats, the decode and the tables):

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b
```

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-llama-3.1-8b.yaml --out results/sopbench/llama-3.1-8b --passes passes/sopbench/llama-3.1-8b
```

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-gemma-4-12b.yaml --out results/sopbench/gemma-4-12b --passes passes/sopbench/gemma-4-12b
```

Or phase by phase, one job per GPU over the same `--out` (jobs sharing a directory take its
lock to write). For the `qwen2.5-7b` lineup:

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b --phase passes
```

```bash
CUDA_VISIBLE_DEVICES=0 uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b --phase episodes --readers Qwen/Qwen2.5-7B-Instruct
```

```bash
CUDA_VISIBLE_DEVICES=1 uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b --phase episodes --readers Qwen/Qwen2.5-7B
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b --phase episodes --readers Qwen/Qwen2-7B-Instruct
```

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b.yaml --out results/sopbench/qwen2.5-7b --passes passes/sopbench/qwen2.5-7b --phase report
```

`--domains a,b` splits a phase further (the `passes` phase is the long one: 13 domains, three
seeds, each S-Pass fitted under the four probes and the domain's agent conversations). The
other lineups use the same commands with their config, their `--out`, their `--passes` and
their reader ids from the table above.

What each phase measures:

- **Utility (Section 5.3, Table 2).** Every case runs through the tool-calling agent for at
  most ten turns of 1,024 tokens, with the procedure delivered at the first turn as text, as its
  own embeddings (`plain`), as the S-Pass, or not at all. Success is every expected output
  field equal to its value. The report pools the domains where the procedure as text lifts the
  authorized model by at least 20 points (`USABLE_UPLIFT_MIN`), tasks concatenated, and gives
  `rho`, the injection cost `iota`, and Repeat RL. A pool over every domain sits beside it.
- **Reconstruction (Section 5.4, Table 3).** Each S-Pass is decoded with the authorized model's
  embedding layer, fetched from the checkpoint's shards without loading the model, by nearest
  token and after stripping 1 to 256 leading principal directions; Decode RL is the best ROUGE-L
  against the procedure over the ranks, Floor RL the decoy's own overlap, every domain at equal
  weight. Every readout is stored as text under `decode/<domain>/<seed>.json`.
- **Transfer (Section 5.5, Table 4).** The base and previous-generation relatives run the same
  episodes; the report gives their success under the four presentations, their uplift from
  `plain` and from the S-Pass, `r_L` (a dash where the plain uplift is under 20 points), and
  their Repeat RL, over the same pool.

**Defence baselines (Section 5.1).** Beside each lineup's config, `-offset.yaml` moves each
skill embedding 2.2 margins along a random direction (three seeds) and `-decoy.yaml` supplies
the decoy's embeddings unchanged (one seed). Their `passes` phase needs no model, and
`--reuse` copies the S-Pass run's `none`, `text` and `plain` episodes, which no tensor enters,
so only the baseline's own episodes run:

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b-offset.yaml --out results/sopbench/qwen2.5-7b-offset --passes passes/sopbench/qwen2.5-7b-offset --phase passes
```

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b-offset.yaml --out results/sopbench/qwen2.5-7b-offset --passes passes/sopbench/qwen2.5-7b-offset --phase episodes --reuse results/sopbench/qwen2.5-7b
```

```bash
uv run python experiments/sopbench.py --config experiments/sopbench-qwen2.5-7b-offset.yaml --out results/sopbench/qwen2.5-7b-offset --passes passes/sopbench/qwen2.5-7b-offset --phase report
```

The same three commands with `-decoy` in place of `-offset` run the decoy baseline, and the
same with `llama-3.1-8b` or `gemma-4-12b` in place of `qwen2.5-7b` run the other lineups'.

**The paper's tables.** Once the runs are done, assemble Tables 2, 3 and 4 across the lineups
and constructions:

```bash
uv run python experiments/tables.py --runs results/sopbench --out results/sopbench/tables.md
```

### What a run writes

- `manifest.json`: the config's sha256, the lineup, the construction, the settings, each
  tensor's sha256 and sidecar, each reader's identity, each session.
- `answers.jsonl`: every answer or episode (with its expected and final answers, success,
  SOP-Bench's lenient match, the turns, the tool calls and the whole trajectory) and every
  repetition, appended as it happens; a rerun skips what is there.
- `passes/<seed>/<skill>/spass.json`: each S-Pass's sidecar (the settings, the final terms, the
  nearest-token decode CER of the written tensor against the skill and the decoy).
- `decode/<skill or domain>/<seed>.json`: the model-free readouts at every rank.
- `report.json` and `table.md`: the figures.

## Making one S-Pass in code

```python
from skillpass import registry
from skillpass.models import make_reader
from skillpass.skills import SUITE_ROOT, load_skill
from skillpass.spass.make import make_spass

lineup = registry.lineup("qwen2.5-7b")
authorized = make_reader(lineup.authorized, blocks=registry.MATCH_BLOCK)
relatives = [make_reader(r, blocks=registry.MATCH_BLOCK) for r in lineup.relatives]
made = make_spass(
    authorized,
    load_skill(SUITE_ROOT / "fictional-api"),
    load_skill(SUITE_ROOT / "vell-units"),  # the decoy
    "passes/example/fictional-api",
    relatives=relatives,
    seed=1,
    log=print,
)
print(made.sidecar["util"], made.sidecar["trans"], made.sidecar["decode_cer"])
```

The directory then holds `spass.safetensors` (float16, `(tokens, width)`), `plain.safetensors`
(the skill's own embeddings) and `spass.json`. A reader shows it to the model with
`read_embeds(tensor, prompt, skill_text)` or, inside a conversation, `chat_many`.

## Not included

The hyperparameter search (the values above are fixed), the adaptive reconstruction adversary
(Section 5.6), the ablations (Section 5.7), the second benchmark (SOPBench, Li et al.) and the
72B lineup are left out of this implementation.
