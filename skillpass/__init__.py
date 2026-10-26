"""SkillPass: agent skills as model-bound continuous representations (S-Passes).

An S-Pass is a tensor in the authorized model's input space that takes the place of a skill
document's tokens. It is optimised, with every model frozen, so that the authorized model's
early hidden states match the skill's (workflow fidelity), nearest-token decoding reads an
unrelated decoy document (content confidentiality), and known relatives of the authorized
model compute the decoy's hidden states from it (model specificity). See the paper, Section 3.

- `registry`: the model lineups, the presentations and the construction's settings.
- `spass/`: the construction (embeddings and margins, the decoy, the objective, Adam, the
  on-disk format) and the two defence baselines.
- `models`: every reader behind one interface, with the injection path.
- `attacks/`: the model-free adversary (nearest-token decoding, subspace stripping) and the
  public embedding table it holds.
- `poc/`: the proof of concept on synthetic skills (Section 5.2).
- `sopbench/`: the SOP-Bench evaluation (Sections 5.3 to 5.5).
"""

__version__ = "0.1.0"
