"""The model-free adversary on toy tensors: what it recovers from the plain embeddings, the
`offset` baseline (Section 4.2: distance alone hides nothing) and the `decoy` baseline."""

import numpy as np

from skillpass.attacks.decode import decode_report, nearest_token, strip
from skillpass.spass.baselines import decoy_tensor, margins, offset_tensor
from skillpass.spass.decoy import align, decoy_ids

SKILL = "Open every session with twf_greet; it returns a ferry_sigil."
DECOY = "Round a vell to the nearest half-span before writing it down."


def test_nearest_token_recovers_the_table_rows(toy):
    ids = np.asarray(toy.tokenise(SKILL))
    assert nearest_token(toy.table, toy.table[ids]).tolist() == ids.tolist()


def test_strip_rank_zero_is_identity_and_higher_ranks_remove_directions(toy):
    vectors = toy.table[np.asarray(toy.tokenise(SKILL))]
    assert np.allclose(strip(vectors, 0), vectors)
    stripped = strip(vectors, 2)
    centred = vectors - vectors.mean(axis=0)
    _, _, vh = np.linalg.svd(centred.astype(np.float64), full_matrices=False)
    assert np.allclose((stripped - vectors.mean(axis=0)) @ vh[:2].T, 0.0, atol=1e-6)


def test_align_cycles_or_cuts_the_decoy():
    assert align([1, 2, 3], 7).tolist() == [1, 2, 3, 1, 2, 3, 1]
    assert align([1, 2, 3], 2).tolist() == [1, 2]


def test_offset_baseline_decodes_to_the_skill_in_high_dimension():
    # Section 4.2: a random step of a few margins almost never changes the nearest token.
    rng = np.random.default_rng(1)
    table = rng.standard_normal((400, 512)).astype(np.float32)
    ids = rng.integers(0, 400, size=60)
    plain = table[ids]
    moved = offset_tensor(plain, margins(table, ids), 2.2, seed=3)
    recovered = nearest_token(table, moved)
    assert np.mean(recovered == ids) > 0.95


def test_decoy_baseline_reads_as_the_decoy_and_scores_the_floor(toy):
    ids = np.asarray(toy.tokenise(SKILL))
    dec = decoy_ids(toy.tokenise, DECOY, len(ids))
    report = decode_report(
        toy.table,
        decoy_tensor(toy.table, dec),
        ids,
        dec,
        detokenise=toy.detokenise,
        skill_text=SKILL,
    )
    rank0 = report["strip"][0]
    assert rank0["rank"] == 0 and rank0["applicable"]
    assert rank0["decoy_recovery"] == 1.0
    assert rank0["readout"] == toy.detokenise(dec.tolist())
    assert report["best_rougeL_skill"] >= report["floor_rougeL"]  # at rank 0 it is the floor itself
    assert rank0["rougeL_skill"] == report["floor_rougeL"]
    assert all(r["rank"] in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256) for r in report["strip"])
    # a rank at or past the available directions is marked not applicable
    assert any(not r["applicable"] for r in report["strip"])


def test_plain_embeddings_decode_in_full(toy):
    ids = np.asarray(toy.tokenise(SKILL))
    dec = decoy_ids(toy.tokenise, DECOY, len(ids))
    report = decode_report(
        toy.table, toy.table[ids], ids, dec, detokenise=toy.detokenise, skill_text=SKILL
    )
    assert report["strip"][0]["token_recovery"] == 1.0
    assert report["strip"][0]["rougeL_skill"] == 1.0
    assert report["strip"][0]["decode_cer"] == 0.0
