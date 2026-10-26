import pytest

from skillpass.metrics import cer, iota, normalise, residual_uplift, rho, rouge_l, spread, uplift


def test_normalise_collapses_whitespace_and_keeps_case():
    assert normalise("  A\n\tb   c ") == "A b c"


def test_cer_zero_on_itself_and_counts_edits():
    assert cer("hello world", "hello  world") == 0.0
    assert cer("abcd", "abxd") == pytest.approx(0.25)
    assert cer("ab", "abcdef") == 2.0  # unbounded above
    with pytest.raises(ValueError):
        cer("   ", "x")


def test_rouge_l_is_f1_over_words():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("the cat sat", "dog") == 0.0
    # LCS "the sat" = 2 words; precision 2/2, recall 2/3 -> F1 0.8
    assert rouge_l("the cat sat", "the sat") == pytest.approx(0.8)


def test_uplift_rho_iota_and_residual():
    assert uplift(0.8, 0.1) == pytest.approx(0.7)
    assert rho(0.7, 0.6) == pytest.approx(0.1)
    assert iota(0.7, 0.68) == pytest.approx(0.02)
    assert residual_uplift(0.02, 0.4) == pytest.approx(0.05)
    assert residual_uplift(0.02, 0.1) is None  # plain uplift under the 20-point floor
    assert residual_uplift(-0.02, 0.4) < 0  # the S-Pass left the relative worse than no skill


def test_spread_reports_mean_min_max_and_skips_none():
    s = spread([0.2, None, 0.4])
    assert s["mean"] == pytest.approx(0.3)
    assert (s["min"], s["max"]) == (0.2, 0.4)
    assert s["per_seed"] == [0.2, None, 0.4]
    assert spread([None]) is None
