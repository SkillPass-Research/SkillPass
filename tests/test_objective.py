import numpy as np
import pytest

from skillpass.spass.objective import hinge, reconstruction_loss, relative_error


def test_hinge_is_zero_once_on_the_decoy_side_by_the_margin():
    rng = np.random.default_rng(0)
    skill = rng.standard_normal((5, 8))
    decoy = rng.standard_normal((5, 8))
    margins = np.full(5, 0.5)
    alpha = 0.3
    # at the decoy: nearer the decoy than the skill by far more than alpha * mu
    assert np.all(hinge(decoy, skill, decoy, margins, alpha) == 0.0)
    # at the skill: the penalty is the normalised shortfall, positive
    at_skill = hinge(skill, skill, decoy, margins, alpha)
    assert np.all(at_skill > 0)
    expected = (np.linalg.norm(skill - decoy, axis=1) + alpha * margins) / margins
    assert at_skill == pytest.approx(expected)
    assert reconstruction_loss(skill, skill, decoy, margins, alpha) == pytest.approx(
        expected.mean()
    )


def test_hinge_boundary_is_exactly_alpha_margins():
    skill = np.zeros((1, 2))
    decoy = np.array([[2.0, 0.0]])
    margins = np.array([1.0])
    # a point at x is at distance x from the skill and 2 - x from the decoy;
    # the hinge vanishes when (2 - x) - x + alpha <= 0, i.e. x >= 1 + alpha / 2
    alpha = 0.4
    assert hinge(np.array([[1.2, 0.0]]), skill, decoy, margins, alpha)[0] == pytest.approx(0.0)
    assert hinge(np.array([[1.1, 0.0]]), skill, decoy, margins, alpha)[0] == pytest.approx(0.2)


def test_relative_error_shape():
    target = np.ones((3, 4))
    assert relative_error(target, target) == 0.0
    assert relative_error(2 * target, target) == pytest.approx(1.0)
