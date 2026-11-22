import numpy as np
import pytest

from coupling_hoi.config import CouplingSpec, IntegrationParams
from coupling_hoi.coupling import CouplingOperator
from coupling_hoi.experiment import GridPoint, condition_seed, deterministic_amplitude
from coupling_hoi.model import simulate, simulate_linear
from coupling_hoi.networks import normalise, synthetic_networks

SHORT = IntegrationParams(dt=0.1, record_every=20, n_burn_records=50, n_records=60)


def test_pairwise_plus_fan_is_sum_of_terms():
    a = normalise(synthetic_networks()["random"].adjacency, "global")
    op = CouplingOperator.from_adjacency(a)
    x = np.random.default_rng(0).uniform(0, 0.1, (3, 10))
    for form in ("additive", "diffusive"):
        np.testing.assert_allclose(op.input(x, form, "pairwise+fan", 20.0),
                                   op.pairwise(x, form) + 20.0 * op.fan(x, form), atol=1e-14)


def test_unknown_order_rejected():
    with pytest.raises(ValueError):
        CouplingSpec("additive", "triangle")


def test_unit_gain_equals_default():
    op = CouplingOperator.from_adjacency(normalise(synthetic_networks()["ring"].adjacency, "global"))
    spec = CouplingSpec("diffusive", "pairwise+fan")
    a = simulate(op, spec, [0.2], [4.0], [3], integration=SHORT)
    b = simulate(op, spec, [0.2], [4.0], [3], integration=SHORT, gain=[1.0])
    np.testing.assert_array_equal(a, b)


def test_seed_independent_of_form_and_dependent_on_parameters():
    p = GridPoint(0.2, 4.0, 1)
    s_add = condition_seed(1, "ring", "global", CouplingSpec("additive", "pairwise"), p)
    s_dif = condition_seed(1, "ring", "global", CouplingSpec("diffusive", "pairwise"), p)
    assert s_add == s_dif
    assert s_add != condition_seed(1, "ring", "global", CouplingSpec("additive", "pairwise"),
                                   GridPoint(0.2, 4.0, 2))


def test_deterministic_network_oscillates_only_at_intermediate_drive():
    """Without coupling, nodes rest at low drive, oscillate at intermediate
    drive and saturate at a fixed point at high drive."""
    a = synthetic_networks()["three_node"].adjacency
    pts = [GridPoint(p, 0.0, 0) for p in (0.0, 0.2, 0.6)]
    amp = deterministic_amplitude(a, "global", CouplingSpec("additive", "pairwise"), pts,
                                  integration=SHORT)
    assert amp[0] < 1e-6 and amp[2] < 1e-6 and amp[1] > 1e-3


def test_linear_model_shapes_and_reproducibility():
    a = normalise(synthetic_networks()["random"].adjacency, "global")
    x1 = simulate_linear(a, [4, 5], n_records=50, n_burn=20)
    x2 = simulate_linear(a, [4], n_records=50, n_burn=20)
    assert x1.shape == (2, 50, 10)
    np.testing.assert_array_equal(x1[0], x2[0])


def test_circular_shift_preserves_marginals_and_breaks_alignment():
    from coupling_hoi.surrogates import circular_shift
    x = np.random.default_rng(0).standard_normal((570, 5)).cumsum(axis=0)
    y = circular_shift(x, seed=1)
    np.testing.assert_allclose(np.sort(x, axis=0), np.sort(y, axis=0))
    assert not np.allclose(x, y)
    np.testing.assert_array_equal(y, circular_shift(x, seed=1))
