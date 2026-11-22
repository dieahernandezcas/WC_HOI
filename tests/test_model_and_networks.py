import numpy as np
import pytest

from coupling_hoi.config import CouplingSpec, IntegrationParams, stable_seed
from coupling_hoi.coupling import CouplingOperator
from coupling_hoi.model import simulate
from coupling_hoi.networks import (erdos_renyi, in_strength_heterogeneity, normalise,
                                   ring_with_shortcuts, synthetic_networks, three_node_all_to_all)
from coupling_hoi.preprocessing import preprocess

SHORT = IntegrationParams(dt=0.1, record_every=20, n_burn_records=20, n_records=60)


def _op(name="random", scheme="global"):
    return CouplingOperator.from_adjacency(normalise(synthetic_networks()[name].adjacency, scheme))


def test_synthetic_networks_degrees():
    assert np.all(three_node_all_to_all().in_strength == 2)
    assert sorted(set(ring_with_shortcuts().in_strength)) == [4, 5]
    d = erdos_renyi().in_strength
    assert d.min() == 1 and d.max() == 7


def test_normalisation_properties():
    a = erdos_renyi().adjacency
    np.testing.assert_allclose(normalise(a, "row").sum(1), 1.0)
    assert abs(normalise(a, "global").sum(1).mean() - 1.0) < 1e-12
    assert in_strength_heterogeneity(normalise(a, "row")) < 1e-12
    assert abs(in_strength_heterogeneity(normalise(a, "global")) - in_strength_heterogeneity(a)) < 1e-12
    with pytest.raises(ValueError):
        normalise(a, "unknown")


def test_simulation_is_reproducible_and_batch_independent():
    spec = CouplingSpec("diffusive", "pairwise")
    op = _op()
    one = simulate(op, spec, [0.6], [0.8], [123], integration=SHORT)
    batch = simulate(op, spec, [0.3, 0.6, 1.0], [0.1, 0.8, 1.5], [7, 123, 9], integration=SHORT)
    np.testing.assert_array_equal(one[0], batch[1])


def test_activity_is_bounded():
    for order in ("pairwise", "fan"):
        out = simulate(_op(), CouplingSpec("additive", order), [2.0], [3.0], [1], integration=SHORT)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_forms_coincide_without_coupling():
    op = _op()
    add = simulate(op, CouplingSpec("additive", "pairwise"), [0.6], [0.0], [5], integration=SHORT)
    dif = simulate(op, CouplingSpec("diffusive", "pairwise"), [0.6], [0.0], [5], integration=SHORT)
    np.testing.assert_array_equal(add, dif)


def test_stable_seed_is_deterministic_and_distinct():
    assert stable_seed(1, "a", 2) == stable_seed(1, "a", 2)
    assert stable_seed(1, "a", 2) != stable_seed(1, "a", 3)


def test_preprocess_zscores_along_time():
    x = np.random.default_rng(0).normal(3, 2, (4, 570)).cumsum(axis=1)
    y = preprocess(x)
    np.testing.assert_allclose(y.mean(axis=1), 0, atol=1e-8)
    np.testing.assert_allclose(y.std(axis=1), 1, atol=1e-6)
    z = preprocess(x.T, axis=0)
    np.testing.assert_allclose(z.T, y, atol=1e-10)
