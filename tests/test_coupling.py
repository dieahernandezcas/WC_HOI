import numpy as np
import pytest

from coupling_hoi.coupling import CouplingOperator, fan_reference, pairwise_reference
from coupling_hoi.networks import erdos_renyi, normalise


def _random_weighted(n, seed, directed=True):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, (n, n)) * (rng.uniform(0, 1, (n, n)) < 0.5)
    if not directed:
        a = np.triu(a, 1)
        a = a + a.T
    np.fill_diagonal(a, 0.0)
    return a


@pytest.mark.parametrize("form", ["additive", "diffusive"])
@pytest.mark.parametrize("directed", [True, False])
def test_pairwise_matches_reference(form, directed):
    a = _random_weighted(9, 1, directed)
    x = np.random.default_rng(2).uniform(0, 1, 9)
    op = CouplingOperator.from_adjacency(a)
    np.testing.assert_allclose(op.pairwise(x[None], form)[0], pairwise_reference(a, x, form), atol=1e-12)


@pytest.mark.parametrize("form", ["additive", "diffusive"])
@pytest.mark.parametrize("directed", [True, False])
def test_fan_matches_reference(form, directed):
    a = _random_weighted(8, 3, directed)
    x = np.random.default_rng(4).uniform(0, 1, 8)
    op = CouplingOperator.from_adjacency(a)
    np.testing.assert_allclose(op.fan(x[None], form)[0], fan_reference(a, x, form), atol=1e-12)


def test_diffusive_equals_additive_minus_self_term():
    a = _random_weighted(7, 5)
    x = np.random.default_rng(6).uniform(0, 1, (4, 7))
    op = CouplingOperator.from_adjacency(a)
    np.testing.assert_allclose(op.pairwise(x, "diffusive"),
                               op.pairwise(x, "additive") - a.sum(1) * x, atol=1e-12)


def test_fan_expansion_unit_weights():
    """F_dif = F_add - 2 (n_i - 1) x_i sum_j x_j + n_i (n_i - 1) x_i^2 for unit weights."""
    a = erdos_renyi().adjacency
    x = np.random.default_rng(7).uniform(0, 1, a.shape[0])
    op = CouplingOperator.from_adjacency(a)
    n = a.sum(1)
    expected = (op.fan(x[None], "additive")[0] - 2 * (n - 1) * x * (a @ x) + n * (n - 1) * x ** 2)
    np.testing.assert_allclose(op.fan(x[None], "diffusive")[0], expected, atol=1e-12)


def test_diffusive_inputs_vanish_on_synchrony():
    a = _random_weighted(10, 8)
    op = CouplingOperator.from_adjacency(a)
    x = np.full((3, 10), 0.37)
    np.testing.assert_allclose(op.pairwise(x, "diffusive"), 0.0, atol=1e-12)
    np.testing.assert_allclose(op.fan(x, "diffusive"), 0.0, atol=1e-12)


def test_row_normalisation_makes_forms_differ_by_uniform_decay():
    a = normalise(_random_weighted(12, 9), "row")
    op = CouplingOperator.from_adjacency(a)
    x = np.random.default_rng(10).uniform(0, 1, (2, 12))
    np.testing.assert_allclose(op.pairwise(x, "additive") - op.pairwise(x, "diffusive"), x, atol=1e-12)


def test_batched_equals_individual():
    a = _random_weighted(6, 11)
    op = CouplingOperator.from_adjacency(a)
    xs = np.random.default_rng(12).uniform(0, 1, (5, 6))
    for form in ("additive", "diffusive"):
        batched = op.fan(xs, form)
        for b in range(5):
            np.testing.assert_allclose(batched[b], fan_reference(a, xs[b], form), atol=1e-12)
