import numpy as np

from coupling_hoi.information import (entropy_kl, omega_gaussian, omega_gaussian_copula,
                                      omega_ksg, omega_triplets, omega_triplets_copula)


def _gaussian_samples(cov, t, seed):
    return np.random.default_rng(seed).multivariate_normal(np.zeros(len(cov)), cov, size=t)


def test_kl_entropy_of_standard_normal():
    x = np.random.default_rng(0).standard_normal((20000, 1))
    assert abs(entropy_kl(x) - 0.5 * np.log(2 * np.pi * np.e)) < 0.02


def test_kl_entropy_of_uniform_square():
    x = np.random.default_rng(1).uniform(0, 1, (20000, 2))
    assert abs(entropy_kl(x)) < 0.03


def test_omega_ksg_close_to_gaussian_closed_form():
    cov = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.55], [0.5, 0.55, 1.0]])
    x = _gaussian_samples(cov, 20000, 2)
    assert abs(omega_ksg(x) - omega_gaussian(x)) < 0.02
    assert omega_gaussian(x) > 0


def test_omega_sign_for_synergistic_gaussian():
    """Two independent variables and their sum: a synergy-dominated triplet."""
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(20000), rng.standard_normal(20000)
    x = np.c_[a, b, a + b + 0.1 * rng.standard_normal(20000)]
    assert omega_gaussian(x) < 0
    assert omega_ksg(x) < 0


def test_omega_scale_invariance_requires_standardisation():
    """Omega itself is scale invariant, but the maximum-norm nearest-neighbour
    estimator is not: variables must be standardised before estimation, as the
    pipeline does. After z-scoring, rescaled data give identical estimates."""
    cov = np.array([[1.0, 0.4, 0.3], [0.4, 1.0, 0.2], [0.3, 0.2, 1.0]])
    x = _gaussian_samples(cov, 3000, 4)
    y = x * np.array([0.01, 5.0, 300.0])
    z = lambda v: (v - v.mean(0)) / v.std(0)
    assert abs(omega_ksg(z(x)) - omega_ksg(z(y))) < 1e-9
    assert abs(omega_ksg(x) - omega_ksg(y)) > 0.1


def test_omega_independent_variables_near_zero():
    x = np.random.default_rng(5).standard_normal((20000, 3))
    assert abs(omega_ksg(x)) < 0.02
    assert abs(omega_gaussian(x)) < 0.01


def test_triplet_cache_matches_direct_computation():
    x = np.random.default_rng(6).standard_normal((570, 8)) @ np.random.default_rng(7).normal(size=(8, 8))
    triplets = np.array([[0, 1, 2], [0, 1, 3], [2, 5, 7], [1, 3, 6]])
    cached = omega_triplets(x, triplets)
    direct = np.array([omega_ksg(x[:, t]) for t in triplets])
    np.testing.assert_allclose(cached, direct, atol=1e-12)


def test_triplet_copula_matches_direct_computation():
    x = np.random.default_rng(8).standard_normal((570, 6)) @ np.random.default_rng(9).normal(size=(6, 6))
    triplets = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 4]])
    np.testing.assert_allclose(omega_triplets_copula(x, triplets),
                               [omega_gaussian_copula(x[:, t]) for t in triplets], atol=1e-10)
