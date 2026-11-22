r"""Network models.

Wilson-Cowan network model
--------------------------
Each node has an excitatory (E) and an inhibitory (I) population:

    tau_E dE_i = [-E_i + (1 - E_i) S(g u_i^E)] dt + sigma dW_i
    tau_I dI_i = [-I_i + (1 - I_i) S(g u_i^I)] dt + sigma dW'_i

    u_i^E = c_EE E_i - c_EI I_i + P   + K G_i(E)
    u_i^I = c_IE E_i - c_II I_i + P_I + K G_i(I)

with the response function S(u) = beta * tanh(u + gamma u^2) for u > 0 and
S(u) = 0 otherwise, and a response gain ``g`` (1 unless a gain sweep is run).
``G`` is the coupling input selected by a
:class:`~coupling_hoi.config.CouplingSpec` (see :mod:`coupling_hoi.coupling`),
computed from the activity of the same population in the other nodes.
Activities are kept in [0, 1].

The same model and the same local parameters are used for every network;
only the connectome, the drive ``P``, the coupling strength ``K`` and, in the
gain sweep, ``g`` change between simulations.

Integration uses the Euler-Maruyama scheme. Each simulation owns an
independent random generator, so a trajectory depends only on its seed and
parameters, not on which other simulations share the batch.

Linear model
------------
A first-order vector autoregressive process on the connectome,
``x_{t+1} = lambda A_s x_t + sigma eta_t``, with ``A_s`` the connectome divided
by its largest absolute eigenvalue. It contains only pairwise linear
interactions and serves as a Gaussian reference.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import CouplingSpec, IntegrationParams, WilsonCowanParams
from .coupling import CouplingOperator


def response(u: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Bounded, rectified response function S(u)."""
    up = np.maximum(u, 0.0)
    return beta * np.tanh(up + gamma * up * up)


def simulate(operator: CouplingOperator,
             spec: CouplingSpec,
             drive: Sequence[float],
             coupling: Sequence[float],
             seeds: Sequence[int],
             params: WilsonCowanParams = WilsonCowanParams(),
             integration: IntegrationParams = IntegrationParams(),
             gain: Sequence[float] | None = None) -> np.ndarray:
    """Simulate a batch of Wilson-Cowan networks.

    Parameters
    ----------
    operator : precomputed coupling matrices of the (normalised) connectome.
    spec     : coupling form and order, shared by the whole batch.
    drive, coupling, seeds, gain : one value per simulation (gain defaults to 1).

    Returns
    -------
    Array ``(B, n_records, N)`` with the excitatory activity sampled every
    ``record_every`` integration steps after the burn-in period.
    """
    seeds = list(seeds)
    b, n = len(seeds), operator.d.shape[0]
    drive = np.asarray(drive, dtype=float).reshape(-1, 1)
    k = np.asarray(coupling, dtype=float).reshape(-1, 1)
    g = np.ones((b, 1)) if gain is None else np.asarray(gain, dtype=float).reshape(-1, 1)
    if not (len(drive) == len(k) == len(g) == b):
        raise ValueError("drive, coupling, gain and seeds must have the same length")

    p, integ = params, integration
    rngs = [np.random.default_rng(s) for s in seeds]
    e = np.stack([r.uniform(0.0, 1.0, n) for r in rngs]) * 0.05
    i = np.stack([r.uniform(0.0, 1.0, n) for r in rngs]) * 0.05

    dt, step = integ.dt, integ.record_every
    noise_scale = p.sigma * np.sqrt(dt)
    out = np.empty((b, integ.n_records, n), dtype=np.float64)

    def coupling_input(x: np.ndarray) -> np.ndarray:
        return operator.input(x, spec.form, spec.order, spec.fan_ratio)

    for rec in range(integ.n_burn_records + integ.n_records):
        noise = np.stack([r.standard_normal((step, 2, n)) for r in rngs])
        for s in range(step):
            u_e = p.c_ee * e - p.c_ei * i + drive + k * coupling_input(e)
            u_i = p.c_ie * e - p.c_ii * i + p.drive_i + k * coupling_input(i)
            de = dt / p.tau_e * (-e + (1.0 - e) * response(g * u_e, p.beta, p.gamma))
            di = dt / p.tau_i * (-i + (1.0 - i) * response(g * u_i, p.beta, p.gamma))
            e = np.clip(e + de + noise_scale * noise[:, s, 0], 0.0, 1.0)
            i = np.clip(i + di + noise_scale * noise[:, s, 1], 0.0, 1.0)
        if rec >= integ.n_burn_records:
            out[:, rec - integ.n_burn_records] = e
    return out


def simulate_linear(adjacency: np.ndarray, seeds: Sequence[int], n_records: int = 570,
                    n_burn: int = 500, lam: float = 0.85, sigma: float = 0.2) -> np.ndarray:
    """Linear vector autoregressive model on the connectome.

    Returns an array ``(B, n_records, N)``. The noise amplitude does not affect
    the O-information, which is invariant to the scale of the signals.
    """
    a = np.asarray(adjacency, dtype=float)
    a_s = a / np.max(np.abs(np.linalg.eigvals(a)))
    n = a.shape[0]
    out = np.empty((len(seeds), n_records, n))
    for m, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        x = 0.1 * rng.standard_normal(n)
        for t in range(n_burn + n_records):
            x = lam * (a_s @ x) + sigma * rng.standard_normal(n)
            if t >= n_burn:
                out[m, t - n_burn] = x
    return out
