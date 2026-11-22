# Coupling functions and higher-order interactions

Simulation and analysis code for a study of how the coupling function of a
network model (additive or diffusive; pairwise or third-order) shapes the
higher-order statistical dependencies of its activity, measured with the
O-information, and how such models compare with whole-brain calcium
recordings of larval zebrafish.

A single Wilson–Cowan network model is used throughout. Synthetic networks
and the zebrafish connectome differ only in their adjacency matrix.

## Layout

```
coupling_hoi_study/
├── configs/                  experiment descriptions (JSON); every parameter lives here
├── data/raw/zebrafish/       recordings and connectome (not tracked; see Data)
├── src/coupling_hoi/         library
│   ├── config.py             parameter dataclasses, seeds
│   ├── networks.py           synthetic networks, connectome normalisation
│   ├── coupling.py           additive/diffusive, pairwise/fan coupling inputs
│   ├── model.py              Wilson–Cowan network model (batched Euler–Maruyama)
│   ├── information.py        O-information (nearest-neighbour and Gaussian-copula estimators)
│   ├── preprocessing.py      filtering, detrending, z-scoring
│   ├── surrogates.py         covariance-preserving and phase-randomised surrogates
│   ├── metrics.py            functional connectivity and summaries
│   ├── data.py               zebrafish loading, region selection, triplet sampling
│   └── experiment.py         grids of simulations and per-run summaries
├── scripts/                  numbered entry points (run in order)
├── tests/                    unit tests
└── results/                  generated outputs (figures, arrays, logs)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest            # all tests should pass
```

Python ≥ 3.10, NumPy, SciPy and Matplotlib are required.

## Reproducing the results

```bash
python scripts/01_analyse_zebrafish_data.py               # O-information of recordings and surrogates
python scripts/02_explore_dynamical_regimes.py            # exploratory regime maps
python scripts/03_plot_regimes.py                         # exploratory regime figures
python scripts/04_run_main_experiment.py --part synthetic_drive  --workers 4
python scripts/04_run_main_experiment.py --part synthetic_gain   --workers 4
python scripts/04_run_main_experiment.py --part zebrafish_scan   --workers 4
python scripts/04_run_main_experiment.py --part zebrafish_select
python scripts/04_run_main_experiment.py --part zebrafish_final  --workers 4
python scripts/06_zebrafish_null_correction.py            # circular-shift null for data and models
python scripts/05_figure_synthetic_sweeps.py              # sweep figures
python scripts/07_figure_zebrafish.py                     # zebrafish figure (global and row)
python scripts/08_figure_methodology.py                   # overview figure
```

Every output of a script is written under `results/`; figure scripts only
read those files. Per-run summaries (triplet O-information, functional
connectivity, activity statistics, noise seed) are stored for every
simulation; raw signals are stored for the final zebrafish simulations, and
any other run can be regenerated exactly from its seed.

Scripts skip outputs that already exist, so an interrupted run can be resumed
by calling the same command again.

## Model

Each node has an excitatory (E) and an inhibitory (I) population,

```
tau_E dE_i = [-E_i + (1 - E_i) S(u_i^E)] dt + sigma dW_i
tau_I dI_i = [-I_i + (1 - I_i) S(u_i^I)] dt + sigma dW'_i
u_i^E = c_EE E_i - c_EI I_i + P + K G_i(E)
u_i^I = c_IE E_i - c_II I_i     + K G_i(I)
S(u)  = beta tanh(u + gamma u^2) for u > 0, 0 otherwise
```

`G` is one of four coupling inputs (`coupling.py`):

| | additive | diffusive |
|---|---|---|
| pairwise | `sum_j A_ij x_j` | `sum_j A_ij (x_j - x_i)` |
| fan (third order) | `sum_{j≠k} A_ij A_ik x_j x_k` | `sum_{j≠k} A_ij A_ik (x_j - x_i)(x_k - x_i)` |

Connectomes are rescaled either globally (mean in-strength 1, preserving
differences between nodes) or per row (every node has in-strength 1).

## Design choices that matter for reproducibility

* **Paired noise.** The noise seed of a simulation depends on network,
  normalisation, coupling order, drive, coupling strength and replicate, but
  not on the coupling form. Additive and diffusive runs of the same condition
  therefore share their noise.
* **Batch independence.** Each simulation owns its random generator, so results
  do not depend on how simulations are grouped into batches (tested).
* **Identical preprocessing.** Recorded and simulated signals pass through the
  same filter, detrending and z-scoring before any statistic is computed.
* **Estimator consistency.** Nearest-neighbour distances use the maximum norm,
  matching the `d log 2` volume term. The estimator is not invariant to
  rescaling individual variables, so signals are always z-scored first
  (tested).
* **Fixed triplets.** The 300 zebrafish triplets are drawn once with a fixed
  seed and shared by data, surrogates and models.
* **Matched null.** The nearest-neighbour estimator is biased by the marginal
  distribution and autocorrelation of the signals. Recordings and simulations
  are therefore compared with their own circular-shift surrogates, which keep
  both and destroy only the relations between regions.
* **Operating points.** Zebrafish models are placed at the drive and coupling
  that best reproduce the empirical functional connectivity; the O-information
  is never used for this choice.
* **Validation.** `01_analyse_zebrafish_data.py --check-against DIR` recomputes
  the recorded O-information and compares it with a stored reference.

## Data

The zebrafish recordings (36 larvae, 130 regions, 570 samples at 1 Hz) and
the structural connectome are shared under a data-sharing agreement and are
not distributed with this code. Place the files listed in
`data/raw/zebrafish/SHA256SUMS` in that folder and verify them with
`sha256sum -c SHA256SUMS`.
