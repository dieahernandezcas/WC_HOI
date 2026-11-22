# Data Dictionary — Zebrafish Whole-Brain Calcium Imaging
## Dataset: Prejaas Tewarie Lab (CERVO / Nottingham)

---

## General Description

Spontaneous whole-brain neural activity recorded from **36 larval zebrafish** using widefield calcium imaging. Neural signals are aggregated into brain regions using two spatial parcellations (65 and 120 brain areas). The dataset also includes structural connectivity (SC) derived from a zebrafish connectome atlas, spatial coordinates, and projection images for visualization.

> **Key note on hemispheres:** All multi-region arrays contain data for **both hemispheres**. The convention is **left hemisphere first, right hemisphere second**. For the 65-region parcellation, indices 0–64 = left hemisphere, indices 65–129 = right hemisphere. For the 120-region parcellation, indices 0–119 = left, 120–239 = right. Regions are sorted **anterior → posterior** within each hemisphere.

---

## Files

### Timeseries (neural activity)

| File | Shape | Description |
|------|-------|-------------|
| `timeseries_65.npy` | `(36, 130, 570)` | Calcium signal per animal, per region (65-parcel, both hemispheres), per timepoint |
| `timeseries_120.npy` | `(36, 240, 570)` | Calcium signal per animal, per region (120-parcel, both hemispheres), per timepoint |

**Details:**
- **Axis 0** — Animal index (0–35), 36 subjects total
- **Axis 1** — Brain region index (0–129 for 65-parcel; 0–239 for 120-parcel)
- **Axis 2** — Timepoint (0–569), 570 samples
- **Value range:** −0.82 to +10.16 (65-parcel); similar for 120-parcel
- **Signal type:** ΔF/F or equivalent calcium proxy. Mean ≈ 0.82–0.84, std ≈ 0.25–0.29. Positively skewed (skew 0.04–1.85 depending on region). **Not z-scored per region** — raw amplitude matters.
- **Lag-1 autocorrelation:** High, confirming temporal structure (oscillatory/AR dynamics)

---

### Structural Connectivity (SC)

| File | Shape | Description |
|------|-------|-------------|
| `SC_undirected_65.npy` | `(130, 130)` | Symmetric SC, 65-parcel (both hemispheres) |
| `SC_directed_65.npy` | `(130, 130)` | Asymmetric SC, 65-parcel ⚠️ (see caveats) |
| `SC_undirected_120.npy` | `(240, 240)` | Symmetric SC, 120-parcel (both hemispheres) |
| `SC_directed_120.npy` | `(240, 240)` | Asymmetric SC, 120-parcel ⚠️ (see caveats) |

**Details:**
- **Value range:** 0.0 to 1.0 (normalized connection strength from tractography)
- **SC density:** 35.1% for 65-parcel (2,943 edges), 22.3% for 120-parcel (6,388 edges)
- **Triplet structure (65-parcel, valid regions only):**
  - SC triangles (all 3 edges present): **24,486** (15.1% of 161,700 triplets)
  - Open paths (2 of 3 edges): **41,970** (26.0%)
  - Sparse (≤1 edge): **95,244** (58.9%)
- ⚠️ **Directed SC caveat:** Directionality is difficult to infer (axons vs. dendrites not distinguished). Use directed SC only for exploratory analyses; prefer undirected for main results.

---

### Region Metadata

| File | Shape | Description |
|------|-------|-------------|
| `region_names_65.npy` | `(65,)` | Anatomical name for each of the 65 unique regions |
| `region_names_120.npy` | `(120,)` | Anatomical name for each of the 120 unique regions |
| `node_coordinates_65.npy` | `(130, 3)` | 3D spatial coordinates (x, y, z) in voxels for all 130 nodes (65×2 hemispheres) |
| `node_coordinates_120.npy` | `(240, 3)` | 3D spatial coordinates for all 240 nodes |
| `n_neurons_per_region_65.npy` | `(36, 130)` | Number of neurons recorded per animal per region (65-parcel) |
| `n_neurons_per_region_120.npy` | `(36, 240)` | Number of neurons recorded per animal per region (120-parcel) |
| `excluded_regions_65.npy` | `(30,)` | Indices of regions to exclude in 65-parcel (poor coverage) |
| `excluded_regions_120.npy` | `(42,)` | Indices of regions to exclude in 120-parcel (poor coverage) |

**Coordinates:**
- Axis 1 = x (anterior–posterior), Axis 2 = y (dorsal–ventral), Axis 3 = z (medial–lateral)
- 65-parcel coord range: x=104–466, y=104–814, z=31–314

**Excluded regions:** 30/130 nodes excluded in 65-parcel, 42/240 in 120-parcel → after exclusion: **100 valid nodes** (65-parcel), **198 valid nodes** (120-parcel). Always apply exclusion mask before analysis.

**Neurons per region:** Some animals have 0 neurons in some regions (coverage gaps). Regions with `n_neurons_per_region == 0` should be treated as missing for that animal.

---

### Brain Projection Images

| File | Shape | Description |
|------|-------|-------------|
| `brain_side.npy` | `(974, 359)` | 2D lateral view projection of zebrafish brain (for overlaying spatial plots) |
| `brain_top.npy` | `(974, 597)` | 2D dorsal view projection of zebrafish brain |

**Usage:** Use `plt.imshow(brain_top, cmap='gray')` or `cmap='Greys_r'` as background, then scatter node coordinates on top using `node_coordinates[:, 0]` (anterior–posterior axis = x-axis of dorsal view) and `node_coordinates[:, 2]` (medial–lateral axis).

---

## Analysis Notes

### Recommended parcellation for HOI analysis
Use **65-parcel** as primary parcellation:
- Higher neurons-per-region → more reliable region-level signal (mean=429 neurons)
- 100 valid nodes after exclusion → C(100,3) = 161,700 triplets
- SC density (35%) provides sufficient structural context

### Signal preprocessing checklist
```python
# Standard pipeline:
ts = np.load('timeseries_65.npy')         # (36, 130, 570)
ex = np.load('excluded_regions_65.npy')   # regions to drop
nn = np.load('n_neurons_per_region_65.npy')  # (36, 130)
# 1. Apply spatial exclusion mask
mask = np.ones(130, dtype=bool); mask[ex] = False
ts_valid = ts[:, mask, :]                  # (36, 100, 570)
nn_valid = nn[:, mask]                     # (36, 100)
# 2. Per-animal z-score (removes baseline differences)
ts_z = (ts_valid - ts_valid.mean(axis=2, keepdims=True)) / \
       (ts_valid.std(axis=2, keepdims=True) + 1e-8)
# 3. Per-animal, per-region: mask regions with 0 neurons
zero_mask = (nn_valid == 0)               # (36, 100), True where unreliable
```

### KSG estimator parameters
- Use **k=5** nearest neighbors (consistent with paper)
- **570 timepoints** is sufficient for 3D KSG estimation (well above minimum ~100)
- Use **Chebyshev** (L∞) norm for joint space, **Euclidean** for marginals (Kraskov et al. 2004)

---

## Anatomical Highlights (65-parcel, key regions)

| Index | Region | Note |
|-------|--------|------|
| 0 | olfactory_epithelium | Anterior-most |
| 1 | olfactory_bulb | Strong oscillatory activity known |
| 17 | tectal_neuropil | Visual processing hub; high activity |
| 33 | superior_raphe | Serotonergic; neuromodulatory |
| 35 | locus_coeruleus | Noradrenergic; oscillatory |
| 36 | cerebellum | Motor coordination; highly structured |
| 64 | area_postrema | Posterior-most |

---

*Dictionary generated: 2026-04-24 | Dataset courtesy: Prejaas Tewarie, CERVO Brain Research Center*
