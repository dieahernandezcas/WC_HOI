"""Configuration objects and helpers.

Every quantity that affects a result is defined here or in a JSON file under
``configs/``. Experiment scripts read a JSON configuration, convert it into the
dataclasses below, and store the resolved configuration next to their outputs.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
RAW_ZEBRAFISH_DIR = DATA_DIR / "raw" / "zebrafish"
RESULTS_DIR = PACKAGE_ROOT / "results"


@dataclass(frozen=True)
class WilsonCowanParams:
    """Local (within-node) parameters of the Wilson-Cowan model.

    Naming follows target-source order: ``c_ei`` is the weight from the
    inhibitory to the excitatory population, ``c_ie`` the weight from the
    excitatory to the inhibitory population.
    """

    c_ee: float = 16.0
    c_ei: float = 12.0
    c_ie: float = 15.0
    c_ii: float = 3.0
    tau_e: float = 2.5
    tau_i: float = 3.75
    beta: float = 0.1
    gamma: float = 3.0
    sigma: float = 0.015
    drive_i: float = 0.0


@dataclass(frozen=True)
class IntegrationParams:
    """Numerical integration and sampling settings.

    One recorded sample corresponds to ``record_every * dt`` model time units.
    With the defaults, the simulated series have the same number of samples
    as the zebrafish recordings.
    """

    dt: float = 0.1
    record_every: int = 20
    n_burn_records: int = 300
    n_records: int = 570


COUPLING_ORDERS = ("pairwise", "fan", "pairwise+fan")


@dataclass(frozen=True)
class CouplingSpec:
    """Which coupling input a node receives.

    form:  ``"additive"`` or ``"diffusive"``; applies to every term.
    order: ``"pairwise"``      input ``K * C``
           ``"fan"``           input ``K * F``
           ``"pairwise+fan"``  input ``K * (C + fan_ratio * F)``
    where ``C`` and ``F`` are defined in :mod:`coupling_hoi.coupling`. The fan
    input scales with the square of the activity, so ``fan_ratio`` brings the
    two terms to comparable magnitudes for typical activity levels.
    """

    form: str
    order: str
    fan_ratio: float = 20.0

    def __post_init__(self) -> None:
        if self.form not in ("additive", "diffusive"):
            raise ValueError(f"unknown coupling form: {self.form!r}")
        if self.order not in COUPLING_ORDERS:
            raise ValueError(f"unknown coupling order: {self.order!r}")

    @property
    def label(self) -> str:
        return f"{self.order}-{self.form}"


@dataclass(frozen=True)
class InformationParams:
    """Settings of the O-information estimator."""

    k_neighbours: int = 5


@dataclass(frozen=True)
class PreprocessingParams:
    """Preprocessing applied identically to recorded and simulated signals."""

    sampling_rate_hz: float = 1.0
    highpass_cutoff_hz: float = 0.005
    highpass_order: int = 2


@dataclass
class ExperimentConfig:
    """Container for a full experiment description loaded from JSON."""

    name: str
    base_seed: int
    model: WilsonCowanParams = field(default_factory=WilsonCowanParams)
    integration: IntegrationParams = field(default_factory=IntegrationParams)
    information: InformationParams = field(default_factory=InformationParams)
    preprocessing: PreprocessingParams = field(default_factory=PreprocessingParams)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        return cls(
            name=raw["name"],
            base_seed=int(raw["base_seed"]),
            model=WilsonCowanParams(**raw.get("model", {})),
            integration=IntegrationParams(**raw.get("integration", {})),
            information=InformationParams(**raw.get("information", {})),
            preprocessing=PreprocessingParams(**raw.get("preprocessing", {})),
            extra={k: v for k, v in raw.items()
                   if k not in ("name", "base_seed", "model", "integration",
                                "information", "preprocessing")},
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "base_seed": self.base_seed,
            "model": asdict(self.model),
            "integration": asdict(self.integration),
            "information": asdict(self.information),
            "preprocessing": asdict(self.preprocessing),
        }
        out.update(self.extra)
        return out

    def fingerprint(self) -> str:
        """Short hash identifying the resolved configuration."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def stable_seed(base_seed: int, *keys: Any) -> int:
    """Deterministic 32-bit seed derived from a base seed and a tuple of keys.

    The coupling form is deliberately *not* passed as a key by the experiment
    runners, so that the additive and diffusive simulations of a given
    condition receive identical noise and can be compared as paired samples.
    """
    text = "|".join(str(k) for k in keys)
    return int(np.random.SeedSequence([base_seed, zlib.crc32(text.encode())])
               .generate_state(1)[0])
