"""Dataset migrations for the ICML 2025 CARE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion, circular_region_inner_care
from rehearsal.models import LinearGaussianSRM, parents_from_theta


Theta = Mapping[str, Mapping[str, float]]
RegionPredicate = Callable[[np.ndarray], np.ndarray]

_BERMUDA_NEC_BIAS = -0.5402197079153048


@dataclass(frozen=True)
class ICML2025DatasetSpec:
    """Runnable SEM-backed dataset specification used by the ICML 2025 paper."""

    name: str
    task: AUFTask
    theta: Theta
    covariance: np.ndarray
    true_region_contains: RegionPredicate
    paper_claim: Mapping[str, float]
    default_n_data: int
    metadata: Mapping[str, object]


def manage_icml2025() -> ICML2025DatasetSpec:
    """Synthetic Manage dataset from ``previous_works/05-ICML 2025``."""

    nodes = (
        "competitor_feature",
        "economic_index",
        "competitor_raw_cost",
        "raw_cost",
        "self_pricing",
        "competitor_pricing",
        "total_profit",
        "custom_number",
    )
    theta = _theta(
        {
            "competitor_feature": {"competitor_raw_cost": 10.0},
            "economic_index": {"raw_cost": 10.0},
            "self_pricing": {"total_profit": 0.9, "custom_number": -0.9},
            "raw_cost": {
                "competitor_pricing": 0.5,
                "self_pricing": 2.0,
                "total_profit": -1.0,
                "custom_number": 1.6,
            },
            "competitor_raw_cost": {"competitor_pricing": 1.3, "self_pricing": 0.4},
        }
    )
    covariance = np.array(
        [
            [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.03, 0.016, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.016, 0.06, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12],
        ],
        dtype=float,
    )
    center = np.array([1.2, 1.5], dtype=float)
    radius = 1.0
    desired_region = circular_region_inner_care(
        center,
        radius,
        covariance=np.diag([0.04, 0.12]),
    )
    task = AUFTask(
        observed_variables=("competitor_feature", "economic_index"),
        alterable_variables=("raw_cost", "self_pricing"),
        outcome_variables=("total_profit", "custom_number"),
        desired_region=desired_region,
        alteration_domain=AlterationDomain({"raw_cost": (-6.0, 6.0), "self_pricing": (-6.0, 6.0)}),
        parents=parents_from_theta(theta, nodes),
        candidate_alteration_sets=(("raw_cost", "self_pricing"),),
        variable_order=nodes,
        metadata={
            "paper": "ICML 2025",
            "dataset": "Synthetic Manage",
            "true_region": {"kind": "circle", "center": tuple(center), "radius": radius},
        },
    )

    def true_region(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float)
        return np.linalg.norm(values - center.reshape(1, -1), axis=1) <= radius

    return ICML2025DatasetSpec(
        name="manage",
        task=task,
        theta=theta,
        covariance=covariance,
        true_region_contains=true_region,
        paper_claim={
            "ours_care_success_percent": 99.01,
            "ours_100_round_success": 98.93,
            "ours_avg_time_ms": 5.86,
        },
        default_n_data=100,
        metadata={"source": "previous_works/05-ICML 2025/code/main_syn.py"},
    )


def bermuda_icml2025(
    data_path: str | Path | None = None,
    *,
    covariance_profile: str = "paper",
) -> ICML2025DatasetSpec:
    """Bermuda dataset and SEM parameters used in the ICML 2025 experiments."""

    nodes = (
        "Light",
        "Temp",
        "Sal",
        "DIC",
        "TA",
        "Omega",
        "Nutrients_PC1",
        "Chla",
        "pHsw",
        "CO2",
        "NEC",
    )
    theta = _theta(
        {
            "Light": {"Temp": 0.08336954980497932, "Chla": -0.15106684218500258, "NEC": 0.0322460348829162},
            "Temp": {
                "Sal": -0.4809837373167684,
                "Omega": 0.5182253589055726,
                "Chla": -0.04451583134247557,
                "pHsw": -0.7482789216296077,
                "CO2": 0.8613318110706953,
                "NEC": 5.227658403563992,
            },
            "Sal": {
                "DIC": 0.4777168829735183,
                "TA": 0.5457734531124397,
                "Omega": 0.03507218718555735,
                "pHsw": 0.013001179873522933,
                "CO2": 0.04051812201172802,
            },
            "DIC": {"Omega": -1.1056652215053286, "pHsw": -0.5879618774787132, "CO2": 0.5700488513842487},
            "TA": {"Omega": 1.6104231541803835, "pHsw": 0.7676261914081877, "CO2": -0.596251974686561},
            "Nutrients_PC1": {"Chla": -0.07690378415962325, "NEC": 0.09881771775808317},
            "Omega": {"NEC": -2.343629162533968},
            "Chla": {"NEC": 0.13182892043084415},
            "pHsw": {"NEC": 2.0492558654639},
            "CO2": {"NEC": -2.5146414696724295},
        }
    )
    covariance = _bermuda_covariance(covariance_profile)
    y_bias = _bermuda_nec_bias(data_path)
    desired_region = DesiredRegion(M=[1.0, -1.0], d=[2.0 - y_bias, -0.5 + y_bias])
    task = AUFTask(
        observed_variables=("Light", "Temp", "Sal"),
        alterable_variables=("DIC", "TA", "Omega", "Chla", "Nutrients_PC1"),
        outcome_variables=("NEC",),
        desired_region=desired_region,
        alteration_domain=AlterationDomain(
            {
                "DIC": (-1.0, 1.0),
                "TA": (-1.0, 1.0),
                "Omega": (-1.0, 1.0),
                "Chla": (-1.0, 1.0),
                "Nutrients_PC1": (-1.0, 1.0),
            }
        ),
        parents=parents_from_theta(theta, nodes),
        candidate_alteration_sets=(("DIC",), ("TA",), ("Omega",), ("Chla",), ("Nutrients_PC1",)),
        variable_order=nodes,
        metadata={
            "paper": "ICML 2025",
            "dataset": "Bermuda",
            "raw_data_path": str(_default_bermuda_data_path()),
            "nec_bias": y_bias,
            "covariance_profile": covariance_profile,
            "true_region": {"kind": "interval", "lower": 0.5 - y_bias, "upper": 2.0 - y_bias},
        },
    )

    def true_region(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).reshape(-1)
        return (values >= 0.5 - y_bias) & (values <= 2.0 - y_bias)

    return ICML2025DatasetSpec(
        name="bermuda",
        task=task,
        theta=theta,
        covariance=covariance,
        true_region_contains=true_region,
        paper_claim={
            "ours_care_success_percent": 82.76,
            "ours_100_round_success": 83.26,
            "ours_avg_time_ms": 0.91,
        },
        default_n_data=1000,
        metadata={
            "source": "previous_works/05-ICML 2025/code/main_bermuda.py",
            "nec_bias": y_bias,
            "covariance_profile": covariance_profile,
        },
    )


def generate_observational_data(
    spec: ICML2025DatasetSpec,
    n_samples: int,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate observational samples from the migrated linear Gaussian SEM."""

    rng = np.random.default_rng(seed)
    return _simulate_sem(spec, n_samples, rng=rng)


def sample_observation(spec: ICML2025DatasetSpec, rng: np.random.Generator) -> dict[str, float]:
    """Sample the observed stage as the ICML legacy scripts do."""

    noise = rng.multivariate_normal(np.zeros(len(spec.task.variable_order)), spec.covariance)
    index = {name: idx for idx, name in enumerate(spec.task.variable_order)}
    return {name: float(noise[index[name]]) for name in spec.task.observed}


def estimate_true_auf_success_rate(
    spec: ICML2025DatasetSpec,
    observation: Mapping[str, float],
    alterations: Mapping[str, float],
    n_samples: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo AUF success rate under the true paper region."""

    if rng is None:
        rng = np.random.default_rng(seed)
    simulated = _simulate_sem(spec, n_samples, rng=rng, observation=observation, alterations=alterations)
    outcomes = np.column_stack([simulated[name] for name in spec.task.outcomes])
    return float(np.mean(spec.true_region_contains(outcomes)))


def load_bermuda_standardized_data(data_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load the migrated Bermuda ``SEM_data.mat`` and apply StandardScaler-style normalization."""

    data = _load_bermuda_mat(data_path)
    keep = {key: value for key, value in data.items() if key not in _bermuda_excluded_keys()}
    standardized: dict[str, np.ndarray] = {}
    for key, value in keep.items():
        arr = np.asarray(value, dtype=float).reshape(-1)
        standardized[key] = (arr - np.nanmean(arr)) / np.nanstd(arr)
    return standardized


def _simulate_sem(
    spec: ICML2025DatasetSpec,
    n_samples: int,
    *,
    rng: np.random.Generator,
    observation: Mapping[str, float] | None = None,
    alterations: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    model = LinearGaussianSRM(tuple(spec.task.variable_order), dict(spec.theta), spec.covariance)
    return model.simulate(
        n_samples,
        rng=rng,
        observation=observation,
        alterations=alterations,
    )


def _theta(values: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    return {parent: dict(children) for parent, children in values.items()}


def _bermuda_nec_bias(data_path: str | Path | None) -> float:
    try:
        data = load_bermuda_standardized_data(data_path)
    except ImportError:
        return _BERMUDA_NEC_BIAS

    parents = ("Nutrients_PC1", "Light", "pHsw", "Omega", "Chla", "CO2", "Temp")
    y_name = "NEC"
    matrix = np.column_stack([data[name] for name in parents + (y_name,)])
    valid = ~np.isnan(matrix).any(axis=1)
    x = matrix[valid, : len(parents)]
    y = matrix[valid, len(parents)]
    design = np.column_stack([np.ones(x.shape[0]), x])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coef[0])


def _bermuda_covariance(profile: str) -> np.ndarray:
    if profile == "legacy_isotropic":
        return 2e-2 * np.eye(11)
    if profile != "paper":
        raise ValueError("covariance_profile must be 'paper' or 'legacy_isotropic'.")
    variances = np.array(
        [
            1.2e-2,
            1.6e-2,
            1.6e-2,
            1.0e-2,
            2.0e-2,
            1.6e-2,
            1.6e-2,
            1.8e-2,
            1.0e-3,
            1.6e-3,
            1.5e-2,
        ],
        dtype=float,
    )
    return np.diag(variances)


def _load_bermuda_mat(data_path: str | Path | None) -> dict[str, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Loading the Bermuda .mat file requires scipy.") from exc

    path = Path(data_path) if data_path is not None else _default_bermuda_data_path()
    return loadmat(path)


def _default_bermuda_data_path() -> Path:
    return Path(str(resources.files("rehearsal.datasets").joinpath("data/icml2025_bermuda_sem_data.mat")))


def _bermuda_excluded_keys() -> set[str]:
    return {"__header__", "__version__", "__globals__", "Site", "Lat", "Lon", "Year", "Month", "Day"}
