"""Bermuda SEM dataset used by Rehearsal experiments."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

from rehearsal.core import AUFTask, AlterationDomain, DesiredRegion
from rehearsal.datasets.sem import RehearsalDatasetSpec, theta_from_children
from rehearsal.models import parents_from_theta

_BERMUDA_NEC_BIAS = -0.5402197079153048


def bermuda(
    data_path: str | Path | None = None,
    *,
    covariance_profile: str = "paper",
) -> RehearsalDatasetSpec:
    """Return the Bermuda SEM dataset and task specification."""

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
    theta = theta_from_children(
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
            "dataset": "Bermuda",
            "reference_paper": "ICML 2025 CARE",
            "raw_data_path": str(default_bermuda_data_path()),
            "nec_bias": y_bias,
            "covariance_profile": covariance_profile,
            "true_region": {"kind": "interval", "lower": 0.5 - y_bias, "upper": 2.0 - y_bias},
        },
    )

    def true_region(y: np.ndarray) -> np.ndarray:
        values = np.asarray(y, dtype=float).reshape(-1)
        return (values >= 0.5 - y_bias) & (values <= 2.0 - y_bias)

    return RehearsalDatasetSpec(
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
            "reference_paper": "ICML 2025 CARE",
            "nec_bias": y_bias,
            "covariance_profile": covariance_profile,
        },
    )


def load_bermuda_standardized_data(data_path: str | Path | None = None) -> dict[str, np.ndarray]:
    """Load the migrated Bermuda ``SEM_data.mat`` and apply StandardScaler-style normalization."""

    data = _load_bermuda_mat(data_path)
    keep = {key: value for key, value in data.items() if key not in _bermuda_excluded_keys()}
    standardized: dict[str, np.ndarray] = {}
    for key, value in keep.items():
        arr = np.asarray(value, dtype=float).reshape(-1)
        standardized[key] = (arr - np.nanmean(arr)) / np.nanstd(arr)
    return standardized


def default_bermuda_data_path() -> Path:
    """Return the package path for the migrated Bermuda SEM data."""

    return Path(str(resources.files("rehearsal.datasets").joinpath("data/bermuda_sem_data.mat")))


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

    path = Path(data_path) if data_path is not None else default_bermuda_data_path()
    return loadmat(path)


def _bermuda_excluded_keys() -> set[str]:
    return {"__header__", "__version__", "__globals__", "Site", "Lat", "Lon", "Year", "Month", "Day"}
