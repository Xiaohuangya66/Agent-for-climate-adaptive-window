from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SpectralState:
    wavelengths_nm: np.ndarray
    transmittance: np.ndarray
    front_reflectance: np.ndarray
    back_reflectance: np.ndarray


@dataclass(frozen=True)
class SpectrumValidation:
    valid: bool
    errors: tuple[str, ...]
    max_front_energy_sum: float | None
    max_back_energy_sum: float | None


def parse_spectral_values(value: str | Iterable[float]) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray([float(item) for item in value.split(",")], dtype=float)
    return np.asarray(list(value), dtype=float)


def validate_spectral_state(
    state: SpectralState,
    *,
    expected_start_nm: float = 300,
    expected_stop_nm: float = 2500,
    expected_step_nm: float = 10,
    tolerance: float = 1e-6,
) -> SpectrumValidation:
    errors: list[str] = []
    arrays = {
        "wavelengths": np.asarray(state.wavelengths_nm, dtype=float),
        "T": np.asarray(state.transmittance, dtype=float),
        "Rf": np.asarray(state.front_reflectance, dtype=float),
        "Rb": np.asarray(state.back_reflectance, dtype=float),
    }
    expected_count = int(round((expected_stop_nm - expected_start_nm) / expected_step_nm)) + 1
    for name, array in arrays.items():
        if array.ndim != 1:
            errors.append(f"{name} must be one-dimensional.")
        if len(array) != expected_count:
            errors.append(f"{name} has {len(array)} values; expected {expected_count}.")
        if not np.isfinite(array).all():
            errors.append(f"{name} contains NaN or infinity.")

    wavelengths = arrays["wavelengths"]
    if len(wavelengths) == expected_count and np.isfinite(wavelengths).all():
        expected = np.linspace(expected_start_nm, expected_stop_nm, expected_count)
        if not np.allclose(wavelengths, expected, atol=tolerance, rtol=0):
            errors.append("Wavelength grid does not match the configured 300–2500 nm grid.")

    for name in ("T", "Rf", "Rb"):
        array = arrays[name]
        if np.isfinite(array).all() and (np.any(array < -tolerance) or np.any(array > 1 + tolerance)):
            errors.append(f"{name} contains values outside [0, 1].")

    max_front = max_back = None
    if len({len(arrays[key]) for key in ("T", "Rf", "Rb")}) == 1:
        front_sum = arrays["T"] + arrays["Rf"]
        back_sum = arrays["T"] + arrays["Rb"]
        if np.isfinite(front_sum).all():
            max_front = float(np.max(front_sum))
            if max_front > 1 + tolerance:
                errors.append(f"Front energy sum max(T+Rf)={max_front:.8f} exceeds 1+tolerance.")
        if np.isfinite(back_sum).all():
            max_back = float(np.max(back_sum))
            if max_back > 1 + tolerance:
                errors.append(f"Back energy sum max(T+Rb)={max_back:.8f} exceeds 1+tolerance.")
    return SpectrumValidation(not errors, tuple(errors), max_front, max_back)


def state_from_strings(
    wavelengths_nm: Iterable[float],
    transmittance: str,
    front_reflectance: str,
    back_reflectance: str,
) -> SpectralState:
    return SpectralState(
        np.asarray(list(wavelengths_nm), dtype=float),
        parse_spectral_values(transmittance),
        parse_spectral_values(front_reflectance),
        parse_spectral_values(back_reflectance),
    )
