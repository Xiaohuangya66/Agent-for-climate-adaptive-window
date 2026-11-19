from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from .spectra import SpectralState, validate_spectral_state
from .structures import StructureValidation, validate_structure


MATERIAL_FILES = {
    "TiO2": "tio2.txt",
    "ZnO": "zno.txt",
    "Ta2O5": "ta2o5.txt",
    "SiO2": "sio2.txt",
    "MgF2": "mgf2.txt",
    "WO3": "wo3.txt",
    "Ag": "Ag.txt",
    "ITO": "ito.txt",
}


@dataclass(frozen=True)
class TMMResult:
    structure: StructureValidation
    cold: SpectralState
    hot: SpectralState


def _load_tmm_core(path: Path):
    spec = importlib.util.spec_from_file_location("thermo_agent_legacy_tmm_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TMM core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_nk_material(path: str | Path) -> Callable[[float], complex]:
    path = Path(path)
    current = "n"
    blocks = {"n": ([], []), "k": ([], [])}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("wl"):
            fields = lower.split()
            current = "k" if len(fields) > 1 and fields[1] == "k" else "n"
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            wavelength, value = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        blocks[current][0].append(wavelength)
        blocks[current][1].append(value)

    if not blocks["n"][0]:
        raise ValueError(f"No refractive-index data in {path}")

    def make_interpolator(wavelengths: list[float], values: list[float], default: float | None = None):
        if not wavelengths:
            if default is None:
                raise ValueError(f"Missing optical data in {path}")
            return lambda wavelength_um: float(default)
        x = np.asarray(wavelengths, dtype=float)
        y = np.asarray(values, dtype=float)
        if x[0] > 100:
            x = x / 1000.0
        order = np.argsort(x)
        return interp1d(x[order], y[order], kind="linear", fill_value="extrapolate")

    n_interp = make_interpolator(*blocks["n"])
    k_interp = make_interpolator(*blocks["k"], default=0.0)
    return lambda wavelength_um: complex(n_interp(wavelength_um), k_interp(wavelength_um))


class TMMCalculator:
    @staticmethod
    def coherence_list(layers):
        # The search space no longer guarantees that its last token is glass.
        # Keep the reference's 10-um SiO2 substrate phase averaging, including
        # internal occurrences; ordinary coating layers remain coherent.
        return ['i',*['i' if layer.material=='SiO2' and layer.thickness_nm==10000 else 'c'
                      for layer in layers],'i']

    def __init__(self, workspace: str | Path, *, max_layers: int = 30) -> None:
        self.workspace = Path(workspace).resolve()
        self.max_layers = max_layers
        nk_dir = self.workspace / "TMM" / "TMM_ML"
        self.tmm_core = _load_tmm_core(self.workspace / "TMM" / "tmm_core.py")
        self.materials = {
            name: load_nk_material(nk_dir / filename) for name, filename in MATERIAL_FILES.items()
        }
        self.wvo2 = {
            "cold": load_nk_material(nk_dir / "wvo2_cold.txt"),
            "hot": load_nk_material(nk_dir / "wvo2_hot.txt"),
        }
        self.wavelengths_nm = np.linspace(300.0, 2500.0, 221)

    def calculate(
        self,
        structure: str,
        *,
        allowed_tokens: set[str] | None = None,
        wvo2_min_thickness_nm: float = 20,
        wvo2_max_thickness_nm: float = 40,
        nk_perturbation: dict[str, tuple[float, float]] | None = None,
    ) -> TMMResult:
        validation = validate_structure(
            structure,
            max_layers=self.max_layers,
            allowed_tokens=allowed_tokens,
            wvo2_min_thickness_nm=wvo2_min_thickness_nm,
            wvo2_max_thickness_nm=wvo2_max_thickness_nm,
        )
        if not validation.valid:
            raise ValueError("Invalid structure: " + "; ".join(validation.errors))

        layers = validation.layers
        thicknesses = [layer.thickness_nm for layer in layers]
        d_list = [np.inf, *thicknesses, np.inf]
        c_list = self.coherence_list(layers)
        states: dict[str, SpectralState] = {}

        for state_name, wvo2_function in self.wvo2.items():
            transmission, front, back = [], [], []
            for wavelength_nm in self.wavelengths_nm:
                wavelength_um = wavelength_nm / 1000.0
                n_list = [1.0]
                for layer in layers:
                    function = wvo2_function if layer.material == "WVO2" else self.materials[layer.material]
                    refractive_index = function(wavelength_um)
                    if nk_perturbation and layer.material in nk_perturbation:
                        n_scale, k_scale = nk_perturbation[layer.material]
                        refractive_index = complex(
                            refractive_index.real * n_scale,
                            refractive_index.imag * k_scale,
                        )
                    n_list.append(refractive_index)
                n_list.append(1.0)

                result_front = self.tmm_core.inc_tmm("s", n_list, d_list, c_list, 0.0, wavelength_nm)
                result_back = self.tmm_core.inc_tmm(
                    "s", n_list[::-1], d_list[::-1], c_list[::-1], 0.0, wavelength_nm
                )
                transmission.append(float(result_front["T"]))
                front.append(float(result_front["R"]))
                back.append(float(result_back["R"]))

            spectral_state = SpectralState(
                self.wavelengths_nm.copy(),
                np.asarray(transmission),
                np.asarray(front),
                np.asarray(back),
            )
            spectral_validation = validate_spectral_state(spectral_state, tolerance=1e-5)
            if not spectral_validation.valid:
                raise ValueError(
                    f"TMM produced invalid {state_name} spectrum: "
                    + "; ".join(spectral_validation.errors)
                )
            states[state_name] = spectral_state
        return TMMResult(validation, states["cold"], states["hot"])
