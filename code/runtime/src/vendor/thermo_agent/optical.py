from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .tmm import TMMResult


FEATURE_ORDER = (
    "T_lum_cold", "T_lum_hot", "T_sol_cold", "T_sol_hot",
    "Rf_sol_cold", "Rb_sol_cold", "Rf_sol_hot", "Rb_sol_hot",
)


class OpticalFeatureCalculator:
    def __init__(self, workspace: str | Path) -> None:
        workspace = Path(workspace).resolve()
        data_dir = workspace / "Transopto_TRL" / "dataset" / "udi_energy"
        target_wavelengths = np.linspace(300, 2500, 221)
        cie = pd.read_csv(data_dir / "CIE_sle_photopic.csv", header=None, names=["wl", "w"])
        self.luminous_weight = np.interp(
            target_wavelengths, cie["wl"], cie["w"], left=0, right=0
        )
        self.luminous_weight /= self.luminous_weight.sum()
        astm = pd.read_csv(data_dir / "astmg173.csv", skiprows=1)
        self.solar_weight = np.interp(
            target_wavelengths, astm.iloc[:, 0], astm.iloc[:, 2], left=0, right=0
        )
        self.solar_weight /= self.solar_weight.sum()

    def calculate(self, result: TMMResult) -> dict[str, float]:
        return {
            "T_lum_cold": float(np.dot(result.cold.transmittance, self.luminous_weight)),
            "T_lum_hot": float(np.dot(result.hot.transmittance, self.luminous_weight)),
            "T_sol_cold": float(np.dot(result.cold.transmittance, self.solar_weight)),
            "T_sol_hot": float(np.dot(result.hot.transmittance, self.solar_weight)),
            "Rf_sol_cold": float(np.dot(result.cold.front_reflectance, self.solar_weight)),
            "Rb_sol_cold": float(np.dot(result.cold.back_reflectance, self.solar_weight)),
            "Rf_sol_hot": float(np.dot(result.hot.front_reflectance, self.solar_weight)),
            "Rb_sol_hot": float(np.dot(result.hot.back_reflectance, self.solar_weight)),
        }
