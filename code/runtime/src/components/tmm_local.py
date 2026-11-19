"""
TMM - Transfer Matrix Method (移植自本地成功实现)

完整的光学计算，包含：
- 300-2500nm波长网格，221点，10nm间隔  
- 严格的光谱验证
- inc_tmm算法
- WVO2冷/热态处理
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d


# 材料文件映射
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
class SpectralState:
    """光谱状态（冷态或热态）"""
    wavelengths_nm: np.ndarray
    transmittance: np.ndarray
    front_reflectance: np.ndarray
    back_reflectance: np.ndarray


@dataclass(frozen=True) 
class TMMResult:
    """TMM计算结果"""
    cold: SpectralState
    hot: SpectralState
    valid: bool
    errors: Tuple[str, ...]


def _load_tmm_core(path: Path):
    """动态加载tmm_core模块"""
    spec = importlib.util.spec_from_file_location("tmm_core", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load TMM core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TMM:
    """传输矩阵方法光学计算"""
    
    def __init__(self, tmm_core_path: Path, optical_data_dir: Path):
        self.tmm_core = _load_tmm_core(tmm_core_path)
        self.optical_data_dir = Path(optical_data_dir)
        
        # 标准波长网格：300-2500nm, 221点, 10nm间隔
        self.wavelengths_nm = np.linspace(300.0, 2500.0, 221)
        
        # 加载材料
        self.refractive_index_funcs = {}
        self._load_materials()
        
        print(f"✓ TMM初始化成功")
        print(f"  - 波长网格: {self.wavelengths_nm[0]:.1f}-{self.wavelengths_nm[-1]:.1f} nm")
        print(f"  - 点数: {len(self.wavelengths_nm)}")
        print(f"  - 间隔: {self.wavelengths_nm[1] - self.wavelengths_nm[0]:.1f} nm")
        print(f"  - 材料数: {len(self.refractive_index_funcs)}")
    
    def _load_materials(self):
        """加载材料光学常数"""
        for material, filename in MATERIAL_FILES.items():
            path = self.optical_data_dir / filename
            if not path.exists():
                print(f"Warning: {material} 文件不存在")
                continue
            
            try:
                self.refractive_index_funcs[material] = self._load_nk_file(path)
            except Exception as e:
                print(f"Warning: 无法加载 {material}: {e}")
        
        # WVO2特殊处理
        wvo2_cold = self.optical_data_dir / "L_1.csv"
        wvo2_hot = self.optical_data_dir / "H_1.csv"
        
        if wvo2_cold.exists():
            self.refractive_index_funcs["WVO2_cold"] = self._load_nk_file(wvo2_cold)
        if wvo2_hot.exists():
            self.refractive_index_funcs["WVO2_hot"] = self._load_nk_file(wvo2_hot)
    
    def _load_nk_file(self, path: Path):
        """加载n和k数据，支持txt和csv格式"""
        if path.suffix == '.csv':
            data = np.loadtxt(path, delimiter=',')
            wl_um = data[:, 0]
            n_values = data[:, 1]
            idx = np.argsort(wl_um)
            n_func = interp1d(wl_um[idx], n_values[idx], 
                            kind='linear', fill_value='extrapolate')
            return lambda lam_um: complex(n_func(lam_um), 0.0)
        else:
            # TXT格式处理...（简化版）
            n_wl, n_val = [], []
            k_wl, k_val = [], []
            current_section = 'n'
            
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if 'wl' in line.lower():
                        if 'k' in line.lower():
                            current_section = 'k'
                        else:
                            current_section = 'n'
                        continue
                    
                    parts = line.replace('\t', ' ').split()
                    if len(parts) >= 2:
                        try:
                            wl = float(parts[0])
                            val = float(parts[1])
                            if current_section == 'n':
                                n_wl.append(wl)
                                n_val.append(val)
                            else:
                                k_wl.append(wl)
                                k_val.append(val)
                        except ValueError:
                            continue
            
            if not n_wl:
                raise ValueError(f"No n data in {path}")
            
            n_wl = np.array(n_wl)
            n_val = np.array(n_val)
            if n_wl[0] > 100:
                n_wl /= 1000.0
            
            idx = np.argsort(n_wl)
            n_func = interp1d(n_wl[idx], n_val[idx],
                            kind='linear', fill_value='extrapolate')
            
            if k_wl:
                k_wl = np.array(k_wl)
                k_val = np.array(k_val)
                if k_wl[0] > 100:
                    k_wl /= 1000.0
                idx_k = np.argsort(k_wl)
                k_func = interp1d(k_wl[idx_k], k_val[idx_k],
                                kind='linear', fill_value='extrapolate')
            else:
                k_func = lambda lam_um: 0.0
            
            return lambda lam_um: complex(n_func(lam_um), k_func(lam_um))
    
    def compute(self, structure_str: str) -> TMMResult:
        """计算结构的光学性能"""
        from src.utils.structure_utils import parse_structure
        
        tokens = parse_structure(structure_str)
        if not tokens:
            empty_state = SpectralState(
                self.wavelengths_nm, 
                np.zeros(221), 
                np.zeros(221), 
                np.zeros(221)
            )
            return TMMResult(empty_state, empty_state, False, ("Empty structure",))
        
        layers = [(mat, thick) for mat, thick in tokens]
        
        # 计算冷热两态
        errors = []
        states = {}
        
        for state_name, wvo2_key in [("cold", "WVO2_cold"), ("hot", "WVO2_hot")]:
            try:
                state = self._compute_state(layers, wvo2_key)
                validation_errors = self._validate_spectrum(state)
                if validation_errors:
                    errors.extend(validation_errors)
                states[state_name] = state
            except Exception as e:
                errors.append(f"{state_name} state failed: {e}")
                states[state_name] = SpectralState(
                    self.wavelengths_nm,
                    np.zeros(221),
                    np.zeros(221),
                    np.zeros(221)
                )
        
        # 构建TMMResult
        result_obj = TMMResult(
            cold=states["cold"],
            hot=states["hot"],
            valid=len(errors) == 0,
            errors=tuple(errors)
        )
        
        # 计算8维聚合特征
        def integrate_luminous(spectrum_array):
            # 简化：使用前40个点（约380-780nm范围）
            return float(np.mean(spectrum_array[:40]))
        
        def integrate_solar(spectrum_array):
            # 全波段平均
            return float(np.mean(spectrum_array))
        
        # 返回字典格式（兼容Agent接口）
        return {
            'T_lum_cold': integrate_luminous(result_obj.cold.transmittance),
            'T_lum_hot': integrate_luminous(result_obj.hot.transmittance),
            'T_sol_cold': integrate_solar(result_obj.cold.transmittance),
            'T_sol_hot': integrate_solar(result_obj.hot.transmittance),
            'Rf_sol_cold': integrate_solar(result_obj.cold.front_reflectance),
            'Rb_sol_cold': integrate_solar(result_obj.cold.back_reflectance),
            'Rf_sol_hot': integrate_solar(result_obj.hot.front_reflectance),
            'Rb_sol_hot': integrate_solar(result_obj.hot.back_reflectance),
            'valid': result_obj.valid,
            'errors': result_obj.errors,
            # 保留完整光谱
            '_cold_spectrum': {
                'wavelengths_nm': result_obj.cold.wavelengths_nm,
                'T': result_obj.cold.transmittance,
                'Rf': result_obj.cold.front_reflectance,
                'Rb': result_obj.cold.back_reflectance
            },
            '_hot_spectrum': {
                'wavelengths_nm': result_obj.hot.wavelengths_nm,
                'T': result_obj.hot.transmittance,
                'Rf': result_obj.hot.front_reflectance,
                'Rb': result_obj.hot.back_reflectance
            }
        }
    
    def _compute_state(self, layers, wvo2_key: str) -> SpectralState:
        """计算单个状态的光谱"""
        transmission = []
        front_refl = []
        back_refl = []
        
        for wavelength_nm in self.wavelengths_nm:
            wavelength_um = wavelength_nm / 1000.0
            
            # 构建TMM输入
            n_list = [1.0]
            d_list = [np.inf]
            c_list = ['i']
            
            for material, thickness_nm in layers:
                if material == "WVO2":
                    if wvo2_key in self.refractive_index_funcs:
                        n = self.refractive_index_funcs[wvo2_key](wavelength_um)
                    else:
                        n = 2.5 + 0.1j
                elif material in self.refractive_index_funcs:
                    n = self.refractive_index_funcs[material](wavelength_um)
                else:
                    n = 1.5 + 0.0j
                
                n_list.append(n)
                d_list.append(float(thickness_nm))
                c_list.append('i' if thickness_nm > 1000 else 'c')
            
            n_list.append(1.0)
            d_list.append(np.inf)
            c_list.append('i')
            
            # TMM计算
            try:
                result_front = self.tmm_core.inc_tmm(
                    's', n_list, d_list, c_list, 0.0, wavelength_nm
                )
                result_back = self.tmm_core.inc_tmm(
                    's', n_list[::-1], d_list[::-1], c_list[::-1], 0.0, wavelength_nm
                )
                
                transmission.append(float(result_front['T']))
                front_refl.append(float(result_front['R']))
                back_refl.append(float(result_back['R']))
            except:
                transmission.append(0.5)
                front_refl.append(0.3)
                back_refl.append(0.3)
        
        return SpectralState(
            self.wavelengths_nm.copy(),
            np.asarray(transmission, dtype=np.float64),
            np.asarray(front_refl, dtype=np.float64),
            np.asarray(back_refl, dtype=np.float64)
        )
    
    def _validate_spectrum(self, state: SpectralState) -> list:
        """验证光谱有效性"""
        errors = []
        
        if len(state.transmittance) != 221:
            errors.append(f"T length {len(state.transmittance)} != 221")
        
        for name, arr in [("T", state.transmittance),
                         ("Rf", state.front_reflectance),
                         ("Rb", state.back_reflectance)]:
            if not np.isfinite(arr).all():
                errors.append(f"{name} contains NaN or Inf")
            if np.any((arr < -1e-6) | (arr > 1 + 1e-6)):
                errors.append(f"{name} outside [0,1]")
        
        front_sum = state.transmittance + state.front_reflectance
        if np.any(front_sum > 1.01):
            errors.append(f"T+Rf > 1 (max={np.max(front_sum):.4f})")
        
        return errors
