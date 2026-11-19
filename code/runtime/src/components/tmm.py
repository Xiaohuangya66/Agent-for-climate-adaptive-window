"""
TMM Public Interface (Server Adapter)

保留服务器公共接口，底层调用本地成功实现
"""
from pathlib import Path
import sys
import numpy as np

# 添加vendor路径
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / 'src' / 'vendor'))

from thermo_agent.tmm import TMMCalculator
from thermo_agent.optical import OpticalFeatureCalculator


class TMM:
    """
    TMM公共接口（适配层）
    
    保持服务器现有调用方式：
        tmm = TMM(optical_constants_dir)
        result = tmm.compute(structure_str)
    
    底层使用本地成功实现
    """
    
    def __init__(self, optical_constants_dir: Path):
        """
        Args:
            optical_constants_dir: 光学常数目录（服务器参数，实际使用reference_assets）
        """
        # 使用reference_assets作为材料目录
        from src.support_settings import asset_path
        runtime_assets = asset_path('reference_assets', _ROOT / 'reference_assets')
        
        self.calculator = TMMCalculator(runtime_assets, max_layers=30)
        self.optical = OpticalFeatureCalculator(runtime_assets)
        
        print(f"✓ TMM initialized (using reference implementation)")
        print(f"  Runtime assets: {runtime_assets}")
    
    def compute(self, structure_str: str) -> dict:
        """
        计算结构的光学性能
        
        Args:
            structure_str: "WVO2_30 SiO2_100 TiO2_50 SiO2_10000"
        
        Returns:
            字典，兼容服务器现有格式
        """
        # 调用本地实现
        tmm_result = self.calculator.calculate(structure_str)
        features = self.optical.calculate(tmm_result)
        
        # 转换为服务器期望的字典格式
        return {
            # 八维特征
            'T_lum_cold': float(features['T_lum_cold']),
            'T_lum_hot': float(features['T_lum_hot']),
            'T_sol_cold': float(features['T_sol_cold']),
            'T_sol_hot': float(features['T_sol_hot']),
            'Rf_sol_cold': float(features['Rf_sol_cold']),
            'Rb_sol_cold': float(features['Rb_sol_cold']),
            'Rf_sol_hot': float(features['Rf_sol_hot']),
            'Rb_sol_hot': float(features['Rb_sol_hot']),
            
            # 波长数组
            'wavelengths_nm': tmm_result.cold.wavelengths_nm,
            
            # 完整光谱（用于EnergyPlus）
            'spectrum_cold': {
                'T': tmm_result.cold.transmittance,
                'Rf': tmm_result.cold.front_reflectance,
                'Rb': tmm_result.cold.back_reflectance,
            },
            'spectrum_hot': {
                'T': tmm_result.hot.transmittance,
                'Rf': tmm_result.hot.front_reflectance,
                'Rb': tmm_result.hot.back_reflectance,
            },
            
            # 额外信息
            'valid': True,
            'structure': structure_str,
        }
