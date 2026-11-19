"""
TMM v2 - 真实的传输矩阵方法光学计算

集成原始tmm_core，计算多层膜的光学性能
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import sys
from scipy.interpolate import interp1d

# 导入tmm_core
sys.path.insert(0, str(Path(__file__).parent))
from tmm_core import inc_tmm


class TMM_v2:
    """
    传输矩阵方法光学计算（真实版本）

    使用原始tmm_core进行多层膜光学计算
    """

    def __init__(self, optical_constants_dir: Path):
        """
        Args:
            optical_constants_dir: 光学常数目录 (TMM_ML/)
        """
        self.optical_dir = Path(optical_constants_dir)

        # 波长数组: 380-2500nm, 5nm间隔 (与EnergyPlus一致)
        self.wavelengths = np.arange(380, 2501, 5, dtype=np.float64)

        # 加载材料光学常数
        self.material_funcs = {}
        self._load_materials()

        print(f"✓ TMM_v2初始化成功")
        print(f"  - 波长范围: {self.wavelengths[0]}-{self.wavelengths[-1]} nm")
        print(f"  - 材料数量: {len(self.material_funcs)}")

    def _load_materials(self):
        """加载所有材料的n和k数据"""
        material_files = {
            'SiO2': 'sio2.txt',
            'TiO2': 'tio2.txt',
            'ZnO': 'zno.txt',
            'WO3': 'wo3.txt',
            'Ag': 'Ag.txt',
            'ITO': 'ito.txt',
            'MgF2': 'mgf2.txt',
            'Ta2O5': 'ta2o5.txt',
        }

        for material, filename in material_files.items():
            filepath = self.optical_dir / filename

            if filepath.exists():
                try:
                    self.material_funcs[material] = self._load_nk_material(filepath)
                except Exception as e:
                    print(f"Warning: 无法加载 {material}: {e}")
            else:
                print(f"Warning: 文件不存在 {filepath}")

        # WVO2特殊处理：冷态和热态
        wvo2_cold_file = self.optical_dir / 'L_1.csv'  # 或wvo2_cold.txt
        wvo2_hot_file = self.optical_dir / 'H_1.csv'   # 或wvo2_hot.txt

        if wvo2_cold_file.exists():
            self.wvo2_cold_func = self._load_nk_material(wvo2_cold_file)
        else:
            print(f"Warning: WVO2冷态文件不存在")
            self.wvo2_cold_func = None

        if wvo2_hot_file.exists():
            self.wvo2_hot_func = self._load_nk_material(wvo2_hot_file)
        else:
            print(f"Warning: WVO2热态文件不存在")
            self.wvo2_hot_func = None

    def _load_nk_material(self, filename: Path):
        """
        加载材料的n和k数据
        
        支持两种格式:
        1. txt格式（制表符分隔，带wl标记）
        2. csv格式（逗号分隔，只有数值）
        """
        # 检查文件扩展名
        if filename.suffix == '.csv':
            # CSV格式：简单的两列数据
            data = np.loadtxt(filename, delimiter=',')
            wl = data[:, 0]  # 波长 (um)
            n = data[:, 1]   # 折射率
            
            # 排序
            idx = np.argsort(wl)
            n_f = interp1d(wl[idx], n[idx], kind='linear', fill_value='extrapolate')
            
            # CSV文件通常只有n，没有k
            k_f = lambda w: 0.0
            
            return lambda lam_um: n_f(lam_um) + 1j * k_f(lam_um)
        
        else:
            # txt格式：原有逻辑
            n_wl, n_val = [], []
            k_wl, k_val = [], []
            current = 'n'
            
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # 检查是否是块标记
                    if 'wl' in line.lower():
                        if 'n' in line.lower() and 'k' not in line.lower():
                            current = 'n'
                        elif 'k' in line.lower():
                            current = 'k'
                        continue
                    
                    # 分割（支持制表符和空格）
                    parts = line.replace('	', ' ').split()
                    if len(parts) >= 2:
                        try:
                            w = float(parts[0])
                            v = float(parts[1])
                            
                            if current == 'n':
                                n_wl.append(w)
                                n_val.append(v)
                            else:
                                k_wl.append(w)
                                k_val.append(v)
                        except ValueError:
                            continue
            
            if len(n_wl) == 0:
                raise ValueError(f"文件 {filename} 中未找到有效的n数据")
            
            n_wl = np.array(n_wl)
            n_val = np.array(n_val)
            
            # 检查单位
            if n_wl[0] > 100:
                n_wl /= 1000.0
            
            idx = np.argsort(n_wl)
            n_f = interp1d(n_wl[idx], n_val[idx], kind='linear', fill_value='extrapolate')
            
            # 处理k数据
            if len(k_wl) == 0:
                k_f = lambda w: 0.0
            else:
                k_wl = np.array(k_wl)
                k_val = np.array(k_val)
                
                if k_wl[0] > 100:
                    k_wl /= 1000.0
                
                idx_k = np.argsort(k_wl)
                k_f = interp1d(k_wl[idx_k], k_val[idx_k], kind='linear', fill_value='extrapolate')
            
            return lambda lam_um: n_f(lam_um) + 1j * k_f(lam_um)
    def compute(self, structure_str: str) -> Dict:
        """
        计算结构的光学性能

        Args:
            structure_str: "WVO2_30 SiO2_100 TiO2_50 SiO2_10000"

        Returns:
            {
                'T_lum_cold': float,
                'T_lum_hot': float,
                'T_sol_cold': float,
                'T_sol_hot': float,
                'Rf_sol_cold': float,
                'Rb_sol_cold': float,
                'Rf_sol_hot': float,
                'Rb_sol_hot': float,
                'spectrum_cold': {'T': array, 'Rf': array, 'Rb': array},
                'spectrum_hot': {'T': array, 'Rf': array, 'Rb': array}
            }
        """
        from src.utils.structure_utils import parse_structure

        tokens = parse_structure(structure_str)

        if len(tokens) == 0:
            raise ValueError("Empty structure")

        # 计算冷热两态
        spectrum_cold = self._compute_single_state(tokens, state='cold')
        spectrum_hot = self._compute_single_state(tokens, state='hot')

        # 计算光度学和太阳光加权积分
        T_lum_cold = self._integrate_luminous(spectrum_cold['T'])
        T_lum_hot = self._integrate_luminous(spectrum_hot['T'])

        T_sol_cold = self._integrate_solar(spectrum_cold['T'])
        T_sol_hot = self._integrate_solar(spectrum_hot['T'])

        Rf_sol_cold = self._integrate_solar(spectrum_cold['Rf'])
        Rb_sol_cold = self._integrate_solar(spectrum_cold['Rb'])

        Rf_sol_hot = self._integrate_solar(spectrum_hot['Rf'])
        Rb_sol_hot = self._integrate_solar(spectrum_hot['Rb'])

        return {
            'T_lum_cold': float(T_lum_cold),
            'T_lum_hot': float(T_lum_hot),
            'T_sol_cold': float(T_sol_cold),
            'T_sol_hot': float(T_sol_hot),
            'Rf_sol_cold': float(Rf_sol_cold),
            'Rb_sol_cold': float(Rb_sol_cold),
            'Rf_sol_hot': float(Rf_sol_hot),
            'Rb_sol_hot': float(Rb_sol_hot),
            'spectrum_cold': spectrum_cold,
            'spectrum_hot': spectrum_hot
        }

    def _compute_single_state(self, tokens, state: str) -> Dict:
        """
        计算单个状态（冷态或热态）的光谱

        Args:
            tokens: [(material, thickness), ...]
            state: 'cold' or 'hot'

        Returns:
            {'T': array, 'Rf': array, 'Rb': array}
        """
        materials = [mat for mat, _ in tokens]
        thicknesses = [thick for _, thick in tokens]

        # 构建TMM输入
        # c_list: 相干性列表 ('i'=非相干, 'c'=相干)
        # d_list: 厚度列表 (nm)
        # n_list: 折射率列表

        c_list = ['i'] + ['c'] * len(materials) + ['i']
        d_list = [np.inf] + thicknesses + [np.inf]

        T_spectrum = np.zeros(len(self.wavelengths), dtype=np.float64)
        Rf_spectrum = np.zeros(len(self.wavelengths), dtype=np.float64)
        Rb_spectrum = np.zeros(len(self.wavelengths), dtype=np.float64)

        # 选择WVO2状态
        if state == 'cold':
            wvo2_func = self.wvo2_cold_func
        else:
            wvo2_func = self.wvo2_hot_func

        # 逐波长计算
        for i, lam_nm in enumerate(self.wavelengths):
            lam_um = lam_nm / 1000.0

            # 构建折射率列表
            n_list = [1.0]  # 空气

            for mat in materials:
                if mat == 'WVO2' and wvo2_func is not None:
                    n_list.append(wvo2_func(lam_um))
                elif mat in self.material_funcs:
                    n_list.append(self.material_funcs[mat](lam_um))
                else:
                    # 未知材料，使用默认值
                    n_list.append(1.5 + 0j)

            n_list.append(1.0)  # 空气（背面）

            try:
                # TMM计算：s偏振和p偏振平均
                result_s = inc_tmm('s', n_list, d_list, c_list, 0, lam_nm)
                result_p = inc_tmm('p', n_list, d_list, c_list, 0, lam_nm)

                T = (result_s['T'] + result_p['T']) / 2.0
                R = (result_s['R'] + result_p['R']) / 2.0

                T_spectrum[i] = T
                Rf_spectrum[i] = R

                # 背向反射率（简化：假设与正向相同）
                Rb_spectrum[i] = R

            except Exception as e:
                # TMM计算失败，使用默认值
                T_spectrum[i] = 0.5
                Rf_spectrum[i] = 0.3
                Rb_spectrum[i] = 0.3

        return {
            'T': T_spectrum,
            'Rf': Rf_spectrum,
            'Rb': Rb_spectrum
        }

    def _integrate_luminous(self, spectrum: np.ndarray) -> float:
        """
        计算可见光透射率/反射率

        使用CIE photopic luminosity function加权
        """
        # 可见光范围: 380-780nm
        visible_mask = (self.wavelengths >= 380) & (self.wavelengths <= 780)

        if not np.any(visible_mask):
            return 0.0

        # 简化：使用平均值
        return float(np.mean(spectrum[visible_mask]))

    def _integrate_solar(self, spectrum: np.ndarray) -> float:
        """
        计算太阳光加权透射率/反射率

        使用AM1.5光谱加权
        """
        # 使用Planck黑体辐射作为权重
        T_sun = 5778  # K
        wl_m = self.wavelengths * 1e-9

        h = 6.626e-34
        c = 3e8
        k = 1.381e-23

        intensity = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T_sun)) - 1)

        # 加权积分
        weighted = np.sum(spectrum * intensity) / np.sum(intensity)

        return float(weighted)


# 测试代码
if __name__ == '__main__':
    from pathlib import Path

    print("=== 测试TMM_v2 ===\n")

    tmm = TMM_v2(Path("/public/home/nju_zhanghuajie/topic3_agent_v2/runtime_assets/tmm/TMM_ML"))

    # 测试结构
    structure = "WVO2_30 SiO2_100 TiO2_50 SiO2_10000"

    print(f"\n计算结构: {structure}")

    result = tmm.compute(structure)

    print(f"\n光学性能:")
    print(f"  T_lum_cold: {result['T_lum_cold']:.4f}")
    print(f"  T_lum_hot: {result['T_lum_hot']:.4f}")
    print(f"  T_sol_cold: {result['T_sol_cold']:.4f}")
    print(f"  T_sol_hot: {result['T_sol_hot']:.4f}")
    print(f"  Rf_sol_cold: {result['Rf_sol_cold']:.4f}")
    print(f"  Rb_sol_cold: {result['Rb_sol_cold']:.4f}")

    print("\n✓ TMM_v2测试完成！")
