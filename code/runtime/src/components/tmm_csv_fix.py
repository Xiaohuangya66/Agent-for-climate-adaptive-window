import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def _load_nk_material_fixed(self, filename: Path):
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
                parts = line.replace('\t', ' ').split()
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
        
        # 检查单位（如果>100则是nm，需要转um）
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

print("修复函数已准备")
