"""
光谱扩展工具 v2 - 直接生成221维

统一为221维光谱，匹配Transformer训练数据
"""

import numpy as np
from typing import Dict, Union


def expand_8d_to_221d(target_8d: Union[Dict[str, float], np.ndarray]) -> np.ndarray:
    """
    将8维聚合目标直接扩展为221维光谱

    策略: 分段常数近似
    - 可见光区 (380-780nm): 使用T_lum值
    - 近红外区 (780-2500nm): 根据T_sol和T_lum反推

    注意: 只扩展透射率T，不扩展反射率Rf/Rb

    Args:
        target_8d: 8维目标
            - Dict形式: {'T_lum_cold': 0.5, 'T_sol_cold': 0.4, ...}
            - Array形式: np.array([T_lum_cold, T_lum_hot, T_sol_cold, ...])

    Returns:
        spectrum_221d: (221, 2) 数组
            [:,0] = T_cold(λ) - 冷态透射率光谱
            [:,1] = T_hot(λ)  - 热态透射率光谱

    波长范围: 300-2500nm, 10nm间隔, 221个点
    """
    # 转换为统一的dict格式
    if isinstance(target_8d, np.ndarray):
        if target_8d.shape[0] != 8:
            raise ValueError(f"Expected 8-dim array, got {target_8d.shape[0]}")

        target_dict = {
            'T_lum_cold': float(target_8d[0]),
            'T_lum_hot': float(target_8d[1]),
            'T_sol_cold': float(target_8d[2]),
            'T_sol_hot': float(target_8d[3]),
            'Rf_sol_cold': float(target_8d[4]),
            'Rb_sol_cold': float(target_8d[5]),
            'Rf_sol_hot': float(target_8d[6]),
            'Rb_sol_hot': float(target_8d[7]),
        }
    else:
        target_dict = target_8d

    # 波长数组: 300-2500nm, 10nm间隔, 221个点
    wavelengths = np.arange(300, 2501, 10, dtype=float)

    # 定义波段
    visible_mask = (wavelengths >= 380) & (wavelengths <= 780)   # VIS: 约40个点
    nir_mask = wavelengths > 780                                  # NIR: 约181个点

    # 初始化光谱
    spectrum_221d = np.zeros((221, 2))

    # === 冷态透射率 T_cold ===
    T_lum_cold = target_dict['T_lum_cold']  # 可见光透射率
    T_sol_cold = target_dict['T_sol_cold']  # 全太阳透射率

    # 可见光区: 直接使用T_lum
    spectrum_221d[visible_mask, 0] = T_lum_cold

    # 近红外区: 根据T_sol和T_lum反推
    # T_sol ≈ 0.5 * T_lum + 0.5 * T_nir
    # 因此: T_nir = 2 * T_sol - T_lum
    T_nir_cold = 2.0 * T_sol_cold - T_lum_cold
    T_nir_cold = np.clip(T_nir_cold, 0.0, 1.0)  # 限制在[0,1]
    spectrum_221d[nir_mask, 0] = T_nir_cold

    # === 热态透射率 T_hot ===
    T_lum_hot = target_dict['T_lum_hot']
    T_sol_hot = target_dict['T_sol_hot']

    spectrum_221d[visible_mask, 1] = T_lum_hot

    T_nir_hot = 2.0 * T_sol_hot - T_lum_hot
    T_nir_hot = np.clip(T_nir_hot, 0.0, 1.0)
    spectrum_221d[nir_mask, 1] = T_nir_hot

    # 应用平滑（避免780nm处突变）
    spectrum_221d[:, 0] = smooth_transition_221(spectrum_221d[:, 0], wavelengths, [780])
    spectrum_221d[:, 1] = smooth_transition_221(spectrum_221d[:, 1], wavelengths, [780])

    return spectrum_221d


def smooth_transition_221(spectrum: np.ndarray,
                         wavelengths: np.ndarray,
                         transition_points: list,
                         window_nm: float = 20.0) -> np.ndarray:
    """
    在波段交界处平滑过渡（221维版本）

    Args:
        spectrum: 光谱数组 (221,)
        wavelengths: 波长数组 (221,)
        transition_points: 过渡点波长列表 [780]
        window_nm: 过渡窗口宽度 (nm)

    Returns:
        平滑后的光谱
    """
    result = spectrum.copy()

    for wl_center in transition_points:
        # 找到过渡区域
        transition_mask = np.abs(wavelengths - wl_center) <= window_nm

        if not np.any(transition_mask):
            continue

        # 在过渡区域内进行线性插值
        indices = np.where(transition_mask)[0]

        if len(indices) < 2:
            continue

        # 获取边界值
        left_idx = indices[0] - 1 if indices[0] > 0 else 0
        right_idx = indices[-1] + 1 if indices[-1] < len(spectrum) - 1 else len(spectrum) - 1

        left_val = spectrum[left_idx]
        right_val = spectrum[right_idx]

        # 线性插值
        n_points = len(indices)
        interpolated = np.linspace(left_val, right_val, n_points)
        result[indices] = interpolated

    return result


def validate_expanded_spectrum_221(spectrum_221d: np.ndarray,
                                   target_8d: Dict[str, float],
                                   wavelengths: np.ndarray = None) -> Dict[str, float]:
    """
    验证扩展后的221维光谱是否匹配原始8维目标

    Args:
        spectrum_221d: (221, 2) 扩展后的光谱
        target_8d: 原始8维目标
        wavelengths: 波长数组 (221,)

    Returns:
        验证结果: {'T_lum_cold_error': 0.001, 'T_sol_cold_error': 0.002, ...}
    """
    if wavelengths is None:
        wavelengths = np.arange(300, 2501, 10, dtype=float)

    # 加载AM1.5太阳光谱权重
    solar_weights = get_am15_weights_221(wavelengths)

    # 定义波段
    visible_mask = (wavelengths >= 380) & (wavelengths <= 780)

    # 计算积分值
    T_cold = spectrum_221d[:, 0]
    T_hot = spectrum_221d[:, 1]

    # T_lum: 可见光区的平均
    T_lum_cold_computed = np.mean(T_cold[visible_mask])
    T_lum_hot_computed = np.mean(T_hot[visible_mask])

    # T_sol: 全波段的太阳加权积分
    T_sol_cold_computed = np.sum(T_cold * solar_weights) / np.sum(solar_weights)
    T_sol_hot_computed = np.sum(T_hot * solar_weights) / np.sum(solar_weights)

    # 计算误差
    errors = {
        'T_lum_cold_error': abs(T_lum_cold_computed - target_8d['T_lum_cold']),
        'T_lum_hot_error': abs(T_lum_hot_computed - target_8d['T_lum_hot']),
        'T_sol_cold_error': abs(T_sol_cold_computed - target_8d['T_sol_cold']),
        'T_sol_hot_error': abs(T_sol_hot_computed - target_8d['T_sol_hot']),
    }

    return errors


def get_am15_weights_221(wavelengths: np.ndarray) -> np.ndarray:
    """
    获取AM1.5太阳光谱权重（221维版本）

    Args:
        wavelengths: 波长数组 (nm)

    Returns:
        权重数组 (归一化)
    """
    # 使用Planck黑体辐射作为近似
    T_sun = 5778  # K (太阳表面温度)

    # 转换为米
    wl_m = wavelengths * 1e-9

    # Planck公式
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23

    intensity = (2 * h * c**2 / wl_m**5) / (np.exp(h * c / (wl_m * k * T_sun)) - 1)

    # 归一化
    intensity = intensity / np.sum(intensity)

    return intensity


# 向后兼容：保留旧函数名但标记为deprecated
def expand_8d_to_442d(target_8d):
    """
    已废弃：请使用expand_8d_to_221d

    为了向后兼容，这个函数仍然可用，但会返回221维并打印警告
    """
    print("Warning: expand_8d_to_442d is deprecated. Use expand_8d_to_221d instead.")
    return expand_8d_to_221d(target_8d)


# === 单元测试 ===
def test_expand_8d_to_221d():
    """测试8维到221维的扩展"""

    # 测试用例1: dict输入
    target_dict = {
        'T_lum_cold': 0.5,
        'T_lum_hot': 0.3,
        'T_sol_cold': 0.6,
        'T_sol_hot': 0.4,
        'Rf_sol_cold': 0.2,
        'Rb_sol_cold': 0.2,
        'Rf_sol_hot': 0.3,
        'Rb_sol_hot': 0.3,
    }

    spectrum = expand_8d_to_221d(target_dict)

    assert spectrum.shape == (221, 2), f"Shape error: {spectrum.shape}"
    assert np.all((spectrum >= 0) & (spectrum <= 1)), "Values out of [0,1]"

    print("✓ Dict input test passed")

    # 测试用例2: array输入
    target_array = np.array([0.5, 0.3, 0.6, 0.4, 0.2, 0.2, 0.3, 0.3])
    spectrum = expand_8d_to_221d(target_array)

    assert spectrum.shape == (221, 2), f"Shape error: {spectrum.shape}"
    print("✓ Array input test passed")

    # 测试用例3: 验证积分一致性
    wavelengths = np.arange(300, 2501, 10, dtype=float)
    errors = validate_expanded_spectrum_221(spectrum, target_dict, wavelengths)

    print(f"\n验证结果:")
    for key, error in errors.items():
        print(f"  {key}: {error:.4f}")
        assert error < 0.1, f"Error too large: {key} = {error}"

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_expand_8d_to_221d()
