"""
结构工具函数

处理结构的解析、验证、生成
"""

import re
import numpy as np
from typing import List, Tuple, Optional, Dict
from src.config import STRUCTURE_CONSTRAINTS


def parse_structure(structure_str: str) -> List[Tuple[str, int]]:
    """
    解析结构字符串为token列表

    Args:
        structure_str: "WVO2_30 SiO2_100 TiO2_50 SiO2_10000"

    Returns:
        [("WVO2", 30), ("SiO2", 100), ("TiO2", 50), ("SiO2", 10000)]
    """
    if not structure_str or structure_str.strip() == "":
        return []

    tokens = []
    parts = structure_str.strip().split()

    for part in parts:
        match = re.match(r'([A-Za-z0-9]+)_(\d+)', part)
        if match:
            material = match.group(1)
            thickness = int(match.group(2))
            tokens.append((material, thickness))

    return tokens


def serialize_structure(tokens: List[Tuple[str, int]]) -> str:
    """
    将token列表序列化为结构字符串

    Args:
        tokens: [("WVO2", 30), ("SiO2", 100)]

    Returns:
        "WVO2_30 SiO2_100"
    """
    return " ".join([f"{mat}_{thick}" for mat, thick in tokens])


def validate_structure(structure_str: str) -> Tuple[bool, str]:
    """
    验证结构是否满足约束（使用宽松约束）

    Returns:
        (is_valid, error_message)
    """
    tokens = parse_structure(structure_str)

    if len(tokens) == 0:
        return False, "Empty structure"

    # 检查层数
    max_layers = STRUCTURE_CONSTRAINTS.get("max_layers", 30)
    if len(tokens) > max_layers:
        return False, f"Too many layers: {len(tokens)} > {max_layers}"

    # 检查WVO2数量（0表示不限制）
    required_wvo2 = STRUCTURE_CONSTRAINTS.get("wvo2_count", 0)
    if required_wvo2 > 0:
        wvo2_count = sum(1 for mat, _ in tokens if mat == "WVO2")
        if wvo2_count != required_wvo2:
            return False, f"WVO2 count must be {required_wvo2}, got {wvo2_count}"

    # 检查WVO2厚度（宽松范围）
    wvo2_range = STRUCTURE_CONSTRAINTS.get("wvo2_thickness_range", (5, 10000))
    for mat, thick in tokens:
        if mat == "WVO2":
            if not (wvo2_range[0] <= thick <= wvo2_range[1]):
                return False, f"WVO2 thickness {thick} not in range {wvo2_range}"

    # 检查末层（None表示不限制）
    terminal = STRUCTURE_CONSTRAINTS.get("terminal_layer", None)
    if terminal is not None:
        last_token = f"{tokens[-1][0]}_{tokens[-1][1]}"
        if last_token != terminal:
            return False, f"Last layer must be {terminal}, got {last_token}"

    return True, ""


def repair_structure(structure_str: str) -> str:
    """
    修复结构使其满足约束

    用于NSGA-II的约束修复
    """
    tokens = parse_structure(structure_str)

    if len(tokens) == 0:
        # 生成随机合法结构
        return generate_random_structure()

    # 修复WVO2数量
    wvo2_indices = [i for i, (mat, _) in enumerate(tokens) if mat == "WVO2"]

    if len(wvo2_indices) == 0:
        # 没有WVO2，插入一个
        pos = np.random.randint(0, len(tokens))
        thickness = np.random.choice(range(20, 41, 5))
        tokens.insert(pos, ("WVO2", thickness))
    elif len(wvo2_indices) > 1:
        # 多个WVO2，只保留第一个
        for idx in sorted(wvo2_indices[1:], reverse=True):
            del tokens[idx]

    # 修复WVO2厚度
    wvo2_idx = [i for i, (mat, _) in enumerate(tokens) if mat == "WVO2"][0]
    mat, thick = tokens[wvo2_idx]
    if not (20 <= thick <= 40):
        thick = np.clip(thick, 20, 40)
        thick = round(thick / 5) * 5  # 离散化
        tokens[wvo2_idx] = (mat, thick)

    # 修复末层
    if tokens[-1][0] != "SiO2" or tokens[-1][1] != 10000:
        # 检查是否已有SiO2_10000
        base_idx = [i for i, (mat, thick) in enumerate(tokens) if mat == "SiO2" and thick == 10000]
        if base_idx:
            # 移动到末尾
            tokens.append(tokens.pop(base_idx[0]))
        else:
            # 添加
            tokens.append(("SiO2", 10000))

    # 限制层数
    if len(tokens) > STRUCTURE_CONSTRAINTS["max_layers"]:
        # 保留WVO2和基底，随机删除其他层
        wvo2_idx = [i for i, (mat, _) in enumerate(tokens) if mat == "WVO2"][0]
        base_idx = len(tokens) - 1

        keep_indices = {wvo2_idx, base_idx}
        other_indices = [i for i in range(len(tokens)) if i not in keep_indices]

        n_to_keep = STRUCTURE_CONSTRAINTS["max_layers"] - 2
        if n_to_keep > 0:
            keep_others = np.random.choice(other_indices,
                                          size=min(n_to_keep, len(other_indices)),
                                          replace=False)
            keep_indices.update(keep_others)

        tokens = [tokens[i] for i in sorted(keep_indices)]

    return serialize_structure(tokens)


def generate_random_structure(n_layers: Optional[int] = None, *, seed=None, rng=None) -> str:
    """
    生成随机的合法结构

    Args:
        n_layers: 层数（不包括基底），如果为None则随机

    Returns:
        结构字符串
    """
    if seed is not None and rng is not None:
        raise ValueError("Pass seed or rng, not both")
    generator = rng if rng is not None else (np.random.default_rng(seed) if seed is not None else np.random)
    randint = generator.integers if hasattr(generator, "integers") else generator.randint
    if n_layers is None:
        n_layers = randint(3, STRUCTURE_CONSTRAINTS["max_layers"])

    if not 1 <= n_layers < STRUCTURE_CONSTRAINTS["max_layers"]:
        raise ValueError("n_layers excludes the substrate and must fit max_layers")

    tokens = []

    # 随机选择WVO2位置
    wvo2_pos = randint(0, n_layers)

    allowed_materials = [m for m in STRUCTURE_CONSTRAINTS["allowed_materials"]
                        if m != "WVO2"]

    for i in range(n_layers):
        if i == wvo2_pos:
            # WVO2层
            material = "WVO2"
            thickness = generator.choice(range(20, 41, 5))
        else:
            # 其他材料
            material = generator.choice(allowed_materials)
            thickness = generator.choice(range(5, 201, 5))  # 5-200nm

        tokens.append((material, thickness))

    # 添加基底
    tokens.append(("SiO2", 10000))

    return serialize_structure(tokens)


def get_wvo2_position(structure_str: str) -> Optional[int]:
    """
    获取WVO2层的位置（从外到内，0-indexed）

    Returns:
        WVO2层的索引，如果没有则返回None
    """
    tokens = parse_structure(structure_str)
    for i, (mat, _) in enumerate(tokens):
        if mat == "WVO2":
            return i
    return None


def get_structure_features(structure_str: str) -> Dict:
    """
    提取结构的特征（用于分析）

    Returns:
        {
            'n_layers': int,
            'total_thickness': float,
            'wvo2_position': int,
            'wvo2_thickness': int,
            'has_ag': bool,
            'has_ito': bool,
            'material_counts': Dict[str, int]
        }
    """
    tokens = parse_structure(structure_str)

    if len(tokens) == 0:
        return {}

    # 计算总厚度（不包括基底）
    total_thickness = sum(thick for mat, thick in tokens[:-1])

    # WVO2信息
    wvo2_pos = None
    wvo2_thick = None
    for i, (mat, thick) in enumerate(tokens):
        if mat == "WVO2":
            wvo2_pos = i
            wvo2_thick = thick
            break

    # 材料统计
    material_counts = {}
    for mat, _ in tokens[:-1]:  # 不包括基底
        material_counts[mat] = material_counts.get(mat, 0) + 1

    return {
        'n_layers': len(tokens) - 1,  # 不包括基底
        'total_thickness': total_thickness,
        'wvo2_position': wvo2_pos,
        'wvo2_thickness': wvo2_thick,
        'has_ag': 'Ag' in material_counts,
        'has_ito': 'ITO' in material_counts,
        'material_counts': material_counts
    }


def validate_and_fix_structure(structure: str) -> str:
    """
    验证并修复结构字符串
    
    Args:
        structure: "SiO2_100 TiO2_50 ..."
        
    Returns:
        修复后的结构
    """
    if not structure or structure.strip() == '':
        # 返回默认结构
        return "SiO2_100 TiO2_50 SiO2_10000"
    
    layers = structure.strip().split()
    fixed_layers = []
    
    for layer in layers:
        # 检查格式
        if '_' not in layer:
            continue
        
        parts = layer.split('_')
        if len(parts) != 2:
            continue
        
        material, thickness_str = parts
        
        try:
            thickness = int(thickness_str)
            # 限制厚度范围
            thickness = max(5, min(thickness, 10000))
            fixed_layers.append(f"{material}_{thickness}")
        except ValueError:
            continue
    
    if len(fixed_layers) == 0:
        return "SiO2_100 TiO2_50 SiO2_10000"
    
    # 不再强制末层（末层约束已放开）
    # # 不再强制添加SiO2_10000末层
    
    return ' '.join(fixed_layers)
