"""
项目配置文件

统一管理所有路径和参数
"""

from pathlib import Path
from typing import Dict, List

# ========================================
# 路径配置
# ========================================

from src.runtime_profile import PROJECT_ROOT, CAMPAIGN_ROOT, MODEL_RUN, baseline_rows, registry
from src.support_settings import settings

# 数据路径
DATA_DIR = PROJECT_ROOT / "data"
ENERGYPLUS_RAW_DIR = CAMPAIGN_ROOT / "results"
SURROGATES_DIR = MODEL_RUN / "models"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"

# 运行时资产
RUNTIME_DIR = PROJECT_ROOT / "runtime_assets"
EPW_DIR = CAMPAIGN_ROOT / "weather"
IDF_DIR = RUNTIME_DIR / "idf"
TMM_DIR = RUNTIME_DIR / "tmm"

# 代码路径
SRC_DIR = PROJECT_ROOT / "src"
REFERENCE_DIR = SRC_DIR / "reference"

# 输出路径
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# ========================================
# EnergyPlus配置
# ========================================

ENERGYPLUS_EXE = settings().get('energyplus_exe',"/public/home/nju_zhanghuajie/EnergyPlus23.1.0/energyplus")
ENERGYPLUS_VERSION = "23.1.0"

# ========================================
# 城市配置
# ========================================

CITIES = list(registry()['source_cities']) + list(settings().get('targets',{}))

CITY_EPW_MAP = {city:Path(registry()['cities'][city]['epw']).name for city in CITIES}

# 清玻璃基线（预先计算的值）
BASELINE_ENERGY = {city:float(baseline_rows()[city]['TotalEnergy_J']) for city in CITIES if city in baseline_rows()}

BASELINE_UDI = {city:float(baseline_rows()[city]['UDI']) for city in CITIES if city in baseline_rows()}

# ========================================
# 结构约束
# ========================================

STRUCTURE_CONSTRAINTS = {
    "wvo2_count": 0,                    # 0=不限制WVO₂层数（已放开）
    "wvo2_thickness_range": (5, 10000), # 5-10000 nm（已放开厚度限制）
    "terminal_layer": None,             # None=不要求特定末层（已放开）
    "max_layers": 30,                   # 最大层数（保持合理限制）
    "allowed_materials": [
        "WVO2", "SiO2", "TiO2", "ZnO",
        "Ag", "ITO", "WO3", "MgF2", "Ta2O5"
    ],
    "thickness_discretization": 5,      # 厚度离散化为5nm倍数
    "min_thickness": 5,                 # 最小厚度
    "max_thickness": 500,               # 最大厚度
}
STRUCTURE_CONSTRAINTS.update(settings().get('structure_constraints', {}))

# ========================================
# 实验配置
# ========================================

RANDOM_SEEDS = [5, 7, 11, 17]
DEFAULT_BUDGET = 40
UDI_THRESHOLD_RATIO = 0.90  # Same-city clear-glass UDI retention.

# 固定目标光谱（用于baseline）
FIXED_TARGET_SPECTRUM = {
    "T_lum_cold": 0.5,
    "T_lum_hot": 0.5,
    "T_sol_cold": 1.0,
    "T_sol_hot": 0.0,
    "Rf_sol_cold": 0.7,
    "Rb_sol_cold": 0.3,
    "Rf_sol_hot": 0.7,
    "Rb_sol_hot": 0.3,
}

# ========================================
# 并行计算配置
# ========================================

N_PARALLEL_DEFAULT = 20  # 默认并行数
TIMEOUT_ENERGYPLUS = 7200  # EnergyPlus超时时间（秒）

# ========================================
# 日志配置
# ========================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
