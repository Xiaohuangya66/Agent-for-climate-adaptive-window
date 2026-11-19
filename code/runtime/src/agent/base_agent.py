"""
Agent核心基类

定义所有方法的统一接口
"""

import numpy as np
import pandas as pd
import torch
from src.config import ENERGYPLUS_EXE, CITY_EPW_MAP
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

class BaseAgent(ABC):
    """
    所有方法的基类

    统一接口确保公平比较
    """

    def __init__(self,
                 city: str,
                 seed: int,
                 budget: int = 40,
                 data_dir: str = 'data',
                 runtime_dir: str = 'runtime_assets'):
        """
        Args:
            city: 已训练的源城市，或经城市准备工具注册并具备真实基线的迁移城市
            seed: 随机种子
            budget: EnergyPlus查询预算
            data_dir: 数据目录路径
            runtime_dir: 运行时资产目录
        """
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 基本配置
        self.city = city
        from src.runtime_profile import city_assets
        self.runtime_assets = dict(city_assets(city))
        from src.agent.formal_protocol import FormalProtocol
        self.formal = FormalProtocol(city)
        self.seed = seed
        self.budget_total = budget
        self.budget_used = 0

        # 路径
        self.data_dir = Path(data_dir)
        self.runtime_dir = Path(runtime_dir)

        # 状态
        self.iteration = 0
        self.candidate_archive = []
        self.pareto_front = []
        self.query_history = []
        
        # Agent状态
        self.state = {}

        # 日志
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{city}_{seed}")

        # 加载工具（子类可以override）
        self._load_tools()

    def _load_tools(self):
        """加载基础工具"""
        from src.components.tmm import TMM
        from src.components.surrogate import Surrogate
        from src.components.energyplus_runner import EnergyPlusRunner

        self.tmm = TMM(
            optical_constants_dir=self.runtime_dir / "tmm" / "TMM_ML"
        )

        self.city_support = None
        if self.runtime_assets.get('support_mode') == 'transfer':
            from src.tools.city_support import CitySupportTool
            self.city_support = CitySupportTool(self.city, self.runtime_assets, self.formal, self.tmm).prepare()
            self.surrogate = self.city_support.predictor
        else:
            self.surrogate = Surrogate(model_path=Path(self.runtime_assets['model_directory']))

        self.energyplus = EnergyPlusRunner(
            idf_path=Path(self.runtime_assets['idf']),
            epw_path=Path(self.runtime_assets['epw']),
            energyplus_exe=ENERGYPLUS_EXE  # 服务器路径
        )

    @abstractmethod
    def run(self) -> Dict:
        """
        运行方法的主循环

        Returns:
            {
                'best_structure': str,
                'best_energy_saving': float,
                'best_udi': float,
                'pareto_front': List[Dict],
                'n_energyplus_queries': int,
                'computation_time': float,
                'method_specific_data': Dict  # 方法特定的数据
            }
        """
        pass

    def _extract_pareto_front(self,
                             candidates: List[Dict],
                             objectives: List[str] = ['energy', 'udi']) -> List[Dict]:
        """
        提取Pareto前沿

        Args:
            candidates: 候选列表，每个包含{'energy': float, 'udi': float}
            objectives: 目标列表（默认能耗最小化，UDI最大化）

        Returns:
            Pareto最优候选列表
        """
        if len(candidates) == 0:
            return []

        # 构建目标矩阵
        # 能耗：最小化（不变）
        # UDI：最大化（转为最小化负值）
        objectives_matrix = np.array([
            [c['energy'], -c['udi']]
            for c in candidates
        ])

        # 非支配排序
        pareto_indices = self._non_dominated_sort(objectives_matrix)

        return [candidates[i] for i in pareto_indices]

    def _non_dominated_sort(self, objectives: np.ndarray) -> List[int]:
        """
        非支配排序（NSGA-II算法）

        Args:
            objectives: [n_samples, n_objectives]

        Returns:
            第一前沿的索引列表
        """
        n = len(objectives)

        # 支配计数
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                # 比较i和j
                if self._dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # 第一前沿：支配计数为0
        front = [i for i in range(n) if domination_count[i] == 0]

        return front

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        判断obj1是否支配obj2（所有目标都不差，至少一个更好）
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    def _compute_energy_saving(self,
                              energy: float,
                              baseline_energy: Optional[float] = None) -> float:
        """
        计算能耗节省率

        Args:
            energy: 候选能耗 (J)
            baseline_energy: 清玻璃基线能耗 (J)

        Returns:
            节能率 (0-1)
        """
        if baseline_energy is None:
            # 从预先计算的baseline加载
            baseline_energy = self._get_baseline_energy()

        return (baseline_energy - energy) / baseline_energy

    def _get_baseline_energy(self) -> float:
        """获取清玻璃基线能耗"""
        return self.formal.energy

    def _get_baseline_udi(self) -> float:
        """获取清玻璃基线UDI"""
        return self.formal.clear_udi

    def _log_iteration(self, iteration: int, info: Dict):
        """记录迭代信息"""
        self.logger.info(
            f"Iteration {iteration}: "
            f"Budget {self.budget_used}/{self.budget_total}, "
            f"Candidates {len(self.candidate_archive)}, "
            f"Pareto size {len(self.pareto_front)}"
        )

        if len(self.pareto_front) > 0:
            best = max(self.pareto_front,
                      key=lambda x: x.get('energy_saving', 0))
            self.logger.info(
                f"  Best so far: {best.get('energy_saving', 0):.4f} "
                f"(UDI={best.get('udi', 0):.3f})"
            )

    def _update_pareto(self):
        """更新Pareto前沿"""
        verified = [c for c in self.query_history if c.get('energyplus_verified') and c.get('energy')]
        
        if len(verified) == 0:
            return
        
        # 简单Pareto：能耗低且UDI高
        pareto = []
        for candidate in verified:
            dominated = False
            for other in verified:
                if other['energy'] <= candidate['energy'] and other['udi'] >= candidate['udi'] and (other['energy'] < candidate['energy'] or other['udi'] > candidate['udi']):
                    dominated = True
                    break
            if not dominated:
                pareto.append(candidate)
        
        self.pareto_front = pareto

    def _compute_energy_saving(self, energy: float) -> float:
        """计算节能率"""
        from src.config import BASELINE_ENERGY
        baseline = self.formal.energy
        return (baseline - energy) / baseline
