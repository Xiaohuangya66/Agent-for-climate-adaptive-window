"""
对比方法: 固定目标光谱（AM1.5标准）

不使用城市先验学习，直接使用AM1.5太阳光谱作为目标
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.agent.city_prior_no_refine import CityPriorNoRefine
from src.config import FIXED_TARGET_SPECTRUM


class FixedTarget(CityPriorNoRefine):
    """
    固定目标光谱方法

    使用AM1.5标准太阳光谱作为固定目标，不学习城市先验
    """

    def __init__(self,
                 city: str,
                 seed: int,
                 budget: int,
                 data_dir: str = "data",
                 runtime_dir: str = "runtime_assets"):
        super().__init__(city, seed, budget, data_dir, runtime_dir)

    def run(self) -> Dict:
        """
        运行优化

        与CityPriorNoRefine的区别：
        - 跳过城市先验学习
        - 直接使用FIXED_TARGET_SPECTRUM
        """
        # Phase 1: 使用固定目标（AM1.5）
        print(f"\n{'='*60}")
        print("Phase 1: 使用固定目标光谱（AM1.5）")
        print(f"{'='*60}")

        self.target_spectrum = FIXED_TARGET_SPECTRUM.copy()
        self.target_confidence = 1.0  # 固定目标，置信度=1

        print(f"[固定目标] Tsol_cold={self.target_spectrum['T_sol_cold']:.3f}, "
              f"Tsol_hot={self.target_spectrum['T_sol_hot']:.3f}")
        print(f"[固定目标] 不考虑城市特异性")

        # Phase 2: 迭代优化（使用固定目标）
        print(f"\n{'='*60}")
        print("Phase 2: 迭代优化（固定目标）")
        print(f"{'='*60}")

        while self.budget_used < self.budget_total:
            self.iteration += 1
            remaining = self.budget_total - self.budget_used

            print(f"\n{'─'*60}")
            print(f"迭代 {self.iteration} (剩余预算: {remaining})")
            print(f"{'─'*60}")

            # 生成候选
            self._generate_candidates()

            # 选择Top-K用EnergyPlus验证
            self._select_and_verify(remaining)

            # 更新Pareto前沿
            self._update_pareto()

            # 检查收敛
            if self._check_convergence():
                print(f"[收敛] 已收敛，提前停止")
                break

        # Phase 3: 返回结果
        result = self._get_best_result()

        result['method_specific_data'] = {
            'target_spectrum': self.target_spectrum,
            'target_confidence': self.target_confidence,
            'n_iterations': self.iteration,
            'note': 'Fixed AM1.5 target spectrum (no city prior)'
        }

        return result
