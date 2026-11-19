"""
对比方法: 随机搜索基线

完全随机生成结构，用EnergyPlus验证，保留最优
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.agent.base_agent import BaseAgent
from src.utils.structure_utils import generate_random_structure
from src.components.energyplus_runner import EnergyPlusRunner


class RandomSearch(BaseAgent):
    """
    随机搜索基线方法

    流程:
    1. 随机生成结构
    2. 直接用EnergyPlus验证
    3. 保留最优结果
    """

    def __init__(self,
                 city: str,
                 seed: int,
                 budget: int,
                 data_dir: str = "data",
                 runtime_dir: str = "runtime_assets"):
        super().__init__(city, seed, budget, data_dir, runtime_dir)

        self.energyplus = EnergyPlusRunner()

    def run(self) -> Dict:
        """
        运行随机搜索

        Returns:
            {
                'best_structure': str,
                'best_energy_saving': float,
                'best_udi': float,
                'n_energyplus_queries': int
            }
        """
        print(f"\n{'='*60}")
        print("随机搜索基线")
        print(f"{'='*60}")
        print(f"城市: {self.city}")
        print(f"预算: {self.budget_total}")

        while self.budget_used < self.budget_total:
            # 生成随机结构
            structure = generate_random_structure(seed=self.seed + self.budget_used)

            print(f"\n[{self.budget_used+1}/{self.budget_total}] 验证随机结构...")

            # EnergyPlus验证
            ep_result = self.energyplus.run(structure, self.city)

            if ep_result['status'] == 'success':
                energy_saving = self._compute_energy_saving(ep_result['TotalEnergy'])

                self.query_history.append({
                    'structure': structure,
                    'energy': ep_result['TotalEnergy'],
                    'udi': ep_result['UDI'],
                    'energy_saving': energy_saving,
                    'energyplus_verified': True
                })

                print(f"  节能率: {energy_saving:.4f}, UDI: {ep_result['UDI']:.3f}")
            else:
                print(f"  ✗ EnergyPlus失败")

            self.budget_used += 1

        # 返回最优结果
        result = self._get_best_result()
        result['method_specific_data'] = {'note': 'Random baseline'}

        return result

    def _get_best_result(self) -> Dict:
        """获取最优结果"""
        verified = [q for q in self.query_history if q.get('energyplus_verified')]

        if len(verified) == 0:
            return {
                'best_structure': '',
                'best_energy_saving': 0.0,
                'best_udi': 0.0,
                'n_energyplus_queries': self.budget_used
            }

        # 按节能率排序
        verified.sort(key=lambda x: x['energy_saving'], reverse=True)
        best = verified[0]

        return {
            'best_structure': best['structure'],
            'best_energy_saving': best['energy_saving'],
            'best_udi': best['udi'],
            'n_energyplus_queries': self.budget_used
        }
