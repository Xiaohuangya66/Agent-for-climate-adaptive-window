"""
对比方法: 城市先验学习 + 无在线目标修正

用于验证在线目标修正的有效性（创新点2的消融实验）
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.agent.base_agent import BaseAgent
from src.tools.city_prior_learner import CityPriorLearner
from src.components.transformer_wrapper import TransformerWrapper
from src.components.tmm import TMM
from src.components.surrogate import Surrogate
from src.components.energyplus_runner import EnergyPlusRunner
from src.config import FIXED_TARGET_SPECTRUM


class CityPriorNoRefine(BaseAgent):
    """
    城市先验学习但不做在线修正

    流程:
    1. 从城市历史数据学习先验目标
    2. 固定使用该目标，不再修正
    3. 其他步骤与Agent_Full相同
    """

    def __init__(self,
                 city: str,
                 seed: int,
                 budget: int,
                 data_dir: str = "data",
                 runtime_dir: str = "runtime_assets"):
        super().__init__(city, seed, budget, data_dir, runtime_dir)

        # 初始化组件
        self.city_prior_learner = CityPriorLearner(data_dir=self.data_dir)
        self.transformer = TransformerWrapper()
        self.tmm = TMM(self.runtime_dir / 'tmm' / 'TMM_ML')
        self.surrogate = Surrogate(self.data_dir / 'surrogates' / f'surrogate_model_{city}.pkl')
        self.energyplus = EnergyPlusRunner()

        # 目标光谱（固定，不修正）
        self.target_spectrum = None
        self.target_confidence = 0.0

    def run(self) -> Dict:
        """
        运行优化

        Returns:
            {
                'best_structure': str,
                'best_energy_saving': float,
                'best_udi': float,
                'n_energyplus_queries': int,
                'method_specific_data': {...}
            }
        """
        # Phase 1: 学习城市先验目标（仅一次）
        print(f"\n{'='*60}")
        print("Phase 1: 学习城市先验目标")
        print(f"{'='*60}")

        prior_result = self.city_prior_learner.learn_prior(self.city)

        self.target_spectrum = prior_result['target_spectrum']
        self.target_confidence = prior_result['confidence']

        print(f"[城市先验] 目标光谱: Tsol_cold={self.target_spectrum['T_sol_cold']:.3f}, "
              f"Tsol_hot={self.target_spectrum['T_sol_hot']:.3f}")
        print(f"[城市先验] 置信度: {self.target_confidence:.3f}")
        print(f"[城市先验] 目标将保持固定，不做在线修正")

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
            'note': 'No online target refinement'
        }

        return result

    def _generate_candidates(self):
        """生成候选结构（使用固定目标光谱）"""
        n_candidates = 128

        # 使用固定目标生成
        structures = self.transformer.generate(
            target_spectrum=self.target_spectrum,
            n=n_candidates,
            temperature=1.0,
            top_k=50
        )

        self.candidate_archive.extend(structures)

        print(f"[生成] 生成了 {len(structures)} 个候选结构")
        print(f"[生成] 使用固定目标: Tsol_cold={self.target_spectrum['T_sol_cold']:.3f}")

    def _select_and_verify(self, n_verify: int):
        """选择并用EnergyPlus验证Top-K候选"""
        if len(self.candidate_archive) == 0:
            return

        # 使用代理模型评估
        candidates_with_scores = []

        for structure in self.candidate_archive:
            # 计算光学属性
            tmm_result = self.tmm.compute(structure)
            optical = {
                'T_lum_cold': tmm_result['optical_cold']['T_lum'],
                'T_lum_hot': tmm_result['optical_hot']['T_lum'],
                'T_sol_cold': tmm_result['optical_cold']['T_sol'],
                'T_sol_hot': tmm_result['optical_hot']['T_sol'],
                'Rf_sol_cold': tmm_result['optical_cold']['Rf_sol'],
                'Rb_sol_cold': tmm_result['optical_cold']['Rb_sol'],
                'Rf_sol_hot': tmm_result['optical_hot']['Rf_sol'],
                'Rb_sol_hot': tmm_result['optical_hot']['Rb_sol'],
            }

            # 代理模型预测
            surrogate_pred = self.surrogate.predict(optical)

            candidates_with_scores.append({
                'structure': structure,
                'predicted_energy': surrogate_pred['TotalEnergy'],
                'predicted_udi': surrogate_pred['UDI'],
                'optical': optical
            })

        # 按预测能耗排序
        candidates_with_scores.sort(key=lambda x: x['predicted_energy'])

        # 选择Top-K
        top_k = candidates_with_scores[:min(n_verify, len(candidates_with_scores))]

        print(f"[选择] 从 {len(candidates_with_scores)} 个候选中选择 {len(top_k)} 个验证")

        # EnergyPlus验证
        for i, candidate in enumerate(top_k):
            if self.budget_used >= self.budget_total:
                break

            print(f"  [{i+1}/{len(top_k)}] 验证候选...")

            ep_result = self.energyplus.run(candidate['structure'], self.city)

            if ep_result['status'] == 'success':
                energy_saving = self._compute_energy_saving(ep_result['TotalEnergy'])

                self.query_history.append({
                    'structure': candidate['structure'],
                    'energy': ep_result['TotalEnergy'],
                    'udi': ep_result['UDI'],
                    'energy_saving': energy_saving,
                    'energyplus_verified': True,
                    'iteration': self.iteration
                })

                print(f"    节能率: {energy_saving:.4f}, UDI: {ep_result['UDI']:.3f}")

            self.budget_used += 1

        # 清空候选池
        self.candidate_archive = []

    def _check_convergence(self) -> bool:
        """检查是否收敛（简单版本）"""
        if len(self.query_history) < 5:
            return False

        # 检查最近5次查询的节能率是否变化很小
        recent = self.query_history[-5:]
        savings = [q['energy_saving'] for q in recent if q.get('energyplus_verified')]

        if len(savings) < 5:
            return False

        improvement = max(savings) - min(savings)
        return improvement < 0.01  # 改进<1%则收敛

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
