"""
迁移学习工具

实现创新点3：从其他城市迁移知识
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class TransferLearner:
    """
    迁移学习工具

    从源城市学习特征重要性和映射关系，迁移到目标城市
    """

    def __init__(self,
                 source_cities: List[str],
                 target_city: str,
                 data_dir: Path):
        """
        Args:
            source_cities: 源城市列表（用于迁移学习）
            target_city: 目标城市
            data_dir: 数据目录
        """
        self.source_cities = source_cities
        self.target_city = target_city
        self.data_dir = Path(data_dir)

        # 迁移模型
        self.feature_importance_model = None
        self.domain_adapter = None
        self.scaler_source = StandardScaler()
        self.scaler_target = StandardScaler()

    def learn_transfer(self) -> Dict:
        """
        学习迁移知识

        Returns:
            {
                'feature_importance': Dict[str, float],
                'domain_shift': np.ndarray,
                'confidence': float
            }
        """
        # 加载源城市数据
        source_data = self._load_source_data()

        if len(source_data) == 0:
            return self._default_transfer()

        # 训练特征重要性模型
        feature_importance = self._learn_feature_importance(source_data)

        # 学习域偏移
        domain_shift = self._learn_domain_shift(source_data)

        # 计算迁移置信度
        confidence = self._compute_transfer_confidence(source_data)

        return {
            'feature_importance': feature_importance,
            'domain_shift': domain_shift,
            'confidence': confidence
        }

    def _load_source_data(self) -> pd.DataFrame:
        """加载所有源城市的数据"""
        all_data = []

        for city in self.source_cities:
            csv_file = self.data_dir / f"ML_Dataset_Final_{city}.csv"

            if not csv_file.exists():
                continue

            df = pd.read_csv(csv_file)
            df['city'] = city
            all_data.append(df)

        if len(all_data) == 0:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def _learn_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        学习光学特征的重要性

        使用Random Forest分析哪些光学特征对节能贡献最大
        """
        # 特征列
        feature_cols = [
            'T_lum_cold', 'T_lum_hot',
            'T_sol_cold', 'T_sol_hot',
            'Rf_sol_cold', 'Rb_sol_cold',
            'Rf_sol_hot', 'Rb_sol_hot'
        ]

        X = data[feature_cols].values
        y = data['TotalEnergy'].values

        # 训练Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # 提取特征重要性
        importance_dict = {}
        for i, col in enumerate(feature_cols):
            importance_dict[col] = float(rf.feature_importances_[i])

        self.feature_importance_model = rf

        return importance_dict

    def _learn_domain_shift(self, source_data: pd.DataFrame) -> np.ndarray:
        """
        学习源城市和目标城市之间的域偏移

        计算光学特征分布的差异
        """
        feature_cols = [
            'T_lum_cold', 'T_lum_hot',
            'T_sol_cold', 'T_sol_hot',
            'Rf_sol_cold', 'Rb_sol_cold',
            'Rf_sol_hot', 'Rb_sol_hot'
        ]

        # 源城市特征均值
        source_mean = source_data[feature_cols].mean().values

        # 尝试加载目标城市数据
        target_file = self.data_dir / f"ML_Dataset_Final_{self.target_city}.csv"

        if target_file.exists():
            target_data = pd.read_csv(target_file)
            target_mean = target_data[feature_cols].mean().values
        else:
            # 如果没有目标城市数据，假设零偏移
            target_mean = source_mean

        # 域偏移向量
        domain_shift = target_mean - source_mean

        return domain_shift

    def _compute_transfer_confidence(self, source_data: pd.DataFrame) -> float:
        """
        计算迁移置信度

        基于源数据量和多样性
        """
        n_samples = len(source_data)
        n_cities = source_data['city'].nunique()

        # 基于样本数的置信度（更多样本 = 更高置信）
        sample_confidence = min(1.0, n_samples / 10000)

        # 基于城市数的置信度（更多城市 = 更好泛化）
        city_confidence = min(1.0, n_cities / 3)

        # 综合置信度
        confidence = 0.6 * sample_confidence + 0.4 * city_confidence

        return confidence

    def _default_transfer(self) -> Dict:
        """无源数据时的默认迁移知识"""
        return {
            'feature_importance': {
                'T_lum_cold': 0.125,
                'T_lum_hot': 0.125,
                'T_sol_cold': 0.125,
                'T_sol_hot': 0.125,
                'Rf_sol_cold': 0.125,
                'Rb_sol_cold': 0.125,
                'Rf_sol_hot': 0.125,
                'Rb_sol_hot': 0.125,
            },
            'domain_shift': np.zeros(8),
            'confidence': 0.0
        }

    def adapt_target_spectrum(self,
                             source_spectrum: Dict[str, float],
                             transfer_knowledge: Dict) -> Dict[str, float]:
        """
        使用迁移知识调整目标光谱

        Args:
            source_spectrum: 从源城市学到的光谱目标
            transfer_knowledge: learn_transfer()的输出

        Returns:
            adapted_spectrum: 适配到目标城市的光谱
        """
        feature_names = [
            'T_lum_cold', 'T_lum_hot',
            'T_sol_cold', 'T_sol_hot',
            'Rf_sol_cold', 'Rb_sol_cold',
            'Rf_sol_hot', 'Rb_sol_hot'
        ]

        # 提取源光谱向量
        source_vector = np.array([source_spectrum.get(name, 0.5)
                                 for name in feature_names])

        # 应用域偏移
        domain_shift = transfer_knowledge['domain_shift']
        confidence = transfer_knowledge['confidence']

        # 调整后的光谱（加权应用偏移）
        adapted_vector = source_vector + confidence * domain_shift

        # 裁剪到有效范围
        adapted_vector = np.clip(adapted_vector, 0.0, 1.0)

        # 转换回字典
        adapted_spectrum = {name: float(val)
                           for name, val in zip(feature_names, adapted_vector)}

        return adapted_spectrum
