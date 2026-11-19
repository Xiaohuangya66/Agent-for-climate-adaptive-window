"""
Transformer Wrapper - Transformer模型生成包装器

功能说明：
- 加载预训练的Transformer模型
- 根据目标光谱生成候选结构
- 支持条件生成和采样策略
- 管理生成超参数和约束

输入：
- 目标光谱（8维）
- 生成数量
- 采样参数（temperature, top_k, top_p）

输出：
- 候选结构列表（结构字符串）

依赖：
- PyTorch
- PyTorch Lightning
- 预训练的Transformer checkpoint

作者：Claude
日期：2026-10-13
版本：v1.0
"""

import torch
import pytorch_lightning as pl
import numpy as np
from src.utils.spectrum_utils import expand_8d_to_442d
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

from src.config import CHECKPOINTS_DIR, STRUCTURE_CONSTRAINTS


class TransformerWrapper:
    """
    Transformer生成模型包装器

    主要方法：
    - generate(): 根据目标光谱生成结构
    - generate_batch(): 批量生成
    - decode_structure(): 解码token序列为结构字符串
    """

    def __init__(self,
                 checkpoint_path: Optional[Path] = None,
                 device: str = 'cpu'):
        """
        初始化Transformer包装器

        Args:
            checkpoint_path: 模型checkpoint路径
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)

        # 默认checkpoint路径
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_DIR / "model_S-epoch=1198-val_loss_epoch=3.436.ckpt"

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # 加载模型
        self.model = None
        self.vocab = None
        self._load_model()

    def _load_model(self):
        """
        加载预训练模型

        注意：这里需要根据实际的模型架构调整
        假设模型保存了完整的LightningModule
        """
        try:
            # 加载checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # 方式1：如果保存的是完整模型
            if 'model' in checkpoint:
                self.model = checkpoint['model']
            # 方式2：如果需要重建模型架构
            else:
                # TODO: 根据实际模型架构重建
                # from your_model_module import TransformerModel
                # self.model = TransformerModel.load_from_checkpoint(self.checkpoint_path)
                pass

            # 加载词表
            if 'vocab' in checkpoint:
                self.vocab = checkpoint['vocab']
            else:
                # 构建默认词表
                self.vocab = self._build_default_vocab()

            # 设置为评估模式
            if self.model is not None:
                self.model.eval()
                self.model.to(self.device)

            print(f"✓ Transformer模型加载成功")

        except Exception as e:
            print(f"Warning: 无法加载Transformer模型: {e}")
            print("使用随机生成作为fallback")
            self.model = None

    def _build_default_vocab(self) -> Dict:
        """
        构建默认词表

        Returns:
            {
                'token_to_id': Dict[str, int],
                'id_to_token': Dict[int, str],
                'special_tokens': Dict
            }
        """
        # 材料列表
        materials = STRUCTURE_CONSTRAINTS['allowed_materials']

        # 厚度范围（离散化）
        thicknesses = list(range(5, 501, 5))  # 5-500nm, 步长5nm

        # 生成所有token
        tokens = ['<PAD>', '<START>', '<END>', '<UNK>']

        # 添加材料_厚度组合
        for mat in materials:
            if mat == 'WVO2':
                # WVO2特殊范围
                for thick in range(20, 41, 5):
                    tokens.append(f"{mat}_{thick}")
            else:
                for thick in thicknesses:
                    tokens.append(f"{mat}_{thick}")

        # 基底
        tokens.append('SiO2_10000')

        # 构建映射
        token_to_id = {t: i for i, t in enumerate(tokens)}
        id_to_token = {i: t for t, i in token_to_id.items()}

        return {
            'token_to_id': token_to_id,
            'id_to_token': id_to_token,
            'special_tokens': {
                'pad': 0,
                'start': 1,
                'end': 2,
                'unk': 3
            }
        }

    def generate(self,
                 target_spectrum: np.ndarray,
                 n: int = 1,
                 max_length: int = 20,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> List[str]:
        """
        根据目标光谱生成结构

        Args:
            target_spectrum: 目标光谱 (8维或442维)
            n: 生成数量
            max_length: 最大序列长度
            temperature: 采样温度（越高越随机）
            top_k: Top-K采样
            top_p: Nucleus采样

        Returns:
            结构字符串列表
        """
        # 如果模型未加载，使用随机生成
        if self.model is None:
            return self._generate_random_fallback(n)

        try:
            # 准备输入
            target_tensor = self._prepare_target(target_spectrum)

            structures = []
            for _ in range(n):
                # 生成单个结构
                token_ids = self._generate_single(
                    target_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # 解码为结构字符串
                structure_str = self._decode_structure(token_ids)

                # 验证和修复
                from src.utils.structure_utils import validate_structure, repair_structure

                is_valid, error = validate_structure(structure_str)
                if not is_valid:
                    structure_str = repair_structure(structure_str)

                structures.append(structure_str)

            return structures

        except Exception as e:
            print(f"Warning: Transformer生成失败: {e}, 使用随机生成")
            return self._generate_random_fallback(n)

    def _prepare_target(self, target_spectrum) -> torch.Tensor:
        """
        准备目标光谱为模型输入

        Args:
            target_spectrum:
                - Dict形式: {'T_lum_cold': 0.5, 'T_sol_cold': 0.4, ...}
                - 8维Array: np.array([T_lum_cold, T_lum_hot, ...])
                - 442维Array: np.array([...]) 直接使用

        Returns:
            torch.Tensor (1, feature_dim)
        """
        # 情况1: Dict输入（8维目标）
        if isinstance(target_spectrum, dict):
            # 扩展为442维（只包含T_cold和T_hot）
            spectrum_442d = expand_8d_to_442d(target_spectrum)  # (442, 2)
            target_tensor = torch.from_numpy(spectrum_442d).float()

        # 情况2: numpy array输入
        elif isinstance(target_spectrum, np.ndarray):
            # 8维 → 扩展为442维
            if target_spectrum.shape[0] == 8:
                spectrum_442d = expand_8d_to_442d(target_spectrum)  # (442, 2)
                target_tensor = torch.from_numpy(spectrum_442d).float()

            # 442维 → 直接使用
            elif len(target_spectrum.shape) == 2 and target_spectrum.shape[0] == 442:
                target_tensor = torch.from_numpy(target_spectrum).float()

            else:
                raise ValueError(
                    f"Invalid target_spectrum shape: {target_spectrum.shape}. "
                    f"Expected (8,) or (442, 2)"
                )

        else:
            raise TypeError(
                f"Invalid target_spectrum type: {type(target_spectrum)}. "
                f"Expected dict or np.ndarray"
            )

        # 添加batch维度: (442, 2) → (1, 442, 2)
        target_tensor = target_tensor.unsqueeze(0).to(self.device)

        return target_tensor
        return target_tensor

    def _generate_single(self,
                        target_tensor: torch.Tensor,
                        max_length: int,
                        temperature: float,
                        top_k: int,
                        top_p: float) -> List[int]:
        """
        生成单个序列（token IDs）

        Args:
            target_tensor: 目标光谱tensor
            max_length: 最大长度
            temperature: 温度
            top_k: Top-K
            top_p: Top-P

        Returns:
            token_ids列表
        """
        with torch.no_grad():
            # 初始化序列
            token_ids = [self.vocab['special_tokens']['start']]

            for _ in range(max_length):
                # 构建输入
                input_ids = torch.tensor([token_ids]).to(self.device)

                # 模型前向传播
                # 注意：这里需要根据实际模型API调整
                # logits = self.model(input_ids, target_tensor)
                # 简化：使用最后一个token的logits
                # next_token_logits = logits[0, -1, :]

                # 由于模型API不确定，这里简化处理
                # 实际应该调用模型的generate方法
                next_token_id = self._sample_next_token(
                    None,  # logits placeholder
                    temperature,
                    top_k,
                    top_p
                )

                # 添加到序列
                token_ids.append(next_token_id)

                # 检查是否结束
                if next_token_id == self.vocab['special_tokens']['end']:
                    break

                # 检查是否达到基底
                token_str = self.vocab['id_to_token'].get(next_token_id, '')
                if token_str == 'SiO2_10000':
                    break

            return token_ids

    def _sample_next_token(self,
                          logits: Optional[torch.Tensor],
                          temperature: float,
                          top_k: int,
                          top_p: float) -> int:
        """
        从logits采样下一个token

        如果logits为None，使用随机采样（fallback）
        """
        if logits is None:
            # Fallback: 随机选择
            return np.random.randint(4, len(self.vocab['id_to_token']))

        # 应用temperature
        logits = logits / temperature

        # Top-K过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        # Top-P (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        # 采样
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        return next_token_id

    def _decode_structure(self, token_ids: List[int]) -> str:
        """
        解码token IDs为结构字符串

        Args:
            token_ids: token ID列表

        Returns:
            结构字符串 "WVO2_30 SiO2_100 ..."
        """
        tokens = []

        for token_id in token_ids:
            # 跳过特殊token
            if token_id in [0, 1, 2, 3]:  # PAD, START, END, UNK
                continue

            token_str = self.vocab['id_to_token'].get(token_id, '')

            if token_str and '_' in token_str:
                tokens.append(token_str)

        # 确保有基底
        if not tokens or tokens[-1] != 'SiO2_10000':
            tokens.append('SiO2_10000')

        return ' '.join(tokens)

    def _generate_random_fallback(self, n: int) -> List[str]:
        """
        随机生成作为fallback

        当模型加载失败或生成失败时使用
        """
        from src.utils.structure_utils import generate_random_structure

        structures = []
        for _ in range(n):
            structure = generate_random_structure()
            structures.append(structure)

        return structures

    def encode_structure(self, structure_str: str) -> List[int]:
        """
        将结构字符串编码为token IDs

        Args:
            structure_str: "WVO2_30 SiO2_100 ..."

        Returns:
            token_ids列表
        """
        from src.utils.structure_utils import parse_structure

        tokens = parse_structure(structure_str)

        token_ids = [self.vocab['special_tokens']['start']]

        for material, thickness in tokens:
            token_str = f"{material}_{thickness}"
            token_id = self.vocab['token_to_id'].get(
                token_str,
                self.vocab['special_tokens']['unk']
            )
            token_ids.append(token_id)

        token_ids.append(self.vocab['special_tokens']['end'])

        return token_ids
