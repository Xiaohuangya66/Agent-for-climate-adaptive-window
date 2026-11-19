"""
Transformer Wrapper v2 - 集成原始OptoGPT模型

正确加载并使用训练好的Transformer模型
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import sys

# 添加原始代码路径
sys.path.insert(0, '/public/home/nju_zhanghuajie/topic3')

from model import TransOpto
from inference_1 import top_k_top_p_filtering


class TransformerWrapperV2:
    """
    集成原始OptoGPT的Transformer包装器

    架构：
    - 输入：442维光谱 [T_cold(380-2500nm), T_hot(380-2500nm)]
    - 输出：结构token序列
    """

    def __init__(self,
                 checkpoint_path: str = '/public/home/nju_zhanghuajie/topic3/checkpoints/last.ckpt',
                 vocab_path: str = '/public/home/nju_zhanghuajie/topic3/vocab.txt',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            checkpoint_path: 模型checkpoint路径
            vocab_path: 词表文件路径
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)

        # 加载词表
        self.token2id, self.id2token = self._load_vocab()
        self.vocab_size = len(self.token2id)

        # 特殊token
        self.bos_id = self.token2id['BOS']
        self.eos_id = self.token2id['EOS']
        self.pad_id = self.token2id['PAD']
        self.unk_id = self.token2id['UNK']

        # 加载模型
        self.model = self._load_model()
        self.model.eval()

        print(f"✓ TransformerV2加载成功")
        print(f"  - Vocab size: {self.vocab_size}")
        print(f"  - Device: {self.device}")
        print(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M")

    def _load_vocab(self) -> tuple:
        """加载词表"""
        token2id = {}
        id2token = {}

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                if token:
                    token2id[token] = idx
                    id2token[idx] = token

        return token2id, id2token

    def _load_model(self) -> nn.Module:
        """加载模型"""
        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # 从checkpoint中获取超参数
        # 如果没有存储，使用默认值
        spectrum_dim = 221  # 实际是221维
        d_model = 256
        nhead = 4
        num_encoder_layers = 3
        num_decoder_layers = 6
        dim_feedforward = 512
        dropout = 0.1

        # 尝试从checkpoint中读取（如果有的话）
        if 'hyper_parameters' in checkpoint:
            hp = checkpoint['hyper_parameters']
            d_model = hp.get('d_model', d_model)
            nhead = hp.get('nhead', nhead)
            num_encoder_layers = hp.get('num_encoder_layers', num_encoder_layers)
            num_decoder_layers = hp.get('num_decoder_layers', num_decoder_layers)
            dim_feedforward = hp.get('dim_feedforward', dim_feedforward)
            dropout = hp.get('dropout', dropout)

        # 创建模型
        model = TransOpto(
            spectrum_dim=spectrum_dim,
            vocab_size=self.vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 加载权重
        if 'state_dict' in checkpoint:
            # PyTorch Lightning保存的格式，需要去掉'model.'前缀
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.'):
                    new_key = key[6:]  # 去掉'model.'
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value

            model.load_state_dict(state_dict)
        else:
            raise ValueError("Checkpoint中没有state_dict")

        model.to(self.device)
        return model

    def generate(self,
                 target_spectrum: Union[Dict[str, float], np.ndarray],
                 n: int = 1,
                 max_length: int = 20,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> List[str]:
        """
        根据目标光谱生成结构

        Args:
            target_spectrum: 目标光谱
                - Dict: 8维聚合特征
                - np.ndarray: (221, 2) 详细光谱
            n: 生成数量
            max_length: 最大序列长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Nucleus采样

        Returns:
            结构字符串列表
        """
        # 转换为442维光谱
        if isinstance(target_spectrum, dict):
            from src.utils.spectrum_utils import expand_8d_to_221d
            spectrum_221d = expand_8d_to_221d(target_spectrum)  # (221, 2)
        elif isinstance(target_spectrum, np.ndarray):
            if target_spectrum.shape[0] == 8:
                from src.utils.spectrum_utils import expand_8d_to_221d
                spectrum_221d = expand_8d_to_221d(target_spectrum)
            else:
                spectrum_221d = target_spectrum  # 已经是(221, 2)
        else:
            raise TypeError(f"Invalid target_spectrum type: {type(target_spectrum)}")

        
        # 转换为tensor: (1, 2, 221)
        spectrum_tensor = torch.from_numpy(spectrum_221d).float()
        spectrum_tensor = spectrum_tensor.permute(1, 0)  # (2, 221)
        spectrum_tensor = spectrum_tensor.unsqueeze(0)   # (1, 2, 221)
        spectrum_tensor = spectrum_tensor.to(self.device)

        # 复制为batch
        spectrum_batch = spectrum_tensor.repeat(n, 1, 1)  # (n, 2, 442)

        structures = []

        with torch.no_grad():
            for i in range(n):
                spectrum_single = spectrum_batch[i:i+1]  # (1, 2, 442)

                # 生成序列
                token_ids = self._generate_single(
                    spectrum_single,
                    max_length,
                    temperature,
                    top_k,
                    top_p
                )

                # 转换为字符串
                structure_str = self._ids_to_structure(token_ids)
                structures.append(structure_str)

        return structures

    def _generate_single(self,
                        spectrum: torch.Tensor,
                        max_length: int,
                        temperature: float,
                        top_k: int,
                        top_p: float) -> List[int]:
        """
        生成单个序列

        Args:
            spectrum: (1, 2, 442)

        Returns:
            token_ids列表
        """
        # 初始化：只有BOS
        generated = [self.bos_id]

        for _ in range(max_length):
            # 当前序列
            tgt_tokens = torch.tensor([generated], dtype=torch.long).to(self.device)  # (1, seq_len)

            # 创建padding mask
            tgt_padding_mask = (tgt_tokens == self.pad_id)  # (1, seq_len)

            # 模型前向传播
            logits = self.model(spectrum, tgt_tokens, tgt_padding_mask)  # (1, seq_len, vocab_size)

            # 取最后一个位置的logits
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            # 采样下一个token
            next_token_id = self._sample_next_token(
                next_token_logits,
                temperature,
                top_k,
                top_p
            )

            # 添加到序列
            generated.append(next_token_id)

            # 如果遇到EOS，停止
            if next_token_id == self.eos_id:
                break

        return generated

    def _sample_next_token(self,
                          logits: torch.Tensor,
                          temperature: float,
                          top_k: int,
                          top_p: float) -> int:
        """
        采样下一个token

        Args:
            logits: (vocab_size,)

        Returns:
            token_id
        """
        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature

        # Top-K Top-P过滤
        filtered_logits = top_k_top_p_filtering(
            logits.unsqueeze(0),  # (1, vocab_size)
            top_k=top_k,
            top_p=top_p
        )[0]  # (vocab_size,)

        # 采样
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        return int(next_token_id)


    def _ids_to_structure(self, token_ids: List[int]) -> str:
        """
        将token IDs转换为结构字符串

        Args:
            token_ids: [BOS, token1, token2, ..., EOS]

        Returns:
            "SiO2_100 TiO2_50 ..."
        """
        tokens = []

        for token_id in token_ids:
            # 跳过特殊token
            if token_id in [self.bos_id, self.eos_id, self.pad_id]:
                continue

            # 转换为字符串
            token_str = self.id2token.get(token_id, 'UNK')

            if token_str != 'UNK':
                tokens.append(token_str)

        # 拼接
        structure = ' '.join(tokens)

        # 验证和修复
        from src.utils.structure_utils import validate_and_fix_structure
        structure = validate_and_fix_structure(structure)

        return structure


# 测试代码
if __name__ == '__main__':
    print("测试TransformerWrapperV2...")

    wrapper = TransformerWrapperV2()

    # 测试生成
    target = {
        'T_lum_cold': 0.5,
        'T_lum_hot': 0.5,
        'T_sol_cold': 0.4,
        'T_sol_hot': 0.3,
        'Rf_sol_cold': 0.3,
        'Rb_sol_cold': 0.3,
        'Rf_sol_hot': 0.3,
        'Rb_sol_hot': 0.3,
    }

    structures = wrapper.generate(target, n=5, temperature=1.0)

    print("\n生成的结构:")
    for i, s in enumerate(structures, 1):
        print(f"{i}. {s}")
