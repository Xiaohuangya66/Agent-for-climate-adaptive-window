import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    标准的 Transformer 位置编码，用于 Decoder 端生成的文本/序列
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        return x + self.pe[:, :x.size(1), :]

class TransOpto(nn.Module):
    def __init__(
            self,
            spectrum_dim: int,  # 输入：每个光谱向量的原始维度 (例如: 128, 1024等)
            vocab_size: int,  # 输出：材料序列表的大小 (词表大小)
            d_model: int = 512,  # Transformer 内部维度
            nhead: int = 8,  # 多头注意力的头数
            num_encoder_layers: int = 4,  # Encoder 层数 (通常不需要太深，因为输入只有6)
            num_decoder_layers: int = 6,  # Decoder 层数 (负责复杂的序列生成，建议深一点)
            dim_feedforward: int = 1024,
            dropout: float = 0.1
    ):
        super().__init__()
        # -----------------------------------------------------------
        # 1. Encoder Input Parsing
        # -----------------------------------------------------------
        # 将原始光谱特征投影到 Transformer 的维度
        self.spectrum_projector = nn.Linear(spectrum_dim, d_model)
        self.view_embeddings = nn.Parameter(torch.randn(1, 2, d_model))  # 提供2个角度

        # -----------------------------------------------------------
        # 2. Decoder Input Parsing (处理 "目标序列")
        # -----------------------------------------------------------
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # -----------------------------------------------------------
        # 3. Core Transformer (Encoder-Decoder)
        # -----------------------------------------------------------
        # batch_first=True 让输入输出 shape 变为 [Batch, Seq, Feature] 符合直觉
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # -----------------------------------------------------------
        # 4. Output Layer
        # -----------------------------------------------------------
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, spectrum_vectors, target_sequence, tgt_padding_mask=None):
        """
        参数:
            spectrum_vectors: [Batch, 2, spectrum_dim]
            target_sequence:  [Batch, Seq_Len] 目标序列的 Token IDs (训练时需包含 BOS, 去掉 EOS)。
            tgt_padding_mask: [Batch, Seq_Len] (可选)
                              用于 Mask 掉 Pad Token，避免模型学到填充符。
                              True/1 表示要被 mask (忽略) 的位置。
        """

        # --- Encoder Preparation ---
        # 1. 投影特征
        # src shape: [Batch, 6, d_model]
        src = self.spectrum_projector(spectrum_vectors)
        src = src + self.view_embeddings

        # --- Decoder Preparation ---
        # 3. 目标序列 Embedding + Positional Encoding
        # tgt shape: [Batch, Seq_Len, d_model]
        tgt = self.tgt_embedding(target_sequence)
        tgt = self.pos_encoder(tgt)

        # 4. 生成 Causal Mask (因果掩码/上三角掩码)
        # 保证生成第 t 个 token 时，只能看到 0 到 t-1 的 token，不能看到未来
        seq_len = target_sequence.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len)
        tgt_mask = tgt_mask.to(target_sequence.device)

        # --- Transformer Pass ---
        # 这里的交互过程：
        # Encoder: 6个View互相Attention，互补信息，输出 Memory
        # Decoder: 当前序列 Query 查阅 Encoder Memory，获取生成所需的化学特征
        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,  # 只能看过去
            tgt_key_padding_mask=tgt_padding_mask  # 忽略 Pad Token
            # memory_mask=None (Encoder输出不需要Mask，因为6个必须全看)
        )

        # --- Prediction ---
        # logits: [Batch, Seq_Len, Vocab_Size]
        logits = self.output_head(output)

        return logits
