"""
Temporal Fusion Transformer 模型组件
从 Temporal_Fusion_Transform/tft_model.py 移植并适配 PINN 天气预测场景

包含:
- TimeDistributed: 时间分布包装器
- GLU: 门控线性单元
- GatedResidualNetwork: 门控残差网络 (GRN)
- VariableSelectionNetwork: 变量选择网络
- PositionalEncoder: 正弦/余弦位置编码
- TFT_Encoder: 适配 PINN 的 TFT 编码器
"""

import math
import torch
from torch import nn


# ======================== 基础组件 ========================

class TimeDistributed(nn.Module):
    """
    将模块按时间步独立应用：先把 [B, T, F] 展平为 [B*T, F]，算完再还原。
    """
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # 合并 batch 和 time 维度
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        # 还原时间维度
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class GLU(nn.Module):
    """门控线性单元: sigmoid(W1·x) * (W2·x)"""
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    """
    TFT 核心模块：非线性变换 + 可选上下文 + 门控残差 + 归一化
    """
    def __init__(self, input_size, hidden_state_size, output_size, dropout,
                 hidden_context_size=None, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size

        # 残差投影（输入输出维度不一致时）
        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(
                nn.Linear(self.input_size, self.output_size), batch_first=batch_first
            )

        # 主变换层
        self.fc1 = TimeDistributed(
            nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first
        )
        self.elu1 = nn.ELU()

        # 可选上下文投影
        if self.hidden_context_size is not None:
            self.context = TimeDistributed(
                nn.Linear(self.hidden_context_size, self.hidden_state_size),
                batch_first=batch_first
            )

        self.fc2 = TimeDistributed(
            nn.Linear(self.hidden_state_size, self.output_size), batch_first=batch_first
        )
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(dropout)
        self.bn = TimeDistributed(
            nn.BatchNorm1d(self.output_size), batch_first=batch_first
        )
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):
        # 残差分支
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        # 主变换
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)
        return x


class VariableSelectionNetwork(nn.Module):
    """
    变量选择网络：学习每个变量的重要性权重，并按权重融合变量表示
    """
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.context = context

        # 对拼接后的变量做 GRN，输出变量选择 logits
        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size, self.hidden_size,
                self.num_inputs, dropout, self.context
            )
        else:
            self.flattened_grn = GatedResidualNetwork(
                self.num_inputs * self.input_size, self.hidden_size,
                self.num_inputs, dropout
            )

        # 每个变量各自一个 GRN
        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size,
                                    self.hidden_size, dropout)
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            var_outputs.append(
                self.single_variable_grns[i](
                    embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]
                )
            )

        var_outputs = torch.stack(var_outputs, axis=-1)
        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights


class PositionalEncoder(nn.Module):
    """固定的正弦/余弦位置编码"""
    def __init__(self, d_model, max_seq_len=500):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len, :]
            x = x + pe
            return x


# ======================== TFT 编码器（适配 PINN） ========================

class TFT_Encoder(nn.Module):
    """
    适配 PINN 场景的 TFT 编码器

    输入:
        x: [batch_size, seq_length, num_features] 气象特征序列
    输出:
        hidden: [batch_size, hidden_dim] 编码后的隐藏表示
        attn_weights: 注意力权重（可用于可解释性）
    """
    def __init__(self, num_real_inputs, hidden_dim, embedding_dim,
                 lstm_layers, attn_heads, dropout, seq_length, encode_length,
                 num_static=0, static_embedding_vocab_sizes=None,
                 num_time_varying_cat=0, time_varying_embedding_vocab_sizes=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_real_inputs = num_real_inputs
        self.num_static = num_static
        self.num_time_varying_cat = num_time_varying_cat
        self.seq_length = seq_length
        self.encode_length = encode_length
        self.lstm_layers = lstm_layers

        # 实值变量线性投影层
        self.real_embedding_layers = nn.ModuleList()
        for _ in range(num_real_inputs):
            self.real_embedding_layers.append(
                TimeDistributed(nn.Linear(1, embedding_dim), batch_first=True)
            )

        # 总变量数 = 实值 + 类别
        total_num_inputs = num_real_inputs + num_time_varying_cat

        # 变量选择网络
        self.variable_selection = VariableSelectionNetwork(
            embedding_dim, total_num_inputs, hidden_dim, dropout
        )

        # LSTM 编码器
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )

        # LSTM 后门控 + 归一化
        self.post_lstm_gate = TimeDistributed(GLU(hidden_dim))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(hidden_dim))

        # 位置编码
        self.position_encoding = PositionalEncoder(hidden_dim, seq_length)

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, attn_heads, batch_first=True
        )
        self.post_attn_gate = TimeDistributed(GLU(hidden_dim))
        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(hidden_dim))

        # 输出前馈
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(hidden_dim))
        self.pre_output_gate = TimeDistributed(GLU(hidden_dim))

        # 最终投影：从序列表示到单一向量
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, num_real_inputs] 实值气象特征
        Returns:
            hidden: [batch_size, hidden_dim] 编码向量
            attn_weights: 注意力权重
        """
        batch_size = x.size(0)

        # 1) 每个实值变量独立 embedding
        real_vectors = []
        for i in range(self.num_real_inputs):
            emb = self.real_embedding_layers[i](
                x[:, :, i].unsqueeze(-1)  # [B, T, 1]
            )
            real_vectors.append(emb)

        # 拼接所有变量 embedding: [B, T, num_inputs * embed_dim]
        embeddings = torch.cat(real_vectors, dim=2)

        # 2) 变量选择
        selected, sparse_weights = self.variable_selection(embeddings)
        # selected: [B, T, hidden_dim]

        # 3) 位置编码
        selected = self.position_encoding(selected)

        # 4) LSTM 编码
        lstm_input = selected
        lstm_output, (h_n, c_n) = self.lstm_encoder(lstm_input)

        # LSTM 门控残差
        lstm_output = self.post_lstm_gate(lstm_output + lstm_input)
        lstm_output = self.post_lstm_norm(lstm_output)

        # 5) 自注意力
        attn_output, attn_weights = self.multihead_attn(
            lstm_output, lstm_output, lstm_output
        )

        # 注意力后残差 + 归一化
        attn_output = self.post_attn_gate(attn_output) + lstm_output
        attn_output = self.post_attn_norm(attn_output)

        # 6) 前馈
        output = self.pos_wise_ff(attn_output)
        output = self.pre_output_gate(output) + lstm_output
        output = self.pre_output_norm(output)

        # 7) 取最后时间步作为编码表示
        last_hidden = output[:, -1, :]  # [B, hidden_dim]
        hidden = self.output_projection(last_hidden)

        return hidden, attn_weights
