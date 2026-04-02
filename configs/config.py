"""
配置模块
集中管理模型、训练、PINN 和数据相关的超参数
"""

import argparse
import os


def get_args():
    """解析命令行参数并返回配置"""
    parser = argparse.ArgumentParser(description='PINN + TFT 天气日均气温预测')

    # ======================== 数据相关 ========================
    parser.add_argument('--data_path', type=str, default=None,
                        help='天气数据 CSV 文件路径')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='截取最近的 N 条样本用于训练（为空则使用全量），用于加速训练')
    parser.add_argument('--target_cols', type=str, default='t2m,z_500,t_850',
                        help='预测目标列名，逗号分隔')
    parser.add_argument('--seq_length', type=int, default=72,
                        help='总序列长度（小时数），包含编码+解码')
    parser.add_argument('--encode_length', type=int, default=48,
                        help='编码器历史窗口长度（小时数）')
    parser.add_argument('--forecast_hours', type=str, default='72,120',
                        help='向前预测步数（支持多步），逗号分隔。如 72,120')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')

    # ======================== TFT 模型相关 ========================
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='TFT 隐藏层维度')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='变量 embedding 维度')
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help='LSTM 层数')
    parser.add_argument('--attn_heads', type=int, default=4,
                        help='多头注意力头数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 概率')

    # ======================== 训练相关 ========================
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='基础学习率')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Warmup 轮数')
    parser.add_argument('--warmup_lr', type=float, default=0.0001,
                        help='Warmup 学习率')
    parser.add_argument('--final_lr', type=float, default=0.00005,
                        help='最终学习率')
    parser.add_argument('--lr_F', type=float, default=0.001,
                        help='F 网络学习率')
    parser.add_argument('--early_stop', type=int, default=15,
                        help='早停轮数')

    # ======================== PINN 相关 ========================
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='PDE 损失权重: loss = L_data + alpha*L_PDE + beta*L_physics')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='物理约束损失权重')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='周期性约束损失权重（包含在 beta 内部分配）')
    parser.add_argument('--T_min', type=float, default=200.0,
                        help='物理约束：最低温度（K），约 -73°C')
    parser.add_argument('--T_max', type=float, default=330.0,
                        help='物理约束：最高温度（K），约 57°C')
    parser.add_argument('--delta_T_max', type=float, default=15.0,
                        help='物理约束：相邻小时最大温差（K）')
    parser.add_argument('--F_layers_num', type=int, default=3,
                        help='F 网络层数')
    parser.add_argument('--F_hidden_dim', type=int, default=64,
                        help='F 网络隐藏维度')

    # ======================== 其他 ========================
    parser.add_argument('--save_folder', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--log_dir', type=str, default='train_log.txt',
                        help='日志文件名')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)，默认自动选择')
    parser.add_argument('--demo', action='store_true',
                        help='使用模拟数据运行 demo 模式')

    args = parser.parse_args()

    # 自动选择设备
    if args.device is None:
        import torch
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建保存目录
    if args.save_folder and not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder, exist_ok=True)

    return args
