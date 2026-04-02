"""
PINN + TFT 天气预测 - 主入口 

用法:
    # Demo 模式
    python main.py --demo --epochs 3

    # 真实数据 (1h 预测)
    python main.py --data_path weather.csv --epochs 200

    # 3天预测
    python main.py --data_path weather.csv --forecast_hours 72 --epochs 200

    # 5天预测
    python main.py --data_path weather.csv --forecast_hours 120 --epochs 200

    # 自定义目标变量
    python main.py --data_path weather.csv --target_cols t2m,z_500,t_850
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

from configs.config import get_args
from dataloader.weather_dataloader import (
    generate_demo_data, create_dataloader,
    get_all_feature_cols, classify_columns, DEFAULT_TARGET_COLS
)
from Model.pinn_weather import WeatherPINN
from utils.util import eval_metrix


def main():
    args = get_args()

    # 解析目标列
    target_cols = [c.strip() for c in args.target_cols.split(',')]
    # 解析预测步长列表
    forecast_hours_list = [int(h.strip()) for h in args.forecast_hours.split(',')]
    
    print("=" * 65)
    print("  PINN + TFT 天气预测")
    print("=" * 65)
    print(f"  设备: {args.device}")
    print(f"  模式: {'Demo' if args.demo else '真实数据'}")
    print(f"  预测目标: {target_cols}")
    print(f"  预测步长: {[f'{h}h' for h in forecast_hours_list]}")
    print(f"  序列长度: {args.seq_length} (编码: {args.encode_length})")
    print(f"  Epochs: {args.epochs}")
    print(f"  PINN 权重: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print("=" * 65)

    # ======================== 数据加载 ========================
    if args.demo:
        print("\n[数据] 生成模拟天气数据...")
        df, feature_cols, static_cols, target_cols = generate_demo_data(
            num_days=180)
    else:
        if args.data_path is None or not os.path.exists(args.data_path):
            print("错误: 请提供有效的数据文件路径 --data_path")
            sys.exit(1)

        print(f"\n[数据] 加载: {args.data_path}")
        df = pd.read_csv(args.data_path)
        
        # ======== 缩减数据集 (加速训练) ========
        if args.max_samples is not None and len(df) > args.max_samples:
            print(f"  [提示] 截取数据集末尾 (最近)的 {args.max_samples} 条记录 (原共 {len(df)} 条)")
            df = df.iloc[-args.max_samples:].reset_index(drop=True)
        # ======================================

        # 验证目标列存在
        for tc in target_cols:
            if tc not in df.columns:
                print(f"错误: 目标列 '{tc}' 不在数据中")
                print(f"  可用列: {list(df.columns)[:20]}...")
                sys.exit(1)

        # 自动分类列名
        feature_cols, static_cols = get_all_feature_cols(
            df.columns, target_cols)
        surface, pressure, static = classify_columns(df.columns)
        print(f"  地表变量 ({len(surface)}): {surface}")
        print(f"  气压层变量 ({len(pressure)}): "
              f"{pressure[:5]}... 共{len(pressure)}个")
        print(f"  静态常量 ({len(static)}): {static}")

    num_targets = len(target_cols)
    num_horizons = len(forecast_hours_list)
    num_targets_total = num_targets * num_horizons

    # 构建 DataLoader
    dataloaders, scaler_info, num_features = create_dataloader(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        static_cols=static_cols,
        forecast_hours=args.forecast_hours,
    )

    # ======================== 模型初始化 ========================
    print(f"\n[模型] WeatherPINN (特征: {num_features}, 总输出目标: {num_targets_total})")
    model = WeatherPINN(
        args, num_features=num_features,
        num_targets=num_targets_total, target_names=target_cols
    )

    u_params = sum(p.numel() for p in model.solution_u.parameters()
                   if p.requires_grad)
    f_params = sum(p.numel() for p in model.dynamical_F.parameters()
                   if p.requires_grad)
    print(f"  Solution_u: {u_params:,} 参数")
    print(f"  DynamicalF: {f_params:,} 参数")
    print(f"  总计: {u_params + f_params:,} 参数")

    # ======================== 训练 ========================
    print(f"\n[训练] 开始训练 {args.epochs} epochs...")
    model.Train(
        trainloader=dataloaders['train'],
        validloader=dataloaders['valid'],
        testloader=dataloaders['test'],
        scaler_info=scaler_info,
    )

    # ======================== 最终测试 ========================
    print("\n[测试] 最终评估...")
    if model.best_model is not None:
        model.solution_u.load_state_dict(model.best_model['solution_u'])
        model.dynamical_F.load_state_dict(model.best_model['dynamical_F'])

    true_label, pred_label = model.Test(dataloaders['test'])

    # 物理单位指标
    w_metrics = model.evaluate_weather(true_label, pred_label, scaler_info)

    print("\n" + "=" * 65)
    print(f"  最终测试结果")
    print("-" * 65)
    print(f"  {'变量':<15} {'RMSE':>12} {'MAE':>12} {'单位':>10}")
    print("-" * 65)
    for name, m in w_metrics.items():
        print(f"  {name:<15} {m['RMSE']:>12.4f} {m['MAE']:>12.4f} "
              f"{m['unit']:>10}")
    print("=" * 65)

    # 保存
    if args.save_folder is not None:
        np.save(os.path.join(args.save_folder, 'final_true.npy'), true_label)
        np.save(os.path.join(args.save_folder, 'final_pred.npy'), pred_label)
        print(f"\n结果已保存至: {args.save_folder}")

    print("\nDone!")


if __name__ == '__main__':
    main()
