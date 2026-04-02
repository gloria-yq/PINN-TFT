"""
天气数据加载器 (多目标版)
支持真实 CSV 数据和模拟数据

CSV 列名格式:
    时间列:       time (格式: 1979-01-01 00:00:00)
    地表单层变量: z, u10, v10, t2m, tp, tcc, tisr
    多气压层变量: z_*, t_*, u_*, v_*, q_*, r_*, vo_*, pv_*
    静态常量:     orography, lsm, slt, lat2d, lon2d

预测目标: t2m, z_500, t_850 (可配置)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ======================== 列名分类 ========================

SURFACE_VARS = ['z', 'u10', 'v10', 't2m', 'tp', 'tcc', 'tisr']
PRESSURE_PREFIXES = ['z_', 't_', 'u_', 'v_', 'q_', 'r_', 'vo_', 'pv_']
STATIC_VARS = ['orography', 'lsm', 'slt', 'lat2d', 'lon2d']
TIME_COL_CANDIDATES = ['time', 'datetime', 'date', 'timestamp',
                       'Time', 'Datetime', 'DATE', 'TIME']
DEFAULT_TARGET_COLS = ['t2m', 'z_500', 't_850']
DEFAULT_TARGET_COL = 't2m'

TIME_FEATURE_COLS = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                     'month_sin', 'month_cos']


def classify_columns(columns):
    """自动分类列名 → 地表/气压层/静态"""
    surface_cols, pressure_cols, static_cols = [], [], []
    for col in columns:
        if col in STATIC_VARS:
            static_cols.append(col)
        elif col in SURFACE_VARS:
            surface_cols.append(col)
        elif any(col.startswith(p) for p in PRESSURE_PREFIXES):
            pressure_cols.append(col)
    return surface_cols, pressure_cols, static_cols


def detect_time_col(columns):
    """自动检测时间列名"""
    for c in TIME_COL_CANDIDATES:
        if c in columns:
            return c
    return None


def extract_time_features(df, time_col):
    """从时间列提取 sin/cos 周期编码 (6维)"""
    dt = pd.to_datetime(df[time_col])
    hour = dt.dt.hour + dt.dt.minute / 60.0
    day_of_year = dt.dt.dayofyear
    month = dt.dt.month
    return pd.DataFrame({
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_year / 365.25),
        'day_cos': np.cos(2 * np.pi * day_of_year / 365.25),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
    }, index=df.index)


def get_all_feature_cols(columns, target_cols=None):
    """
    自动提取所有可用特征列

    Returns:
        feature_cols: 所有特征列名
        static_cols: 静态列名
    """
    if target_cols is None:
        target_cols = DEFAULT_TARGET_COLS

    surface_cols, pressure_cols, static_cols = classify_columns(columns)
    feature_cols = surface_cols + pressure_cols + static_cols

    # 确保所有目标列都在特征列中
    for tc in target_cols:
        if tc not in feature_cols and tc in columns:
            feature_cols.insert(0, tc)

    return feature_cols, static_cols


# ======================== 模拟数据生成 ========================

def generate_demo_data(num_days=365, seed=42):
    """生成模拟天气数据 (含多目标: t2m, z_500, t_850)"""
    np.random.seed(seed)
    num_hours = num_days * 24
    t = np.arange(num_hours)

    annual = 15 * np.sin(2 * np.pi * t / (365 * 24) - np.pi / 2)
    diurnal = 5 * np.sin(2 * np.pi * t / 24 - np.pi / 3)
    noise = np.random.normal(0, 1.5, num_hours)
    temp_2m = 288.0 + annual + diurnal + noise

    data = {}
    data['time'] = pd.date_range(start='2023-01-01', periods=num_hours, freq='h')

    # 地表变量
    data['z'] = 50000 + 500 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 100, num_hours)
    data['u10'] = 3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 2, num_hours)
    data['v10'] = 2 * np.cos(2 * np.pi * t / 48) + np.random.normal(0, 1.5, num_hours)
    data['t2m'] = temp_2m
    data['tp'] = np.maximum(0, np.random.exponential(0.5, num_hours) * (np.random.random(num_hours) > 0.7))
    data['tcc'] = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * t / 72) + np.random.normal(0, 0.15, num_hours), 0, 1)
    data['tisr'] = np.maximum(0, 800 * np.sin(2 * np.pi * t / 24 - np.pi / 6) + np.random.normal(0, 50, num_hours))

    # 气压层变量
    for level in [500, 850, 1000]:
        offset = (1000 - level) * 0.03
        data[f't_{level}'] = temp_2m - offset * 10 + np.random.normal(0, 2, num_hours)
        data[f'z_{level}'] = (5000 * (1000 / level) + 200 * np.sin(2 * np.pi * t / (365 * 24))
                              + np.random.normal(0, 50, num_hours)) * 9.81  # m²/s²
        data[f'u_{level}'] = data['u10'] * (1 + offset) + np.random.normal(0, 1, num_hours)
        data[f'v_{level}'] = data['v10'] * (1 + offset) + np.random.normal(0, 1, num_hours)
        data[f'q_{level}'] = 0.008 + 0.004 * np.sin(2 * np.pi * t / (365 * 24)) + np.random.normal(0, 0.001, num_hours)
        data[f'r_{level}'] = 60 + 20 * np.sin(2 * np.pi * t / 24 + np.pi) + np.random.normal(0, 5, num_hours)
        data[f'vo_{level}'] = np.random.normal(0, 1e-4, num_hours)
        data[f'pv_{level}'] = np.random.normal(0, 1e-6, num_hours)

    # 静态常量
    data['orography'] = np.full(num_hours, 150.0)
    data['lsm'] = np.full(num_hours, 1.0)
    data['slt'] = np.full(num_hours, 3.0)
    data['lat2d'] = np.full(num_hours, 39.9)
    data['lon2d'] = np.full(num_hours, 116.4)

    df = pd.DataFrame(data)
    feature_cols, static_cols = get_all_feature_cols(df.columns, DEFAULT_TARGET_COLS)
    return df, feature_cols, static_cols, DEFAULT_TARGET_COLS


# ======================== 数据预处理 ========================

def create_sequences(df, feature_cols, target_cols, seq_length=72,
                     static_cols=None, forecast_hours=1):
    """
    多目标 + 多步预测 滑动窗口

    Args:
        target_cols: list, 预测目标列名 (如 ['t2m', 'z_500', 't_850'])
        forecast_hours: int, 向前预测步数 (1=下一小时, 72=3天, 120=5天)

    Returns:
        X1, X2: [N, seq_length, num_features]
        Y1, Y2: [N, num_targets] 多目标标签
        scaler_info: 标准化信息 (含物理单位反标准化参数)
        num_features: 总特征数
    """
    if static_cols is None:
        static_cols = []
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    num_targets = len(target_cols)

    # ---- 1. 时间列 ----
    time_col = detect_time_col(df.columns)
    if time_col is not None:
        print(f"[时间列] 检测到 '{time_col}'，提取 sin/cos 编码 (6维)")
        time_feature_values = extract_time_features(df, time_col).values.astype(np.float32)
    else:
        print("[时间列] 未检测到时间列")
        time_feature_values = None

    # ---- 2. 分离时变/静态特征 ----
    time_varying_cols = [c for c in feature_cols if c not in static_cols]
    
    # 填补空值 (处理前几个时间步可能的 NaN)：线性插值 -> 向前填充 -> 向后填充
    df[time_varying_cols] = df[time_varying_cols].interpolate(method='linear', limit_direction='both').ffill().bfill()
    
    tv_features = df[time_varying_cols].values.astype(np.float32)

    scaler = StandardScaler()
    tv_features = scaler.fit_transform(tv_features)

    if static_cols:
        static_features = df[static_cols].values.astype(np.float32)
        static_scaler = StandardScaler()
        static_features = static_scaler.fit_transform(static_features)
    else:
        static_features = None

    # ---- 3. 多目标标准化信息 ----
    target_means = {}
    target_stds = {}
    target_indices = {}  # 在 time_varying_cols 中的索引

    for tc in target_cols:
        idx = time_varying_cols.index(tc)
        target_indices[tc] = idx
        target_means[tc] = float(scaler.mean_[idx])
        target_stds[tc] = float(scaler.scale_[idx])

    # 目标值标准化
    targets_scaled = np.zeros((len(df), num_targets), dtype=np.float32)
    for i, tc in enumerate(target_cols):
        raw = df[tc].values.astype(np.float32)
        targets_scaled[:, i] = (raw - target_means[tc]) / target_stds[tc]

    # ---- 4. 拼接所有特征 ----
    feature_parts = [tv_features]
    if time_feature_values is not None:
        feature_parts.append(time_feature_values)
    if static_features is not None:
        feature_parts.append(static_features)
    all_features = np.concatenate(feature_parts, axis=1)
    num_features = all_features.shape[1]

    # ---- 5. 滑动窗口 + 多步预测 ----
    X1, X2, Y1, Y2 = [], [], [], []
    
    # 支持多个预测步长
    if isinstance(forecast_hours, str):
        forecast_hours_list = [int(h.strip()) for h in forecast_hours.split(',')]
    elif isinstance(forecast_hours, int):
        forecast_hours_list = [forecast_hours]
    else:
        forecast_hours_list = list(forecast_hours)
        
    max_forecast_hour = max(forecast_hours_list)
    max_idx = len(all_features) - seq_length - max_forecast_hour

    for i in range(max_idx):
        x1 = all_features[i: i + seq_length]
        x2 = all_features[i + 1: i + 1 + seq_length]
        
        y1_list = []
        y2_list = []
        
        valid = True
        for h in forecast_hours_list:
            y1_idx = i + seq_length - 1 + h
            y2_idx = i + seq_length + h
            if y2_idx >= len(targets_scaled):
                valid = False
                break
            y1_list.append(targets_scaled[y1_idx])
            y2_list.append(targets_scaled[y2_idx])
            
        if not valid:
            break

        X1.append(x1)
        X2.append(x2)
        Y1.append(np.concatenate(y1_list))
        Y2.append(np.concatenate(y2_list))

    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    Y1 = np.array(Y1, dtype=np.float32)  # [N, num_targets * len(forecast_hours_list)]
    Y2 = np.array(Y2, dtype=np.float32)

    scaler_info = {
        'target_means': target_means,
        'target_stds': target_stds,
        'target_cols': target_cols,
        'target_indices': target_indices,
        'scaler': scaler,
        'time_varying_cols': time_varying_cols,
        'time_feature_cols': TIME_FEATURE_COLS if time_feature_values is not None else [],
        'static_cols': static_cols,
        'forecast_hours_list': forecast_hours_list,
    }

    return X1, X2, Y1, Y2, scaler_info, num_features


# ======================== DataLoader 构建 ========================

def create_dataloader(df, feature_cols, target_cols, seq_length=72,
                      batch_size=64, test_ratio=0.2, valid_ratio=0.2,
                      static_cols=None, forecast_hours=1):
    """
    构建 train/valid/test DataLoader (多目标版)

    Returns:
        dataloaders: dict
        scaler_info: 标准化信息
        num_features: 总特征数量
    """
    X1, X2, Y1, Y2, scaler_info, num_features = create_sequences(
        df, feature_cols, target_cols, seq_length, static_cols, forecast_hours
    )

    total_samples = len(X1)
    test_split = int(total_samples * (1 - test_ratio))
    train_X1, test_X1 = X1[:test_split], X1[test_split:]
    train_X2, test_X2 = X2[:test_split], X2[test_split:]
    train_Y1, test_Y1 = Y1[:test_split], Y1[test_split:]
    train_Y2, test_Y2 = Y2[:test_split], Y2[test_split:]

    val_split = int(len(train_X1) * (1 - valid_ratio))
    valid_X1, valid_X2 = train_X1[val_split:], train_X2[val_split:]
    valid_Y1, valid_Y2 = train_Y1[val_split:], train_Y2[val_split:]
    train_X1, train_X2 = train_X1[:val_split], train_X2[:val_split]
    train_Y1, train_Y2 = train_Y1[:val_split], train_Y2[:val_split]

    def to_loader(x1, x2, y1, y2, shuffle=True):
        ds = TensorDataset(
            torch.from_numpy(x1), torch.from_numpy(x2),
            torch.from_numpy(y1), torch.from_numpy(y2)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    dataloaders = {
        'train': to_loader(train_X1, train_X2, train_Y1, train_Y2, shuffle=True),
        'valid': to_loader(valid_X1, valid_X2, valid_Y1, valid_Y2, shuffle=False),
        'test': to_loader(test_X1, test_X2, test_Y1, test_Y2, shuffle=False),
    }

    print(f"[数据集划分] 训练: {len(train_X1)}, 验证: {len(valid_X1)}, 测试: {len(test_X1)}")
    forecast_list_str = [f'{h}h ({h/24:.1f}天)' if h > 24 else f'{h}h' for h in scaler_info['forecast_hours_list']]
    print(f"[预测目标] {target_cols}")
    print(f"[预测步长] {', '.join(forecast_list_str)}")
    print(f"[特征数量] 时变: {len(scaler_info['time_varying_cols'])}, "
          f"时间编码: {len(scaler_info['time_feature_cols'])}, "
          f"静态: {len(scaler_info['static_cols'])}, 总计: {num_features}")

    return dataloaders, scaler_info, num_features
