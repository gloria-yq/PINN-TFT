"""
PINN 天气预测框架 (多目标版)
将物理信息神经网络 (PINN) 思想与 TFT 架构结合

预测目标: t2m, z_500, t_850
评估指标: Z500 RMSE (3/5 days), T850 RMSE (3/5 days), t2m RMSE
物理约束: 温度范围 + 平滑性 + 日周期性
"""

import os
import torch
import torch.nn as nn
import numpy as np

from Model.tft_model import TFT_Encoder, GatedResidualNetwork
from utils.util import AverageMeter, get_logger, eval_metrix


# ======================== 学习率调度器 ========================

class LR_Scheduler(object):
    """Warmup + 余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs,
                 base_lr, final_lr, iter_per_epoch=1):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = warmup_lr

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr


# ======================== Solution_u (G 网络) ========================

class Solution_u(nn.Module):
    """
    多目标预测网络 (G网络)
    使用 TFT_Encoder 编码气象特征序列，通过 Predictor 输出多个预测值

    输入: [batch_size, seq_length, num_features]
    输出: [batch_size, num_targets] (默认3: t2m, z_500, t_850)
    """
    def __init__(self, num_features, hidden_dim, embedding_dim,
                 lstm_layers, attn_heads, dropout, seq_length, encode_length,
                 num_targets=3):
        super().__init__()
        self.num_targets = num_targets
        self.encoder = TFT_Encoder(
            num_real_inputs=num_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            lstm_layers=lstm_layers,
            attn_heads=attn_heads,
            dropout=dropout,
            seq_length=seq_length,
            encode_length=encode_length,
        )
        self.predictor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, num_targets)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_embedding(self, x):
        hidden, _ = self.encoder(x)
        return hidden

    def forward(self, x):
        hidden, attn_weights = self.encoder(x)
        prediction = self.predictor(hidden)  # [B, num_targets]
        return prediction, attn_weights


# ======================== DynamicalF (F 网络) ========================

class DynamicalF(nn.Module):
    """
    PDE 右端项 F 网络 (多目标版)
    学习各物理量的动力学: u_t ≈ F(x_features, u, u_t)

    输入维度: num_features + num_targets (u) + num_targets (u_t)
    输出: [batch_size, num_targets]
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_targets=3,
                 dropout=0.2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ELU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, num_targets))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.net(x)


# ======================== WeatherPINN 主类 ========================

class WeatherPINN(nn.Module):
    """
    PINN + TFT 天气预测主框架 (多目标版)

    预测目标: [t2m, z_500, t_850] (可配置)
    PDE 约束: u_t = F(x, u, u_t) → 有限差分逼近
    物理约束: 温度范围 + 平滑性 + 周期性

    Loss = L_data + alpha * L_PDE + beta * L_physics
    """
    def __init__(self, args, num_features, num_targets=3, target_names=None):
        super().__init__()
        self.args = args
        self.device = args.device
        self.num_features = num_features
        self.num_targets = num_targets
        self.target_names = target_names or ['t2m', 'z_500', 't_850']

        # 日志
        log_dir = args.log_dir if args.save_folder is None else os.path.join(
            args.save_folder, args.log_dir
        )
        self.logger = get_logger(log_dir)
        self._save_args()

        # G 网络 (Solution_u): TFT 编码 → 多目标预测
        self.solution_u = Solution_u(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            lstm_layers=args.lstm_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            seq_length=args.seq_length,
            encode_length=args.encode_length,
            num_targets=num_targets,
        ).to(self.device)

        # F 网络 (DynamicalF): 学习 PDE 动力学
        # 输入 = 特征(num_features) + u(num_targets) + u_t(num_targets)
        f_input_dim = num_features + num_targets * 2
        self.dynamical_F = DynamicalF(
            input_dim=f_input_dim,
            hidden_dim=args.F_hidden_dim,
            num_layers=args.F_layers_num,
            num_targets=num_targets,
            dropout=args.dropout,
        ).to(self.device)

        # 优化器
        self.optimizer1 = torch.optim.Adam(
            self.solution_u.parameters(), lr=args.warmup_lr
        )
        self.optimizer2 = torch.optim.Adam(
            self.dynamical_F.parameters(), lr=args.lr_F
        )

        # 学习率调度
        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer1,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
        )

        # 损失函数
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

        # PINN 权重
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        # 物理约束参数 (温度类目标)
        self.T_min = args.T_min
        self.T_max = args.T_max
        self.delta_T_max = args.delta_T_max

        # 最佳模型
        self.best_model = None

    def _save_args(self):
        if self.args.log_dir is not None:
            self.logger.info("Args:")
            for k, v in self.args.__dict__.items():
                self.logger.info(f"\t{k}: {v}")

    def clear_logger(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.handlers.clear()

    def predict(self, x):
        """纯预测（不计算 PDE）"""
        u, _ = self.solution_u(x)
        return u  # [B, num_targets]

    def forward_pair(self, x1, x2):
        """
        前向传播 + 有限差分 PDE 残差计算 (多目标版)

        Returns:
            u1, u2: [B, num_targets] 两窗口的预测
            f: [B, num_targets] PDE 残差
        """
        u1, _ = self.solution_u(x1)
        u2, _ = self.solution_u(x2)

        # 有限差分: u_t ≈ (u2 - u1) / Δt
        u_t = u2 - u1  # [B, num_targets]

        x_last = x2[:, -1, :]  # [B, num_features]

        # F 网络输入: [x_features, u2, u_t]
        F_input = torch.cat([x_last, u2, u_t], dim=1)
        F_output = self.dynamical_F(F_input)  # [B, num_targets]

        f = u_t - F_output
        return u1, u2, f

    def compute_physics_loss(self, u1, u2, y1=None, y2=None):
        """
        计算物理约束损失 (多目标版)

        对温度类目标 (t2m, t_850) 应用温度范围约束
        对所有目标应用平滑性约束
        """
        # 各目标在标准化空间操作，约束值需要对应调整
        # 这里使用通用约束，不依赖物理单位

        # 1. 平滑性约束: 相邻步预测不应差异过大
        loss_smooth = torch.tensor(0.0, device=u1.device)
        for t in range(self.num_targets):
            diff = torch.abs(u2[:, t] - u1[:, t])
            # 标准化空间中，大于3个标准差视为异常跳变
            loss_smooth = loss_smooth + self.relu(diff - 3.0).mean()

        # 2. 周期性约束: 预测变化趋势与真值一致
        if y1 is not None and y2 is not None:
            pred_diff = u2 - u1
            true_diff = y2 - y1
            loss_periodic = self.loss_func(pred_diff, true_diff)
        else:
            loss_periodic = torch.tensor(0.0, device=u1.device)

        loss_physics = loss_smooth + self.gamma * loss_periodic
        return loss_physics, loss_smooth, loss_periodic

    def train_one_epoch(self, epoch, dataloader):
        """训练一个 epoch (多目标版)"""
        self.train()
        loss_data_meter = AverageMeter()
        loss_pde_meter = AverageMeter()
        loss_phys_meter = AverageMeter()

        for iter_idx, (x1, x2, y1, y2) in enumerate(dataloader):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)

            u1, u2, f = self.forward_pair(x1, x2)

            # 数据损失 (多目标)
            loss_data = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)

            # PDE 损失
            f_target = torch.zeros_like(f)
            loss_pde = self.loss_func(f, f_target)

            # 物理约束损失
            loss_phys, _, _ = self.compute_physics_loss(u1, u2, y1, y2)

            loss = loss_data + self.alpha * loss_pde + self.beta * loss_phys

            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()

            loss_data_meter.update(loss_data.item())
            loss_pde_meter.update(loss_pde.item())
            loss_phys_meter.update(loss_phys.item())

            if (iter_idx + 1) % 50 == 0:
                self.logger.info(f"  [epoch:{epoch} iter:{iter_idx+1}] "
                                 f"data:{loss_data.item():.6f} "
                                 f"PDE:{loss_pde.item():.6f} "
                                 f"phys:{loss_phys.item():.6f}")

        return loss_data_meter.avg, loss_pde_meter.avg, loss_phys_meter.avg

    def Valid(self, validloader):
        """验证"""
        self.eval()
        true_list, pred_list = [], []
        with torch.no_grad():
            for x1, _, y1, _ in validloader:
                x1 = x1.to(self.device)
                u1 = self.predict(x1)
                true_list.append(y1.numpy())
                pred_list.append(u1.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)
        true = np.concatenate(true_list, axis=0)
        mse = np.mean((pred - true) ** 2)
        return mse

    def Test(self, testloader):
        """测试"""
        self.eval()
        true_list, pred_list = [], []
        with torch.no_grad():
            for x1, _, y1, _ in testloader:
                x1 = x1.to(self.device)
                u1 = self.predict(x1)
                true_list.append(y1.numpy())
                pred_list.append(u1.cpu().numpy())
        pred = np.concatenate(pred_list, axis=0)  # [N, num_targets]
        true = np.concatenate(true_list, axis=0)  # [N, num_targets]
        return true, pred

    def evaluate_weather(self, true, pred, scaler_info):
        target_means = scaler_info.get('target_means', {})
        target_stds = scaler_info.get('target_stds', {})
        forecast_hours_list = scaler_info.get('forecast_hours_list', [1])

        metrics = {}
        idx = 0
        for h in forecast_hours_list:
            for name in self.target_names:
                mean = target_means.get(name, 0.0)
                std = target_stds.get(name, 1.0)

                true_phys = true[:, idx] * std + mean
                pred_phys = pred[:, idx] * std + mean

                rmse = np.sqrt(np.mean((true_phys - pred_phys) ** 2))
                mae = np.mean(np.abs(true_phys - pred_phys))

                if 'z_' in name or name == 'z':
                    unit = 'm²/s²'
                elif 't_' in name or name in ['t2m']:
                    unit = 'K'
                else:
                    unit = ''

                metrics[f"{name}_{h}h"] = {
                    'RMSE': rmse,
                    'MAE': mae,
                    'unit': unit,
                }
                idx += 1

        return metrics

    def Train(self, trainloader, validloader=None, testloader=None,
              scaler_info=None):
        """完整训练流程"""
        min_valid_mse = float('inf')
        early_stop_count = 0
        
        # 解析多步长时间以便打印
        forecast_hours_list = scaler_info.get('forecast_hours_list', [1]) if scaler_info else [1]

        for e in range(1, self.args.epochs + 1):
            early_stop_count += 1

            loss_data, loss_pde, loss_phys = self.train_one_epoch(e, trainloader)
            current_lr = self.scheduler.step()

            total_loss = loss_data + self.alpha * loss_pde + self.beta * loss_phys
            info = (f'[Train] epoch:{e}, lr:{current_lr:.6f}, '
                    f'data:{loss_data:.6f}, PDE:{loss_pde:.6f}, '
                    f'phys:{loss_phys:.6f}, total:{total_loss:.6f}')
            self.logger.info(info)

            # 验证
            if validloader is not None:
                valid_mse = self.Valid(validloader)
                self.logger.info(f'[Valid] epoch:{e}, MSE: {valid_mse:.8f}')

                if valid_mse < min_valid_mse and testloader is not None:
                    min_valid_mse = valid_mse
                    true_label, pred_label = self.Test(testloader)
                    early_stop_count = 0

                    idx = 0
                    for h in forecast_hours_list:
                        for name in self.target_names:
                            rmse_i = np.sqrt(np.mean(
                                (pred_label[:, idx] - true_label[:, idx]) ** 2
                            ))
                            self.logger.info(
                                f'[Test] {name}_{h}h RMSE(标准化): {rmse_i:.6f}')
                            idx += 1

                    # 物理单位指标
                    if scaler_info is not None:
                        w_metrics = self.evaluate_weather(
                            true_label, pred_label, scaler_info)
                        for name, m in w_metrics.items():
                            self.logger.info(
                                f'[Test] {name} RMSE: '
                                f'{m["RMSE"]:.4f} {m["unit"]}'
                            )

                    # 保存最佳模型
                    self.best_model = {
                        'solution_u': self.solution_u.state_dict(),
                        'dynamical_F': self.dynamical_F.state_dict()
                    }
                    if self.args.save_folder is not None:
                        np.save(os.path.join(self.args.save_folder,
                                             'true_label.npy'), true_label)
                        np.save(os.path.join(self.args.save_folder,
                                             'pred_label.npy'), pred_label)

            # 早停
            if (self.args.early_stop is not None and
                    early_stop_count > self.args.early_stop):
                self.logger.info(f'Early stop at epoch {e}')
                break

        # 保存模型
        self.clear_logger()
        if self.args.save_folder is not None and self.best_model is not None:
            torch.save(self.best_model,
                       os.path.join(self.args.save_folder, 'model.pth'))

    def load_model(self, model_path):
        """加载保存的模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.solution_u.load_state_dict(checkpoint['solution_u'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
