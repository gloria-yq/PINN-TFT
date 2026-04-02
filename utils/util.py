"""
工具函数模块
包含日志、评估指标、均值计算器等通用工具
"""

from sklearn import metrics
import numpy as np
import logging


def get_logger(log_name='log.txt'):
    """创建日志记录器"""
    logger = logging.getLogger('weather_pinn')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )

    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件输出
    if log_name is not None:
        handler = logging.FileHandler(log_name, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class AverageMeter(object):
    """计算并存储均值和当前值"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_metrix(pred_label, true_label):
    """
    计算评估指标
    Returns: [MAE, MAPE, MSE, RMSE]
    """
    MAE = metrics.mean_absolute_error(true_label, pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label, pred_label)
    MSE = metrics.mean_squared_error(true_label, pred_label)
    RMSE = np.sqrt(MSE)
    return [MAE, MAPE, MSE, RMSE]


def write_to_txt(txt_name, txt):
    """追加写入文本文件"""
    with open(txt_name, 'a', encoding='utf-8') as f:
        f.write(txt)
        f.write('\n')
