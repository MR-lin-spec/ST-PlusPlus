#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数合集
新增：
    init_rl_components  – 延迟初始化 RL 智能体与损失
    RLSummaryWriter     – 可选 tensorboard 封装（暂无额外逻辑，可直接用 SummaryWriter）
其余与原版完全一致
"""
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


# ----------  原工具  ----------
def count_params(model):
    """返回模型参数量（单位：M）"""
    return sum(p.numel() for p in model.parameters()) / 1e6


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (
            self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)
        )
        return iu, np.nanmean(iu)
    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))


def color_map(dataset='pascal'):
    """
    返回调色板（P 模式 PNG 需要）
    支持 pascal / cityscapes
    """
    cmap = np.zeros((256, 3), dtype=np.uint8)
    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r |= bitget(c, 0) << 7 - j
                g |= bitget(c, 1) << 7 - j
                b |= bitget(c, 2) << 7 - j
                c >>= 3
            cmap[i] = np.array([r, g, b])
    elif dataset == 'cityscapes':
        cityscape_palette = [
            [128, 64, 128], [244, 35, 232], [70, 70, 70],
            [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35],
            [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230],
            [119, 11, 32]
        ]
        for i, color in enumerate(cityscape_palette):
            cmap[i] = np.array(color)
    return cmap


# ----------  新增 RL 支持  ----------
def init_rl_components(model, args, device):
    """
    延迟初始化 RL 组件
    仅当用户指定 --algorithm st++_rl 且进入 semi_train 阶段时调用
    返回 (rl_agent, rl_loss_fn)
    """
    # 避免硬依赖，ImportError 在外部捕获
    from model.rl.dynamic_threshold import DynamicThreshold
    from model.rl.RLmodel import PixelSACAgent, RLEntropyHighLoss

    num_classes = 21 if args.dataset == 'pascal' else 19
    rl_agent = PixelSACAgent(
        feat_dim=256, lr=3e-4, gamma=0.99, tau=0.005, ent_lambda=0.2
    ).to(device)

    rl_loss_fn = RLEntropyHighLoss(
        reward_scale=0.001, consist_weight=0.5
    ).to(device)
    rl_loss_fn.init_channel_proj(in_c=num_classes, device=device)

    # 把 channel_proj 参数也交给 RL 优化器
    rl_params = list(rl_agent.parameters()) + \
                list(rl_loss_fn.channel_proj.parameters())
    rl_agent.pi_optim = torch.optim.Adam(rl_params, lr=3e-4)
    return rl_agent, rl_loss_fn