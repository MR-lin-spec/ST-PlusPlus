#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版DQN：快速筛选高质量图像级伪标签
适配ST++框架，5轮内完成训练，冻结后复用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import meanIOU

class SimpleDQN(nn.Module):
    """简化版DQN（无目标网络、无经验回放）"""
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64):
        super(SimpleDQN, self).__init__()
        # 状态：全局熵占比 + 两checkpoint预测一致性（mIOU）
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 动作：0-丢弃，1-保留

    def forward(self, x):
        """前向传播：输入状态返回动作价值"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PseudoLabelSelector:
    """伪标签筛选器：封装DQN训练与筛选逻辑"""
    def __init__(self, dataset_name, device='cuda'):
        self.device = device
        self.dataset_name = dataset_name
        self.num_classes = 21 if dataset_name == 'pascal' else 19
        self.dqn = SimpleDQN().to(device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.frozen = False  # DQN参数冻结标记

    def compute_state(self, pseudo_mask, checkpoint1_pred, checkpoint2_pred):
        """
        计算状态特征（2维）
        :param pseudo_mask: 教师模型伪标签 (H, W)
        :param checkpoint1_pred: 中期checkpoint预测 (H, W)
        :param checkpoint2_pred: 末期checkpoint预测 (H, W)
        :return: 状态向量 (2,)
        """
        # 1. 全局熵占比（高熵区域>30%标记为低可靠）
        prob = F.softmax(torch.from_numpy(pseudo_mask).float().unsqueeze(0).unsqueeze(0), dim=1)
        entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1)  # (1, H, W)
        high_entropy_ratio = (entropy > 0.3).float().mean().item()

        # 2. 两checkpoint预测一致性（mIOU）
        miou_calculator = meanIOU(self.num_classes)
        miou_calculator.add_batch([checkpoint1_pred], [checkpoint2_pred])
        consistency_miou = miou_calculator.evaluate()[1]  # 平均mIOU

        # 归一化到[0,1]区间
        state = np.array([high_entropy_ratio, consistency_miou], dtype=np.float32)
        return torch.from_numpy(state).to(self.device)

    def compute_reward(self, pseudo_mask):
        """
        即时奖励：伪标签置信度均值
        :param pseudo_mask: 教师模型伪标签 (H, W)
        :return: 奖励值（越大越可靠）
        """
        prob = F.softmax(torch.from_numpy(pseudo_mask).float().unsqueeze(0).unsqueeze(0), dim=1)
        conf_mean = prob.max(dim=1)[0].mean().item()  # 每个像素最大置信度的均值
        return conf_mean

    def train_dqn(self, train_data, epochs=5):
        """
        训练DQN（5轮内收敛）
        :param train_data: 训练数据列表，每个元素为 (pseudo_mask, ckpt1_pred, ckpt2_pred)
        :param epochs: 训练轮次（固定为5）
        """
        if self.frozen:
            raise RuntimeError("DQN已冻结，无法继续训练")
        
        self.dqn.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for data in train_data:
                pseudo_mask, ckpt1_pred, ckpt2_pred = data
                
                # 计算状态和奖励
                state = self.compute_state(pseudo_mask, ckpt1_pred, ckpt2_pred)
                reward = self.compute_reward(pseudo_mask)

                # 前向传播获取Q值
                q_values = self.dqn(state.unsqueeze(0))  # (1, 2)
                target_q = torch.tensor([[0.0, reward]], dtype=torch.float32).to(self.device)  # 动作1（保留）的目标Q=奖励

                # 计算损失并优化
                loss = self.loss_fn(q_values, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_data)
            print(f"DQN Train Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f}")
        
        # 训练3轮后冻结参数（固定策略）
        if epoch >= 2:
            self.frozen = True
            for param in self.dqn.parameters():
                param.requires_grad = False
            print("DQN训练完成，已冻结参数")

    def select_reliable(self, pseudo_mask, ckpt1_pred, ckpt2_pred):
        """
        筛选可靠伪标签（冻结后调用）
        :param pseudo_mask: 教师模型伪标签 (H, W)
        :param ckpt1_pred: 中期checkpoint预测 (H, W)
        :param ckpt2_pred: 末期checkpoint预测 (H, W)
        :return: bool，True=保留（高质量），False=丢弃（低质量）
        """
        if not self.frozen:
            raise RuntimeError("DQN未冻结，请先完成训练")
        
        self.dqn.eval()
        with torch.no_grad():
            state = self.compute_state(pseudo_mask, ckpt1_pred, ckpt2_pred)
            q_values = self.dqn(state.unsqueeze(0))  # (1, 2)
            action = torch.argmax(q_values, dim=1).item()  # 0=丢弃，1=保留
        return action == 1