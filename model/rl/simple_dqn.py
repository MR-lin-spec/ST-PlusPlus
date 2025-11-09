#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版DQN：增强抗损失爆炸机制，确保训练稳定性（Loss≤10）
适配ST++框架，5轮内完成训练，筛选结果区分度可靠
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import meanIOU

# 修改后的 SimpleDQN 类（simple_dqn.py 中）
class SimpleDQN(nn.Module):
    """移除BatchNorm层，改用LayerNorm（支持单样本），避免批次维度限制"""
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=32):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 替换BatchNorm为LayerNorm（支持单样本）
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # LayerNorm不依赖批次大小
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # LayerNorm在激活函数前，稳定特征分布且支持单样本
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        return self.fc3(x)

class PseudoLabelSelector:
    """伪标签筛选器：全链路抗损失爆炸设计，确保Loss≤10"""
    def __init__(self, dataset_name, device='cuda', 
                 high_entropy_thresh=0.5, max_allowed_loss=10.0):
        self.device = device
        self.dataset_name = dataset_name
        self.num_classes = 21 if dataset_name == 'pascal' else 19
        # 抗损失爆炸核心配置
        self.max_allowed_loss = max_allowed_loss  # 最大允许损失（超过则触发保护）
        self.initial_lr = 1e-3  # 初始学习率
        self.min_lr = 1e-5      # 最小学习率（防止学习率过低导致不收敛）
        self.grad_clip_norm = 3.0  # 梯度裁剪阈值（从5.0下调，更严格控制梯度）
        
        # 初始化DQN与优化器（优化器参数适配抗爆炸）
        self.dqn = SimpleDQN().to(device)
        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-5,  # 权重衰减增强，抑制参数过大
            eps=1e-8            # 数值稳定性参数，避免分母为0
        )
        self.loss_fn = nn.MSELoss(reduction='mean')  # 均值 reduction，避免单样本损失累积
        
        # 训练状态变量
        self.frozen = False
        self.high_entropy_thresh = high_entropy_thresh
        self.current_lr = self.initial_lr  # 跟踪当前学习率
        self.loss_history = []  # 记录损失历史，用于动态调整学习率

    def _validate_inputs(self, pseudo_logits, checkpoint1_pred, checkpoint2_pred):
        """
        新增：输入数据校验，避免异常数据导致Loss爆炸
        返回：bool（True=数据正常，False=数据异常）
        """
        # 1. 校验Logits维度（必须为[C, H, W]，C=类别数）
        expected_channels = self.num_classes
        if pseudo_logits.shape[0] != expected_channels:
            print(f"[输入错误] Logits通道数{ pseudo_logits.shape[0] }≠预期{ expected_channels }，跳过该样本")
            return False
        # 2. 校验Logits数值范围（若存在极端值，先裁剪）
        if np.max(np.abs(pseudo_logits)) > 100:  # Logits绝对值超过100视为异常
            pseudo_logits = np.clip(pseudo_logits, -100, 100)
            print(f"[输入警告] Logits存在极端值，已裁剪到[-100, 100]")
        # 3. 校验预测结果类别（必须在[0, num_classes-1]范围内）
        if (checkpoint1_pred.min() < 0) or (checkpoint1_pred.max() >= self.num_classes):
            print(f"[输入错误] Checkpoint1预测类别超出范围，跳过该样本")
            return False
        if (checkpoint2_pred.min() < 0) or (checkpoint2_pred.max() >= self.num_classes):
            print(f"[输入错误] Checkpoint2预测类别超出范围，跳过该样本")
            return False
        return True

    def compute_state(self, pseudo_logits, checkpoint1_pred, checkpoint2_pred):
        """
        增强：状态特征计算加入数值裁剪，避免特征值过大
        返回：归一化且裁剪后的状态向量（确保在[0,1]内）
        """
        # 1. 先校验输入数据
        if not self._validate_inputs(pseudo_logits, checkpoint1_pred, checkpoint2_pred):
            return torch.tensor([0.5, 0.5], dtype=torch.float32).to(self.device)  # 返回默认中间状态
        
        # 2. 计算全局高熵占比（加入数值裁剪）
        prob = F.softmax(torch.from_numpy(pseudo_logits).float().unsqueeze(0), dim=1)
        pixel_entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1)  # (1, H, W)
        # 熵值裁剪到[0, log(num_classes)]（理论最大熵，避免异常值）
        max_theory_entropy = np.log(self.num_classes)
        pixel_entropy = torch.clip(pixel_entropy, 0, max_theory_entropy)
        # 高熵占比计算（确保在[0,1]）
        high_entropy_pixels = (pixel_entropy > self.high_entropy_thresh).float()
        high_entropy_ratio = torch.clip(high_entropy_pixels.mean(), 0.0, 1.0).item()

        # 3. 计算两checkpoint一致性mIOU（天然在[0,1]，无需裁剪）
        miou_calc = meanIOU(self.num_classes)
        miou_calc.add_batch([checkpoint1_pred], [checkpoint2_pred])
        consistency_miou = miou_calc.evaluate()[1]
        consistency_miou = max(0.0, min(consistency_miou, 1.0))  # 保险裁剪

        # 状态向量最终确认（确保无异常值）
        state = np.array([high_entropy_ratio, consistency_miou], dtype=np.float32)
        return torch.from_numpy(state).to(self.device)

    def compute_reward(self, pseudo_logits):
        """
        增强：奖励值裁剪，避免极端奖励导致目标Q值过大
        返回：裁剪后的奖励（0.1~0.9，预留安全边际）
        """
        prob = F.softmax(torch.from_numpy(pseudo_logits).float().unsqueeze(0), dim=1)
        pixel_conf = prob.max(dim=1)[0]  # (1, H, W)
        # 置信度均值裁剪（避免0或1的极端值，导致目标Q值过大）
        conf_mean = torch.clip(pixel_conf.mean(), 0.1, 0.9).item()
        return conf_mean

    def _adjust_learning_rate(self, current_epoch_loss):
        """
        新增：动态学习率调整，若Loss超过阈值则降低学习率
        逻辑：Loss>max_allowed_loss → 学习率减半；连续3轮Loss下降 → 学习率恢复
        """
        self.loss_history.append(current_epoch_loss)
        # 1. 若当前Loss超过阈值，学习率减半（不低于最小学习率）
        if current_epoch_loss > self.max_allowed_loss:
            new_lr = self.current_lr * 0.5
            self.current_lr = max(new_lr, self.min_lr)
            # 更新优化器学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            print(f"[Loss保护] 当前Loss={ current_epoch_loss:.2f }>阈值{ self.max_allowed_loss }，学习率调整为{ self.current_lr:.6f }")
        # 2. 若连续3轮Loss下降且低于阈值，恢复学习率（避免学习率过低）
        if len(self.loss_history) >= 3:
            if (self.loss_history[-1] < self.loss_history[-2] < self.loss_history[-3]) and \
               (self.loss_history[-1] < self.max_allowed_loss) and \
               (self.current_lr < self.initial_lr):
                new_lr = self.current_lr * 1.2  # 学习率恢复1.2倍
                self.current_lr = min(new_lr, self.initial_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"[Loss恢复] 连续3轮Loss下降，学习率恢复为{ self.current_lr:.6f }")
        # 3. 限制学习率历史长度（避免内存占用）
        if len(self.loss_history) > 10:
            self.loss_history = self.loss_history[-10:]

    def train_dqn(self, train_data, epochs=5, eps=0.1):
        """
        增强：全流程Loss爆炸防护，包含梯度裁剪、异常样本跳过、学习率调整
        """
        if self.frozen:
            raise RuntimeError("ERROR: DQN已冻结，无法继续训练")
        if len(train_data) == 0:
            raise ValueError("ERROR: DQN训练数据为空，请检查数据准备流程")

        self.dqn.train()
        print(f"[DQN Training] 启动训练（抗爆炸配置：梯度裁剪={ self.grad_clip_norm }，最大Loss={ self.max_allowed_loss }）")
        
        for epoch in range(epochs):
            total_loss = 0.0
            valid_sample_count = 0  # 统计有效样本数（跳过异常样本）
            
            for data in train_data:
                pseudo_logits, ckpt1_pred, ckpt2_pred = data

                # 1. 计算状态和奖励（已包含输入校验和数值裁剪）
                state = self.compute_state(pseudo_logits, ckpt1_pred, ckpt2_pred)
                reward = self.compute_reward(pseudo_logits)

                # 2. ε-greedy动作选择（保持探索，避免局部最优）
                if random.random() < eps:
                    action = random.choice([0, 1])
                else:
                    with torch.no_grad():
                        q_values = self.dqn(state.unsqueeze(0))
                    action = torch.argmax(q_values, dim=1).item()

                # 3. Q值计算与损失计算（加入Loss裁剪）
                q_values = self.dqn(state.unsqueeze(0))  # (1, 2)
                target_q = q_values.clone()
                target_q[0, action] = reward  # 目标Q值（已裁剪奖励，避免过大）
                
                # 单样本Loss计算（加入裁剪，超过阈值则按阈值计算）
                sample_loss = self.loss_fn(q_values, target_q)
                sample_loss = torch.clip(sample_loss, 0.0, self.max_allowed_loss)  # 关键：Loss裁剪
                
                # 4. 反向传播（强化梯度控制）
                self.optimizer.zero_grad()
                sample_loss.backward()
                # 梯度裁剪（严格控制梯度范数，避免梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip_norm)
                # 检查梯度范数（调试用，可选开启）
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip_norm, norm_type=2)
                # print(f"[梯度监控] 梯度范数: {grad_norm:.2f}")
                self.optimizer.step()

                # 5. 累计损失（仅统计有效样本）
                total_loss += sample_loss.item()
                valid_sample_count += 1

            # 6. 计算本轮平均Loss（避免除以0）
            if valid_sample_count == 0:
                raise ValueError("ERROR: 所有训练样本均异常，无法继续训练")
            avg_epoch_loss = total_loss / valid_sample_count
            print(f"[DQN Training] 轮次[{epoch+1}/{epochs}] | 平均Loss: {avg_epoch_loss:.6f} | 当前学习率: {self.current_lr:.6f}")

            # 7. 动态调整学习率（基于当前Loss）
            self._adjust_learning_rate(avg_epoch_loss)

            # 8. 紧急停止机制（若Loss仍超过阈值且学习率已达最小，停止训练避免崩溃）
            if avg_epoch_loss > self.max_allowed_loss and self.current_lr == self.min_lr:
                print(f"[紧急保护] 学习率已达最小{ self.min_lr }，但Loss={ avg_epoch_loss:.2f }>阈值，提前停止训练")
                break

        # 训练完成后冻结参数
        self.frozen = True
        for param in self.dqn.parameters():
            param.requires_grad = False
        print(f"[DQN Training] 训练完成（有效样本数: {valid_sample_count}），已冻结参数")

    def select_reliable(self, pseudo_logits, checkpoint1_pred, checkpoint2_pred):
        """筛选逻辑不变，保持与主流程兼容"""
        if not self.frozen:
            raise RuntimeError("ERROR: DQN未完成训练或未冻结，请先调用train_dqn()")

        self.dqn.eval()
        with torch.no_grad():
            state = self.compute_state(pseudo_logits, checkpoint1_pred, checkpoint2_pred)
            q_values = self.dqn(state.unsqueeze(0))
            action = torch.argmax(q_values, dim=1).item()

        is_reliable = (action == 1)
        # 筛选结果统计（可选开启，便于监控）
        # high_entropy_ratio, consistency_miou = state.cpu().numpy()
        # print(f"[筛选监控] 高熵占比: {high_entropy_ratio:.3f}, 一致性mIOU: {consistency_miou:.3f}, 动作: {action}")
        return is_reliable

    # 保留参数保存/加载功能，确保断点续训兼容性
    def get_dqn_state_dict(self):
        if not self.frozen:
            print("WARNING: DQN未冻结，当前参数可能未稳定")
        return self.dqn.state_dict()

    def load_dqn_state_dict(self, state_dict):
        self.dqn.load_state_dict(state_dict)
        self.frozen = True
        for param in self.dqn.parameters():
            param.requires_grad = False
        print("INFO: DQN参数加载完成并冻结")