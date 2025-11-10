#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版DQN：解决策略失效 + 样本难度感知 + 策略熵正则
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
from typing import Generator, List, Tuple, Optional
from utils import meanIOU

class SimpleDQN(nn.Module):
    """改进DQN网络：增加Dropout防止过拟合"""
    def __init__(self, state_dim=2, action_dim=2, hidden_dim=64):  # 增大hidden_dim
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.3)  # 新增：防止Q值坍塌
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class PseudoLabelSelector:
    def __init__(self, dataset_name: str, device: str = 'cuda',
                 high_entropy_thresh: float = 0.5, max_allowed_loss: float = 10.0,
                 max_abnormal_ratio: float = 0.50,  # 放宽到50%，避免误报警
                 conv_thresh_loss: float = 5.0,
                 entropy_reg_weight: float =0.5):  # 新增：策略熵正则权重
        # 基础配置
        self.device = device
        self.dataset_name = dataset_name
        self.num_classes = 21 if dataset_name == 'pascal' else 19
        self.high_entropy_thresh = high_entropy_thresh

        # 训练稳定性配置
        self.max_allowed_loss = max_allowed_loss
        self.initial_lr = 5e-4
        self.min_lr = 5e-5
        self.grad_clip_norm = 3.0
        self.max_abnormal_ratio = max_abnormal_ratio
        self.conv_thresh_loss = conv_thresh_loss
        
        # 新增：策略熵正则，防止Q值坍塌
        self.entropy_reg_weight = entropy_reg_weight

        # 模型与优化器初始化
        self.dqn = SimpleDQN().to(device)
        self.optimizer = torch.optim.Adam(
            self.dqn.parameters(),
            lr=self.initial_lr,
            weight_decay=1e-5,
            eps=1e-8
        )
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.miou_calc = meanIOU(self.num_classes)
        
        self.frozen = False
        self.current_lr = self.initial_lr
        self.loss_history = []

    def _validate_inputs(self, pseudo_logits: np.ndarray, checkpoint1_pred: np.ndarray, 
                        checkpoint2_pred: np.ndarray) -> bool:
        """诊断式验证：打印具体失败原因"""
        actual_channels = pseudo_logits.shape[0]
        max_pred_class = max(checkpoint1_pred.max(), checkpoint2_pred.max())
        
        # 维度检查
        if pseudo_logits.ndim != 3:
            print(f"[验证失败] pseudo_logits ndim={pseudo_logits.ndim} ≠ 3")
            return False
            
        # 通道数匹配检查（容错±1，因可能有背景类）
        if not (actual_channels in [self.num_classes, self.num_classes+1]):
            print(f"[验证失败] 通道数不匹配: 期望{self.num_classes}，实际{actual_channels}")
            return False
        
        # 预测图维度检查
        if checkpoint1_pred.ndim != 2 or checkpoint2_pred.ndim != 2:
            print(f"[验证失败] 预测图维度错误: pred1={checkpoint1_pred.ndim}, pred2={checkpoint2_pred.ndim}")
            return False
        
        # 类别索引越界检查与自动修复（关键修复）
        if (checkpoint1_pred.min() < 0) or (checkpoint1_pred.max() >= actual_channels):
            print(f"[验证失败] ckpt1_pred 索引越界: min={checkpoint1_pred.min()}, max={checkpoint1_pred.max()}, 通道数={actual_channels}")
            return False
        if (checkpoint2_pred.min() < 0) or (checkpoint2_pred.max() >= actual_channels):
            print(f"[验证失败] ckpt2_pred 索引越界: min={checkpoint2_pred.min()}, max={checkpoint2_pred.max()}, 通道数={actual_channels}")
            return False
        
        return True

    def compute_state_batch(self, pseudo_logits_batch: List[np.ndarray], 
                           ckpt1_pred_batch: List[np.ndarray], 
                           ckpt2_pred_batch: List[np.ndarray]) -> Tuple[torch.Tensor, List[bool], List[dict]]:
        """批量计算状态特征：支持变尺寸输入 + 返回诊断信息"""
        batch_size = len(pseudo_logits_batch)
        states = []
        diagnostics = []  # 新增：存储每个样本的诊断信息
        
        # 批量校验
        valid_mask = [self._validate_inputs(l, p1, p2) for l, p1, p2 in 
                      zip(pseudo_logits_batch, ckpt1_pred_batch, ckpt2_pred_batch)]
        abnormal_count = sum(not m for m in valid_mask)
        abnormal_ratio = abnormal_count / batch_size if batch_size > 0 else 0.0
        
        # 异常报警（仅在异常比例>0时打印，避免刷屏）
        if abnormal_ratio > self.max_abnormal_ratio:
            print(f"[警告] 异常样本占比{abnormal_ratio:.2%}")
        
        # 逐个计算状态
        for idx in range(batch_size):
            if not valid_mask[idx]:
                states.append(np.array([0.5, 0.5], dtype=np.float32))
                diagnostics.append({"high_entropy_ratio": 0.5, "consistency_miou": 0.5})
                continue
            
            pseudo_logits = pseudo_logits_batch[idx]
            ckpt1_pred = ckpt1_pred_batch[idx]
            ckpt2_pred = ckpt2_pred_batch[idx]
            
            # 计算高熵占比
            logits_tensor = torch.from_numpy(pseudo_logits).float().to(self.device).unsqueeze(0)
            prob = F.softmax(logits_tensor, dim=1)
            pixel_entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1)
            max_theory_entropy = np.log(self.num_classes)
            pixel_entropy = torch.clip(pixel_entropy, 0, max_theory_entropy)
            high_entropy_pixels = (pixel_entropy > self.high_entropy_thresh).float()
            high_entropy_ratio = torch.clip(high_entropy_pixels.mean(), 0.0, 1.0).cpu().item()
            
            # 计算一致性mIOU
            self.miou_calc.reset()
            self.miou_calc.add_batch([ckpt1_pred], [ckpt2_pred])
            miou = self.miou_calc.evaluate()[1]
            consistency_miou = max(0.0, min(miou, 1.0))
            
            # 记录诊断信息
            states.append(np.array([high_entropy_ratio, consistency_miou], dtype=np.float32))
            diagnostics.append({
                "high_entropy_ratio": high_entropy_ratio,
                "consistency_miou": consistency_miou
            })
            
            del logits_tensor, prob, pixel_entropy, high_entropy_pixels
        
        torch.cuda.empty_cache()
        return torch.from_numpy(np.stack(states)).to(self.device), valid_mask, diagnostics

    def compute_reward_batch(self, pseudo_logits_batch):
        """重构奖励函数：引入相对质量评分"""
        rewards = []
        
        for pseudo_logits in pseudo_logits_batch:
            logits_tensor = torch.from_numpy(pseudo_logits).float().to(self.device).unsqueeze(0)
            prob = F.softmax(logits_tensor, dim=1)
            pixel_conf = prob.max(dim=1)[0]
            conf_mean = pixel_conf.mean().cpu().item()
            
            # ===== 关键修改：奖励归一化与基准 =====
            # 使用批次内相对置信度，而非绝对阈值
            # 奖励范围调整为[-1, 1]，且以批次中位数为基准
            batch_confidences = []  # 临时存储所有样本置信度
            
            # 实际实现中需要在compute_reward_batch外部计算批次的conf_mean
            # 这里改为动态基准
            
            # 新奖励：相对于批次平均值的置信度优势
            # 临时方案：直接使用conf_mean，但后续会减去批次中位数
            base_reward = (conf_mean - 0.5) * 2  # 放大到[-1, 1]
            
            # 惩罚项保持不变
            pixel_entropy = -(prob * torch.log(prob + 1e-12)).sum(dim=1)
            max_theory_entropy = np.log(self.num_classes)
            entropy_ratio = pixel_entropy.mean().cpu().item() / max_theory_entropy
            entropy_penalty = entropy_ratio * 0.3
            
            reward = base_reward - entropy_penalty
            
            rewards.append(reward)
            
            del logits_tensor, prob, pixel_conf, pixel_entropy
        
        # ===== 关键修改：批次级奖励归一化 =====
        rewards = np.array(rewards, dtype=np.float32)
        # 减去批次中位数，使奖励有正有负
        median_reward = np.median(rewards)
        rewards -= median_reward
        
        torch.cuda.empty_cache()
        return rewards

    def _adjust_learning_rate(self, current_epoch_loss: float):
        """动态学习率调整"""
        self.loss_history.append(current_epoch_loss)
        if len(self.loss_history) > 10:
            self.loss_history = self.loss_history[-10:]
        
        if current_epoch_loss > self.max_allowed_loss:
            new_lr = self.current_lr * 0.5
            self.current_lr = max(new_lr, self.min_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            print(f"[动态调参] Loss={current_epoch_loss:.2f} > 阈值，学习率调整为{self.current_lr:.6f}")
        
        if len(self.loss_history) >= 3:
            if (self.loss_history[-1] < self.loss_history[-2] < self.loss_history[-3]) and \
               (self.loss_history[-1] < self.max_allowed_loss):
                new_lr = self.current_lr * 1.2
                self.current_lr = min(new_lr, self.initial_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                print(f"[动态调参] 连续3轮Loss下降，学习率恢复为{self.current_lr:.6f}")

    def train_dqn(self, train_data_generator: Generator, epochs: int = 5, 
                 initial_eps: float = 0.9, final_eps: float = 0.1, batch_size: int = 32):
        """训练DQN：增加策略熵正则 + 诊断日志"""
        if self.frozen:
            raise RuntimeError("ERROR: DQN已冻结，无法重复训练！")
        
        self.dqn.train()
        print(f"[DQN训练启动] 批量大小：{batch_size} | 总轮次：{epochs} | 收敛阈值Loss：{self.conv_thresh_loss}")
        print(f"[状态特征统计] 将打印每轮次状态分布...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_valid_samples = 0
            eps = initial_eps - (initial_eps - final_eps) * (epoch / (epochs - 1)) if epochs > 1 else final_eps
            
            # 新增：收集状态特征用于诊断
            all_states = []
            all_rewards = []
            
            for batch_data in train_data_generator(batch_size):
                pseudo_logits_batch, ckpt1_pred_batch, ckpt2_pred_batch = zip(*batch_data)
                
                # 1. 批量计算状态 + 诊断信息
                states, valid_mask, diagnostics = self.compute_state_batch(pseudo_logits_batch, ckpt1_pred_batch, ckpt2_pred_batch)
                rewards = self.compute_reward_batch(pseudo_logits_batch)
                
                # 收集用于诊断
                all_states.extend([d for d in diagnostics])
                all_rewards.extend(rewards.tolist())
                
                # 2. 过滤无效样本
                valid_indices = [i for i, m in enumerate(valid_mask) if m]
                if not valid_indices:
                    continue
                valid_states = states[valid_indices]
                valid_rewards = torch.from_numpy(rewards[valid_indices]).float().to(self.device)
                
                # 3. 批量动作选择
                batch_actions = []
                for idx in range(len(valid_states)):
                    if random.random() < eps:
                        batch_actions.append(random.choice([0, 1]))
                    else:
                        with torch.no_grad():
                            q_val = self.dqn(valid_states[idx:idx+1])
                        batch_actions.append(torch.argmax(q_val, dim=1).item())
                batch_actions = torch.tensor(batch_actions, dtype=torch.long).to(self.device)
                
                # 4. Q值更新
                q_values = self.dqn(valid_states)
                target_q = q_values.clone()
                target_q[range(len(target_q)), batch_actions] = valid_rewards
                
                # 5. 损失计算 + 策略熵正则
                mse_loss = self.loss_fn(q_values, target_q)
                
                # 新增：策略熵正则（防止Q值坍塌）
                probs = F.softmax(q_values, dim=1)
                policy_entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1).mean()
                entropy_loss = -self.entropy_reg_weight * policy_entropy  # 鼓励探索
                
                batch_loss = mse_loss + entropy_loss
                batch_loss = torch.clip(batch_loss, 0.0, self.max_allowed_loss)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                
                total_loss += batch_loss.item() * len(valid_states)
                total_valid_samples += len(valid_states)
            
            # 6. 计算平均Loss + 打印诊断信息
            if total_valid_samples == 0:
                print(f"[轮次 {epoch+1}/{epochs}] 警告：本批次无有效样本，跳过")
                continue
                
            avg_epoch_loss = total_loss / total_valid_samples
            
            # 新增：打印状态特征分布
            if all_states:
                entropy_ratios = [s['high_entropy_ratio'] for s in all_states]
                miou_scores = [s['consistency_miou'] for s in all_states]
                print(f"[诊断] 状态分布 → 熵比: μ={np.mean(entropy_ratios):.3f}, σ={np.std(entropy_ratios):.3f} | mIOU: μ={np.mean(miou_scores):.3f}, σ={np.std(miou_scores):.3f}")
                print(f"[诊断] 奖励分布 → μ={np.mean(all_rewards):.3f}, σ={np.std(all_rewards):.3f}, min={np.min(all_rewards):.3f}, max={np.max(all_rewards):.3f}")
            
            print(f"[DQN训练轮次 {epoch+1}/{epochs}] 平均Loss：{avg_epoch_loss:.6f} | 有效样本：{total_valid_samples}")
            
            self._adjust_learning_rate(avg_epoch_loss)
            
          
        
        # 7. 冻结模型
        self.frozen = True
        for param in self.dqn.parameters():
            param.requires_grad = False
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[DQN训练完成] 模型已冻结")

    def select_reliable_batch(self, pseudo_logits_batch: List[np.ndarray], 
                             ckpt1_pred_batch: List[np.ndarray], 
                             ckpt2_pred_batch: List[np.ndarray]) -> np.ndarray:
        """批量筛选可靠伪标签"""
        if not self.frozen:
            raise RuntimeError("ERROR: DQN未训练或未冻结，请先调用train_dqn()完成训练！")
        
        self.dqn.eval()
        with torch.no_grad():
            states, _, _ = self.compute_state_batch(pseudo_logits_batch, ckpt1_pred_batch, ckpt2_pred_batch)
            q_values = self.dqn(states)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            
            # 新增：打印Q值分布用于诊断
            q_diff = q_values[:, 1] - q_values[:, 0]  # 保留与丢弃的Q值差
            print(f"[筛选诊断] Q值差分布 → μ={q_diff.mean():.3f}, σ={q_diff.std():.3f}, 正样本比例={(q_diff > 0).float().mean():.2%}")
        
        return actions == 1

    def select_reliable(self, pseudo_logits: np.ndarray, checkpoint1_pred: np.ndarray, 
                       checkpoint2_pred: np.ndarray) -> bool:
        """单样本筛选接口：兼容原有调用逻辑"""
        return self.select_reliable_batch([pseudo_logits], [checkpoint1_pred], [checkpoint2_pred])[0]

    def get_dqn_state_dict(self):
        if not self.frozen:
            print("WARNING: DQN未冻结，保存需谨慎！")
        return self.dqn.state_dict()

    def load_dqn_state_dict(self, state_dict):
        self.dqn.load_state_dict(state_dict)
        self.frozen = True
        for param in self.dqn.parameters():
            param.requires_grad = False
        print("INFO: DQN模型参数加载完成并冻结")