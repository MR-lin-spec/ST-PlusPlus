import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Optional, Tuple

# ---------- 工具函数：软更新与梯度裁剪 ----------
def soft_update(target, source, tau):
    """软更新目标网络参数（τ 越小，目标网络越稳定）"""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(tau * s.data + (1 - tau) * t.data)

def clip_grad_norm(parameters, max_norm=5.0):
    """RL 组件梯度裁剪（防止梯度爆炸）"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return 0.0
    max_norm = float(max_norm)
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

# ---------- 网络模块：MLPHead（策略网络）+ QHead（价值网络） ----------
class MLPHead(nn.Module):
    """轻量级 3 层 CNN → 输出动作分布的 μ（均值）与 logσ（对数标准差）"""
    def __init__(self, in_c=256, hid=64):  # in_c=256 适配 channel_proj 输出
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_c, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.ReLU(),
            nn.Conv2d(hid, 2, 1)  # 输出 2 通道：μ（第1通道）和 logσ（第2通道）
        )

    def forward(self, f_map):
        """
        Args:
            f_map: [N, 256, H, W]  经过 channel_proj 映射后的特征图（UNet 专用）
        Returns:
            mu: [N, 1, H, W]  动作分布均值
            log_std: [N, 1, H, W]  动作分布对数标准差（限制范围 [-20, 2]）
        """
        out = self.cnn(f_map)  # [N, 2, H, W]
        mu, log_std = out.chunk(2, dim=1)  # 按通道分割为 μ 和 logσ
        log_std = torch.clamp(log_std, -20, 2)  # 限制 logσ 范围，避免方差过大
        return mu, log_std

class QHead(nn.Module):
    """Q 网络：输入特征+动作，输出动作价值（Q值）"""
    def __init__(self, in_c=256, hid=64):  # in_c=256 适配 channel_proj 输出
        super().__init__()
        self.cnv = nn.Sequential(
            nn.Conv2d(in_c + 1, hid, 3, padding=1),  # 特征（256）+ 动作（1）→ 257 通道
            nn.GroupNorm(8, hid), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GroupNorm(8, hid), nn.ReLU(),
            nn.Conv2d(hid, 1, 1)  # 输出单通道 Q 值
        )

    def forward(self, f_map, alpha):
        """
        Args:
            f_map: [N, 256, H, W]  特征图
            alpha: [N, 1, H, W]  RL 动作（图像增强系数，0~1）
        Returns:
            q_val: [N, 1, H, W]  动作价值
        """
        x = torch.cat([f_map, alpha], dim=1)  # 特征与动作拼接
        return self.cnv(x)

# ---------- RL 智能体：PixelSACAgent（像素级 Soft Actor-Critic） ----------
class PixelSACAgent(nn.Module):
    def __init__(self, feat_dim=256, lr=1e-4, gamma=0.99, tau=0.005, ent_lambda=0.2):
        super().__init__()
        self.gamma = gamma    # 折扣因子（未来奖励权重）
        self.tau = tau        # 软更新系数（目标网络更新幅度）
        self.ent_lambda = ent_lambda  # 熵正则化系数（鼓励探索）
        self.lr = lr          # RL 优化器学习率（降低到 1e-4，避免更新过快）

        # 1. 策略网络（输出动作分布）
        self.policy = MLPHead(in_c=feat_dim)
        # 2. 双 Q 网络（避免过估计）+ 目标 Q 网络（延迟更新）
        self.q1 = QHead(in_c=feat_dim)
        self.q2 = QHead(in_c=feat_dim)
        self.q1_targ = QHead(in_c=feat_dim)
        self.q2_targ = QHead(in_c=feat_dim)
        # 初始化目标 Q 网络参数（与当前 Q 网络一致）
        soft_update(self.q1_targ, self.q1, tau=1.0)
        soft_update(self.q2_targ, self.q2, tau=1.0)

        # 3. 优化器（后续会加入 channel_proj 参数）
        self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.lr
        )

        # 4. 伪计数（visit_count）：初始化为 1（避免除以 0 导致 bonus 爆炸）
        self.register_buffer("visit_count", None)  # 动态初始化尺寸

    def _init_visit_count(self, height, width, device):
        """动态初始化 visit_count（确保尺寸与输入匹配，初始值为 1）"""
        if self.visit_count is None or self.visit_count.shape[-2:] != (height, width):
            self.visit_count = torch.ones(1, 1, height, width, device=device)  # 初始为 1，而非 0
            #print(f"[RL-Visit-Count] Initialized: shape={self.visit_count.shape}, device={device}")

    @torch.no_grad()
    def act(self, f_map):
        """
        生成动作（alpha：图像增强系数，0~1）
        Args:
            f_map: [N, 256, H, W]  特征图
        Returns:
            alpha: [N, 1, H, W]  动作（clamp 到 [0.1, 0.9]，避免极端增强）
        """
        mu, log_std = self.policy(f_map)
        dist = Normal(mu, log_std.exp())  # 正态分布采样
        alpha = torch.sigmoid(dist.rsample())  # 重参数化采样 + sigmoid 映射到 0~1
        alpha = torch.clamp(alpha, 0.1, 0.9)  # 限制动作范围，避免极端增强
        return alpha

    def compute_reward(self, H_aug, alpha):
        """
        计算奖励（修正数值异常）：r = 熵 + 0.5*bonus - 0.1*alpha
        Args:
            H_aug: [N, 1, H, W]  高熵区域熵值
            alpha: [N, 1, H, W]  动作（增强系数）
        Returns:
            reward: [N, 1, H, W]  奖励（clamp 到 [-0.1, 0.5]，避免放大过度）
        """
        device = H_aug.device
        # 初始化 visit_count（确保尺寸与 H_aug 匹配）
        self._init_visit_count(H_aug.shape[2], H_aug.shape[3], device)
        
        # 计算 bonus（限制上限为 0.5，避免初始值过大）
        bonus = 1 / (self.visit_count.sqrt() + 1e-6)
        bonus = torch.clamp(bonus, max=0.5)  # 限制 bonus 最大为 0.5，降低奖励贡献
        
        # 确保所有张量尺寸和设备一致
        if bonus.shape[-2:] != H_aug.shape[-2:]:
            bonus = F.interpolate(bonus, size=H_aug.shape[-2:], mode='bilinear', align_corners=False)
        if alpha.shape[-2:] != H_aug.shape[-2:]:
            alpha = F.interpolate(alpha, size=H_aug.shape[-2:], mode='bilinear', align_corners=False)
        bonus = bonus.to(device)
        alpha = alpha.to(device)
        
        # 计算奖励并限制范围
        reward = H_aug + 0.5 * bonus - 0.1 * alpha
        reward = torch.clamp(reward, -0.1, 0.5)  # 限制奖励范围，避免加权系数失控
        return reward

    def update(self, f_map, alpha, reward, f_next, max_grad_norm=5.0):
        """
        更新 RL 智能体（策略网络 + Q 网络），并添加梯度裁剪
        Args:
            f_map: [N, 256, H, W]  当前状态特征
            alpha: [N, 1, H, W]  当前动作
            reward: [N, 1, H, W]  即时奖励
            f_next: [N, 256, H, W]  下一状态特征
            max_grad_norm: 梯度裁剪阈值（默认 5.0）
        """
        # ---------------- 1. 更新 Q 网络 ----------------
        with torch.no_grad():
            # 采样下一动作
            mu_next, log_std_next = self.policy(f_next)
            dist_next = Normal(mu_next, log_std_next.exp())
            alpha_next = torch.sigmoid(dist_next.rsample())
            alpha_next = torch.clamp(alpha_next, 0.1, 0.9)  # 限制下一动作范围
            
            # 计算目标 Q 值（双 Q 网络取最小，避免过估计）
            q1_t = self.q1_targ(f_next, alpha_next)
            q2_t = self.q2_targ(f_next, alpha_next)
            q_t = torch.min(q1_t, q2_t) - self.ent_lambda * dist_next.log_prob(alpha_next).sum(dim=1, keepdim=True)
            
            # 确保 reward 尺寸与 q_t 一致
            if reward.shape[-2:] != q_t.shape[-2:]:
                reward = F.interpolate(reward, size=q_t.shape[-2:], mode='bilinear', align_corners=False)
            target_q = reward + self.gamma * q_t  # 目标 Q 值 = 即时奖励 + 折扣未来 Q 值

        # 计算当前 Q 预测值与目标值的 MSE 损失
        q1_pred = self.q1(f_map, alpha)
        q2_pred = self.q2(f_map, alpha)
        q_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        # Q 网络反向传播 + 梯度裁剪
        self.q_optim.zero_grad()
        q_loss.backward(retain_graph=True)
        clip_grad_norm(list(self.q1.parameters()) + list(self.q2.parameters()), max_grad_norm)
        self.q_optim.step()

        # ---------------- 2. 更新策略网络 ----------------
        # 采样当前动作（带梯度）
        mu, log_std = self.policy(f_map)
        dist = Normal(mu, log_std.exp())
        alpha_samp = torch.sigmoid(dist.rsample())
        alpha_samp = torch.clamp(alpha_samp, 0.1, 0.9)
        
        # 计算策略损失（最大化 Q 值 - 熵正则化）
        q_new = torch.min(self.q1(f_map, alpha_samp), self.q2(f_map, alpha_samp))
        pi_loss = (self.ent_lambda * dist.log_prob(alpha_samp).sum(dim=1, keepdim=True) - q_new).mean()

        # 策略网络反向传播 + 梯度裁剪
        self.pi_optim.zero_grad()
        pi_loss.backward(retain_graph=True)
        clip_grad_norm(self.policy.parameters(), max_grad_norm)
        self.pi_optim.step()

        # ---------------- 3. 软更新目标 Q 网络 ----------------
        soft_update(self.q1_targ, self.q1, self.tau)
        soft_update(self.q2_targ, self.q2, self.tau)

    def count_visit(self, alpha):
        """更新伪计数（记录动作访问频率，用于探索 bonus 计算）"""
        with torch.no_grad():
            device = alpha.device
            self._init_visit_count(alpha.shape[2], alpha.shape[3], device)
            
            # 确保 alpha 尺寸与 visit_count 一致
            if alpha.shape[-2:] != self.visit_count.shape[-2:]:
                alpha = F.interpolate(alpha, size=self.visit_count.shape[-2:], mode='bilinear', align_corners=False)
            alpha = alpha.to(device)
            
            # 仅统计 alpha > 0.1 的区域（有效探索动作）
            self.visit_count += (alpha > 0.1).sum(dim=0, keepdim=True)

    @torch.no_grad()
    def explore_step(self, img, feat, prob_teacher, entropy):
        """
        生成增强图像 + 计算奖励（降低增强强度，避免预测差异过大）
        Args:
            img: [N, 3, H, W]  原始图像
            feat: [N, 256, H, W]  特征图
            prob_teacher: [N, C, H, W]  教师概率图（UNet 输出）
            entropy: [N, 1, H, W]  熵图
        Returns:
            img_aug: [N, 3, H, W]  增强后图像（强度降低）
            reward: [N, 1, H, W]  奖励
            alpha: [N, 1, H, W]  动作（增强系数）
        """
        # 1. 生成动作（alpha）
        alpha = self.act(feat)  # [N, 1, H, W]
        temp_alpha = alpha  # 保存原始尺寸的 alpha，用于后续更新 visit_count

        # 2. 降低图像增强强度（从 0.3/0.2 → 0.1/0.05，避免过度增强）
        alpha_aug = F.interpolate(alpha, size=img.shape[2:], mode='bilinear', align_corners=False)
        img_aug = torch.clamp(img * (1 + 0.1 * alpha_aug) + 0.05 * alpha_aug, 0, 1)  # 强度降低 60%+

        # 3. 计算奖励
        reward = self.compute_reward(entropy, alpha_aug)

        # 4. 更新伪计数（记录访问频率）
        self.count_visit(temp_alpha)

        return img_aug, reward, temp_alpha

# ---------- RL 损失封装：RLEntropyHighLoss（高熵区域损失 + 一致性损失） ----------
class RLEntropyHighLoss(nn.Module):
    def __init__(self, reward_scale=0.001, consist_weight=0.5):
        super().__init__()
        self.reward_scale = reward_scale  # 奖励放大系数（保持 0.001，配合 reward 限制）
        self.consist_weight = consist_weight  # 增强图一致性损失权重
        self.channel_proj = None  # UNet 专用：C→256 通道映射层

    def init_channel_proj(self, in_c=21, device=torch.device('cuda')):
        """
        提前初始化通道映射层（确保与 UNet 输出类别数匹配）
        Args:
            in_c: 输入通道数（= UNet 输出类别数，如 Pascal=21）
            device: 设备（与模型一致）
        """
        if self.channel_proj is None:
            self.channel_proj = nn.Conv2d(in_c, 256, kernel_size=1).to(device)
            # 初始化权重（kaiming 正态分布，确保初始映射稳定）
            with torch.no_grad():
                nn.init.kaiming_normal_(self.channel_proj.weight, mode='fan_out', nonlinearity='relu')
                if self.channel_proj.bias is not None:
                    nn.init.constant_(self.channel_proj.bias, 0.0)
            print(f"[RL-Channel-Proj] Initialized: in_c={in_c} → out_c=256, device={device}")

    def forward(self,
                pred_student: torch.Tensor,
                prob_teacher: torch.Tensor,
                entropy_t: torch.Tensor,
                images: torch.Tensor,
                model: nn.Module,
                target_model: nn.Module,
                explorer: nn.Module,
                high_thresh: float,
                max_grad_norm=5.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pred_student: [N, C, H, W]  学生模型 logits
            prob_teacher: [N, C, H, W]  教师概率图（UNet 输出，C=类别数）
            entropy_t: [N, 1, H, W]  熵图（高熵区域标识）
            images: [N, 3, H, W]  原始图像
            model: 学生模型（用于提取特征）
            target_model: 目标模型（用于生成增强图预测）
            explorer: RL 智能体（PixelSACAgent）
            high_thresh: 高熵区域阈值（动态计算）
            max_grad_norm: 梯度裁剪阈值（与 RL 智能体一致）
        Returns:
            loss_high: 标量，RL 高熵区域总损失
            reward: [N, 1, H, W]  奖励
            prob_aug: [N, C, H, W]  增强图的预测概率
        """
        # ---------------- 1. 特征提取 + 通道映射（UNet 专用） ----------------
        if hasattr(model, 'backbone'):
            # 有 backbone 的模型（如 DeepLab）：直接用 256 通道特征
            feat = model.backbone(images)
        else:
            # UNet：用概率图作为输入，经 channel_proj 映射到 256 通道
            assert self.channel_proj is not None, "Call init_channel_proj() before forward!"
            feat = self.channel_proj(prob_teacher)  # [N, 256, H, W]

        # ---------------- 2. RL 探索：生成增强图 + 奖励 ----------------
        feat32 = F.interpolate(feat, size=(32, 32), mode='bilinear', align_corners=False)
        img_aug, reward, alpha = explorer.explore_step(images, feat32, prob_teacher, entropy_t)

        # ---------------- 3. RL 智能体更新（含 channel_proj 梯度裁剪） ----------------
        # 计算下一状态特征（f_next）：prob_teacher 下采样后映射到 256 通道
        prob_teacher32 = F.interpolate(prob_teacher, size=(32, 32), mode='bilinear', align_corners=False)
        f_next = self.channel_proj(prob_teacher32)  # [N, 256, 32, 32]（修复通道不匹配）
        
        # 加入 channel_proj 参数到 RL 优化器（确保同步更新）
        if not hasattr(explorer, 'channel_proj'):
            explorer.channel_proj = self.channel_proj  # 绑定到 explorer，便于后续优化
            # 更新策略优化器：加入 channel_proj 参数
            rl_params = list(explorer.policy.parameters()) + list(self.channel_proj.parameters())
            explorer.pi_optim = torch.optim.Adam(rl_params, lr=explorer.lr)

        # 更新 RL 智能体（含 channel_proj 梯度裁剪）
        explorer.update(
            f_map=feat32,
            alpha=alpha,
            reward=reward,
            f_next=f_next.detach(),  # f_next 无梯度，避免影响学生模型
            max_grad_norm=max_grad_norm
        )

        # ---------------- 4. 一致性损失计算（降低基础损失值） ----------------
        # 增强图的教师预测（无梯度）
        with torch.no_grad():
            if pred_student.shape[1] == 1:  # 二分类
                prob_aug = torch.sigmoid(target_model(img_aug))
            else:  # 多分类
                prob_aug = torch.softmax(target_model(img_aug), dim=1)

        # 高熵掩膜（限制覆盖范围，避免损失累积过多）
        high_mask = (entropy_t >= high_thresh).float()
        # 保底逻辑：若高熵区域为 0，仅取 top1% 像素
        if high_mask.sum() == 0:
            k = max(1, int(0.01 * high_mask.numel()))
            idx = torch.topk(entropy_t.view(-1), k).indices
            high_mask.view(-1)[idx] = 1.0

        # 计算一致性损失（主损失 + 增强图损失，权重控制）
        # 1. 学生预测与教师概率的一致性
        consist_main = (F.mse_loss(pred_student, prob_teacher.detach(), reduction='none') * high_mask).mean()
        # 2. 增强图预测与教师概率的一致性（权重降低到 0.3，减少影响）
        consist_aug = (F.mse_loss(prob_aug, prob_teacher.detach(), reduction='none') * high_mask).mean()
        consist_loss = consist_main + 0.3 * consist_aug  # 降低增强图损失权重

        # ---------------- 5. 奖励加权（避免系数失控） ----------------
        # 奖励均值放大（配合 reward 的 clamp，确保系数在 1.0~1.05 之间）
        reward_mean = reward.mean().item()
        loss_high = consist_loss * (1 + reward_mean * self.reward_scale)

        # 打印损失分解（便于调试）
        if self.training and torch.rand(1) < 0.1:  # 10% 概率打印，避免日志冗余
            print(f"[RL-Loss] consist_main={consist_main:.3f}, consist_aug={consist_aug:.3f}, reward_mean={reward_mean:.3f}, loss_high={loss_high:.3f}")

        return loss_high, reward, prob_aug