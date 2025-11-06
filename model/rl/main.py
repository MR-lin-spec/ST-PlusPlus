from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
from torch.optim.lr_scheduler import _LRScheduler
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.backbone.Unetmodel import UNet
import logging, os, datetime
import matplotlib.pyplot as plt
import copy
# ========== 新增：RL 与动态阈值相关导入 ==========
from model.rl.dynamic_threshold import DynamicThreshold
from model.rl.RLmodel import PixelSACAgent, RLEntropyHighLoss
from model.rl.utilss import compute_entropy_map
import torch.nn.functional as F
# =========================

# ========== 新增：RL 与动态阈值全局参数 ==========
RL_UPDATE_EVERY = 50      # RL 每 N 个 batch 更新一次（适配无标签数据）
REWARD_SCALE = 0.001      # 奖励放大系数（来自 RLmodel.py）
CONSIST_W = 0.5           # 一致性损失权重（来自 RLmodel.py）
LOW_P, HIGH_P = 20.0, 80.0# 动态阈值百分位（来自 dynamic_threshold.py）
# =========================

# 1. 日志配置（保留原逻辑，新增 RL 相关日志）
log_dir = 'runs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'),
              logging.StreamHandler()]
)
log = logging.getLogger('ST++')

# 2. 全局画图缓存（保留原逻辑）
PLOT = {'epoch':[], 'train_loss':[], 'val_mIOU':[], 'lr':[], 'rl_loss':[]}  # 新增 RL 损失记录
MODE = None  # 标记训练模式：train/semi_train


class Poly(_LRScheduler):
    """修正版 Poly 学习率调度器（保留原逻辑，确保 80epoch 不提前降为 0）"""
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / self.max_iter) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


class EarlyStopping:
    """早停机制（保留原逻辑，防止过拟合）"""
    def __init__(self, patience=10, delta=0.1):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.stop = False

    def __call__(self, metric):
        if self.best is None or metric > self.best + self.delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True


def parse_args():
    """参数解析（保留原逻辑，确保与 ST++ 框架兼容）"""
    parser = argparse.ArgumentParser(description='ST and ST++ Framework (RL Enhanced)')
    # 基础配置
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101','None'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2','unet'], default='deeplabv3plus')
    # 半监督配置
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    # ST++ 专属配置
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true', help='use ST++')
    parser.add_argument('--unet_weights', type=str, default='/root/ST-PlusPlus/pretrained/unet_medical.pth', help='UNet pretrained path')
    args = parser.parse_args()

    # 补全默认参数（适配数据集）
    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]
        # 确保尺寸为 32 倍数（适配 UNet 下采样）
        if args.crop_size % 32 != 0:
            args.crop_size = (args.crop_size // 32) * 32
            log.info(f"Adjust crop size to {args.crop_size} (multiple of 32) for UNet compatibility")
    return args


def init_basic_elems(args):
    """初始化模型与优化器（保留原逻辑，适配 UNet 无 backbone 情况）"""
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2, 'unet': UNet}

    if args.model == 'unet':
        model = model_zoo[args.model](
            backbone=None,
            num_classes=21 if args.dataset == 'pascal' else 19,
            pretrained_path=args.unet_weights
        )
        # UNet 无 backbone，全参数同学习率
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)
        head_lr_multiple = 10.0
        # DeepLabV2 特殊处理（预训练权重 + 头学习率）
        if args.model == 'deeplabv2':
            assert args.backbone == 'resnet101'
            model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
            head_lr_multiple = 1.0
        # 分层学习率（backbone 低，head 高）
        optimizer = SGD([
            {'params': model.backbone.parameters(), 'lr': args.lr},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr * head_lr_multiple}
        ], lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 多卡并行（cuda 设备）
    model = DataParallel(model).cuda()
    return model, optimizer


def draw_plot():
    """画图函数（新增 RL 损失曲线，保留原 loss/mIOU/lr）"""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))  # 4 个子图：loss/rl_loss/mIOU/lr
    # 1. 总训练损失
    ax[0].plot(PLOT['epoch'], PLOT['train_loss'], marker='o', color='blue')
    ax[0].set_title('Total Train Loss')
    ax[0].set_xlabel('Epoch')
    # 2. RL 高熵损失
    if PLOT['rl_loss']:
        ax[1].plot(PLOT['epoch'], PLOT['rl_loss'], marker='s', color='red')
        ax[1].set_title('RL High-Entropy Loss')
    else:
        ax[1].set_title('RL High-Entropy Loss (Not Activated)')
    ax[1].set_xlabel('Epoch')
    # 3. 验证 mIOU
    ax[2].plot(PLOT['epoch'], PLOT['val_mIOU'], marker='o', color='orange')
    ax[2].set_title('Val mIOU')
    ax[2].set_xlabel('Epoch')
    # 4. 学习率
    ax[3].plot(PLOT['epoch'], PLOT['lr'], marker='o', color='green')
    ax[3].set_title('Learning Rate')
    ax[3].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'curve_rl.png'))
    plt.close()


def select_reliable(models, dataloader, args):
    """
    修正版：融合 mIOU 一致性与熵置信度的可靠数据筛选（借鉴 main1.py 的 select_reliable_RL_mIOU）
    核心思想：
    1. mIOU 反映模型间预测一致性（高一致性 → 可靠）
    2. 动态阈值计算熵的可靠度（低熵比例高 → 可靠）
    3. 加权融合得分，排序后划分可靠/不可靠
    """
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)
    # 所有模型设为评估模式
    for m in models:
        m.eval()

    id_to_score = []
    tbar = tqdm(dataloader, desc='[RL-Select] mIOU+Entropy Reliable Selection')
    # 初始化动态阈值器（来自 dynamic_threshold.py）
    dt = DynamicThreshold(low_percentile=LOW_P, high_percentile=HIGH_P, adaptive=True)

    with torch.no_grad():
        for img, mask, img_id in tbar:
            img = img.cuda()  # 数据移至 GPU
            # ---------------- 1. 计算 mIOU 一致性得分 ----------------
            preds = []
            for model in models:
                pred_logits = model(img)
                preds.append(torch.argmax(pred_logits, dim=1).cpu().numpy())  # 预测类别（CPU 存）
            
            # 以最后一个模型为「Teacher」，计算与第一个模型的 mIOU（简化一致性度量）
            metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
            metric.add_batch(preds[-1], preds[0])
            miou_score = metric.evaluate()[-1]  # 0~1，越高越一致

            # ---------------- 2. 计算熵置信度得分 ----------------
            # 用最后一个模型算概率图（更稳定）
            pred_prob = torch.softmax(models[-1](img), dim=1)  # (N,C,H,W)
            # 计算熵图（适配二分类/多分类）
            if pred_prob.shape[1] == 1:  # 二分类
                ent_map = compute_entropy_map(pred_prob)  # (N,1,H,W)
            else:  # 多分类（香农熵：-sum(p*log2(p))）
                ent_map = -(pred_prob * (pred_prob + 1e-12).log2()).sum(dim=1, keepdim=True)  # (N,1,H,W)
            
            # 动态计算低阈值，低熵比例 = 熵 ≤ 低阈值的像素占比（0~1，越高越可靠）
            low_thresh = dt.compute_low_threshold(ent_map)
            entropy_score = (ent_map <= low_thresh).float().mean().item()

            # ---------------- 3. 加权融合得分 ----------------
            miou_weight = 0.6  # 权重可调，越大越看重一致性
            total_score = miou_weight * miou_score + (1 - miou_weight) * entropy_score
            id_to_score.append((img_id[0], total_score))

    # 按得分降序排序，划分前 50% 为可靠
    id_to_score.sort(key=lambda x: x[1], reverse=True)
    split_idx = len(id_to_score) // 2
    # 保存可靠/不可靠 ID
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for idx in range(split_idx):
            f.write(id_to_score[idx][0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for idx in range(split_idx, len(id_to_score)):
            f.write(id_to_score[idx][0] + '\n')

    log.info(f"[RL-Select] Done | Reliable: {split_idx}, Unreliable: {len(id_to_score)-split_idx}")


def label(model, dataloader, args):
    """
    修正版：生成伪标签 + 熵图（借鉴 main1.py 的 label 函数，修复熵图保存逻辑）
    核心改进：
    1. 同时保存伪标签（.png）和熵图（.npy）
    2. 适配二分类/多分类的熵图计算
    3. 新增 mIOU 实时监控伪标签质量
    """
    model.eval()  # 评估模式
    tbar = tqdm(dataloader, desc='[RL-Label] Pseudo Mask + Entropy Map Generation')
    num_classes = 21 if args.dataset == 'pascal' else 19
    metric = meanIOU(num_classes=num_classes)  # 监控伪标签与真实标签的 mIOU
    cmap = color_map(args.dataset)  # 伪标签调色板

    with torch.no_grad():
        for img, real_mask, img_id in tbar:
            img = img.cuda()
            # 1. 模型预测（logits → 类别）
            pred_logits = model(img)  # (N,C,H,W)
            pred_cls = torch.argmax(pred_logits, dim=1).cpu()  # (N,H,W)，CPU 用于保存

            # 2. 计算伪标签质量（mIOU）
            metric.add_batch(pred_cls.numpy(), real_mask.numpy())
            current_miou = metric.evaluate()[-1] * 100  # 转百分比

            # 3. 保存伪标签（.png）
            mask_basename = os.path.basename(img_id[0].split(' ')[1])  # 提取原始 mask 文件名
            pred_pil = Image.fromarray(pred_cls.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred_pil.putpalette(cmap)
            pred_pil.save(os.path.join(args.pseudo_mask_path, mask_basename))

            # 4. 计算并保存熵图（.npy）
            if pred_logits.shape[1] == 1:  # 二分类：sigmoid 后算熵
                pred_prob = torch.sigmoid(pred_logits)  # (N,1,H,W)
                ent_map = compute_entropy_map(pred_prob)  # (N,1,H,W)
            else:  # 多分类：softmax 后算香农熵
                pred_prob = torch.softmax(pred_logits, dim=1)  # (N,C,H,W)
                ent_map = -(pred_prob * (pred_prob + 1e-12).log2()).sum(dim=1, keepdim=True)  # (N,1,H,W)
            
            # 熵图保存（与伪标签同目录，后缀替换为 .npy）
            ent_save_path = os.path.join(args.pseudo_mask_path, mask_basename.replace('.png', '.npy'))
            np.save(ent_save_path, ent_map.squeeze(0).cpu().numpy())  # 去除 batch 维，CPU 保存

            # 更新进度条
            tbar.set_description(f'[RL-Label] mIOU: {current_miou:.2f}% | Saved: {mask_basename}')

    log.info(f"[RL-Label] Done | Pseudo masks + entropy maps saved to {args.pseudo_mask_path}")


def init_rl_components(model, args, device):

    """
    独立的 RL 组件初始化函数（非嵌套，供 train 函数调用）
    功能：初始化 PixelSACAgent + RLEntropyHighLoss（含 UNet 专用 channel_proj）
    Args:
        model: 当前训练的模型（如 UNet/DeepLab）
        args: 全局配置参数（含 model 类型、数据集等）
        device: 模型运行设备（cuda/cpu）
    Returns:
        rl_agent: 初始化后的 PixelSAC 智能体
        rl_loss_fn: 初始化后的 RL 高熵损失函数（含 channel_proj）
    """
    from model.rl.RLmodel import PixelSACAgent, RLEntropyHighLoss
    import torch

    # 1. 获取当前任务的类别数（UNet 从 final 层取，其他模型从属性取）
    if hasattr(model, 'module'):  # 多卡训练（DataParallel 包装）
        if args.model == 'unet':
            num_classes = model.module.head.final.out_channels
        else:
            num_classes = model.module.num_classes
    else:  # 单卡训练
        if args.model == 'unet':
            num_classes = model.head.final.out_channels
        else:
            num_classes = model.num_classes
    print(f"[RL-Init] Task num_classes: {num_classes} (device: {device})")

    # 2. 初始化 PixelSAC 智能体（feat_dim=256 适配 channel_proj 输出）
    rl_agent = PixelSACAgent(
        feat_dim=256, lr=3e-4, gamma=0.99, tau=0.005, ent_lambda=0.2
    ).to(device)

    # 3. 初始化 RL 高熵损失函数 + 提前创建 channel_proj（解决 UNet 通道不匹配）
    rl_loss_fn = RLEntropyHighLoss(
        reward_scale=REWARD_SCALE, consist_weight=CONSIST_W
    ).to(device)
    # 提前初始化 channel_proj（输入=num_classes，输出=256，适配 RL 智能体）
    rl_loss_fn.init_channel_proj(in_c=num_classes, device=device)

    # 4. 组合 RL 优化器参数（智能体参数 + channel_proj 参数，确保两者都能训练）
    rl_params = list(rl_agent.parameters()) + list(rl_loss_fn.channel_proj.parameters())
    # 覆盖 RL 智能体的优化器（替换默认优化器，包含 channel_proj）
    rl_agent.pi_optim = torch.optim.Adam(rl_params, lr=3e-4)
    rl_agent.q_optim = torch.optim.Adam(
        list(rl_agent.q1.parameters()) + list(rl_agent.q2.parameters()),
        lr=3e-4
    )

    print("[RL-Init] PixelSACAgent + RLEntropyHighLoss initialized (含 UNet channel_proj)")
    return rl_agent, rl_loss_fn

def train(model, trainloader, valloader, criterion, optimizer, args):
    """
    完整训练函数（非嵌套调用 RL 初始化，适配 ST++ + UNet + RL 流程）
    核心功能：
    1. 基础分类损失（CrossEntropy）+ RL 高熵探索损失（仅 semi_train 模式）
    2. 修正版 Poly 调度器（防 LR 归零）+ 早停机制（防过拟合）
    3. 最佳模型保存（自动删除旧模型）+ 实时日志 + 训练曲线绘制
    4. 独立调用 init_rl_components，避免嵌套依赖
    """
    # 导入必要模块（确保函数内依赖完整）
    import torch
    import torch.nn.functional as F
    import copy
    from torch.optim.lr_scheduler import _LRScheduler
    from model.rl.dynamic_threshold import DynamicThreshold
    from utils import meanIOU
    from tqdm import tqdm
    import os

    # ---------------- 1. 全局变量与核心组件初始化 ----------------
    global MODE, PLOT, log  # 引用全局变量（日志/画图缓存/训练模式）
    total_iters = len(trainloader) * args.epochs  # 总迭代次数
    device = next(model.parameters()).device  # 自动获取模型设备（cuda/cpu）
    rl_agent = None  # RL 智能体（初始为 None，需触发条件才初始化）
    rl_loss_fn = None  # RL 损失函数（初始为 None）

    # 1.1 修正版 Poly 学习率调度器（min_lr=1e-6 防止 LR 降为 0）
    class Poly(_LRScheduler):
        def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1, min_lr=1e-6):
            self.max_iter = max_iter
            self.power = power
            self.min_lr = min_lr
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            factor = max(1 - self.last_epoch / self.max_iter, 1e-8) ** self.power
            return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]

    scheduler = Poly(optimizer, max_iter=total_iters, power=0.9, min_lr=1e-6)

    # 1.2 早停机制（连续 10 个 epoch mIOU 无提升则停止）
    class EarlyStopping:
        def __init__(self, patience=10, delta=0.1):
            self.patience = patience
            self.delta = delta
            self.best = None
            self.counter = 0
            self.stop = False

        def __call__(self, metric):
            if self.best is None or metric > self.best + self.delta:
                self.best = metric
                self.counter = 0
            else:
                self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    early_stopper = EarlyStopping(patience=10, delta=0.1)
    previous_best_miou = 0.0  # 历史最佳验证 mIOU
    best_model = copy.deepcopy(model)  # 保存最佳模型副本

    # 打印训练启动信息
    log.info(
        f"[RL-Train] 启动训练 | "
        f"模式: {MODE} | "
        f"总迭代: {total_iters} | "
        f"设备: {device} | "
        f"模型: {args.model} | "
        f"数据集: {args.dataset}"
    )

    # ---------------- 2. 训练循环 ----------------
    for epoch in range(args.epochs):
        model.train()  # 模型设为训练模式
        total_train_loss = 0.0  # 总损失（基础损失 + RL 损失）
        total_rl_loss = 0.0     # RL 损失单独统计
        # 训练进度条（显示 epoch、总损失、RL 损失）
        train_tbar = tqdm(
            trainloader,
            desc=f"[Epoch {epoch:03d}] 训练 | 总损失: ? | RL 损失: ?"
        )

        for batch_idx, (img, mask) in enumerate(train_tbar):
            # 2.1 数据预处理（移至模型设备，确保类型匹配）
            img = img.to(device, non_blocking=True)  # non_blocking 加速数据传输
            mask = mask.to(device, non_blocking=True).long()  # 标签转长整型

            # 2.2 基础分类损失计算（交叉熵，忽略 255 无效标签）
            pred_student = model(img)  # 学生模型输出（logits）
            base_loss = criterion(pred_student, mask)

            # 2.3 RL 高熵损失计算（仅 semi_train 模式 + 有未标注数据时启用）
            rl_loss = torch.tensor(0.0, device=device)  # 默认为 0（不启用时无影响）
            if MODE == 'semi_train' and hasattr(trainloader.dataset, 'unlabeled_ids'):
                # 首次调用时初始化 RL 组件（仅初始化一次，避免重复建图）
                if rl_agent is None or rl_loss_fn is None:
                    rl_agent, rl_loss_fn = init_rl_components(model, args, device)

                # 2.3.1 计算学生概率图 + 熵图（适配二分类/多分类）
                with torch.no_grad():  # 无梯度计算，加速
                    if pred_student.shape[1] == 1:  # 二分类（如医学分割）
                        prob_student = torch.sigmoid(pred_student)
                        # 二分类熵计算：-p*log2(p) - (1-p)*log2(1-p)
                        ent_map = -(
                            prob_student * (prob_student + 1e-12).log2() +
                            (1 - prob_student) * (1 - prob_student + 1e-12).log2()
                        )
                    else:  # 多分类（如 Pascal/Cityscapes）
                        prob_student = torch.softmax(pred_student, dim=1)
                        # 多分类熵计算：-sum(p*log2(p))（逐像素）
                        ent_map = -(prob_student * (prob_student + 1e-12).log2()).sum(dim=1, keepdim=True)

                    # 动态计算高熵阈值（来自 dynamic_threshold.py，兜底不小于 0.7）
                    dt = DynamicThreshold(high_percentile=HIGH_P, adaptive=True)
                    high_thresh = dt.compute_high_threshold(ent_map)

                # 2.3.2 调用 RL 损失函数计算高熵区域损失
                rl_loss, _, _ = rl_loss_fn(
                    pred_student=pred_student,
                    prob_teacher=prob_student.detach(),  # 自蒸馏：学生概率作为 teacher
                    entropy_t=ent_map,
                    images=img,
                    model=model,
                    target_model=model,
                    explorer=rl_agent,
                    high_thresh=high_thresh
                )

            # 2.4 总损失 = 基础损失 + RL 损失（权重可调整，此处为 1:1）
            total_loss = base_loss + rl_loss

            # 2.5 反向传播与参数更新
            optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none 更高效）
            total_loss.backward()                # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 梯度裁剪防爆炸
            optimizer.step()                     # 更新模型参数
            scheduler.step()                     # 更新学习率（一步一调）

            # 2.6 损失统计与进度条更新
            total_train_loss += total_loss.item()
            total_rl_loss += rl_loss.item()
            avg_train_loss = total_train_loss / (batch_idx + 1)
            avg_rl_loss = total_rl_loss / (batch_idx + 1)
            train_tbar.set_description(
                f"[Epoch {epoch:03d}] 训练 | 总损失: {avg_train_loss:.3f} | RL 损失: {avg_rl_loss:.3f}"
            )

        # ---------------- 3. 验证阶段（计算 val mIOU）----------------
        model.eval()  # 模型设为评估模式
        val_metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        val_tbar = tqdm(valloader, desc=f"[Epoch {epoch:03d}] 验证 | mIOU: ?")

        with torch.no_grad():  # 关闭梯度，加速验证
            for img, mask, _ in val_tbar:
                img = img.to(device, non_blocking=True)
                pred = model(img)
                pred_cls = torch.argmax(pred, dim=1)  # 取概率最大的类别
                # 累积混淆矩阵（用于计算 mIOU）
                val_metric.add_batch(pred_cls.cpu().numpy(), mask.numpy())
                # 实时更新验证进度条（显示当前 mIOU）
                current_miou = val_metric.evaluate()[-1] * 100
                val_tbar.set_description(f"[Epoch {epoch:03d}] 验证 | mIOU: {current_miou:.2f}%")

        # 计算最终验证 mIOU（转百分比，便于阅读）
        val_miou = val_metric.evaluate()[-1] * 100

        # ---------------- 4. 日志记录与训练曲线绘制 ----------------
        current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        # 写入日志（包含 epoch、LR、损失、mIOU，便于回溯）
        log.info(
            f"[Epoch {epoch:03d}] 结果 | "
            f"LR: {current_lr:.6f} | "
            f"训练总损失: {avg_train_loss:.4f} | "
            f"RL 损失: {avg_rl_loss:.4f} | "
            f"验证 mIOU: {val_miou:.2f}%"
        )

        # 更新全局画图缓存（PLOT 用于 draw_plot 函数绘制曲线）
        PLOT['epoch'].append(epoch)
        PLOT['train_loss'].append(avg_train_loss)
        PLOT['rl_loss'].append(avg_rl_loss)
        PLOT['val_mIOU'].append(val_miou)
        PLOT['lr'].append(current_lr)

        # 绘制训练曲线（loss/rl_loss/mIOU/lr，保存至 runs/curve_rl.png）
        draw_plot()

        # ---------------- 5. 早停机制与最佳模型保存 ----------------
        # 检查早停条件（连续 patience 个 epoch 无提升则停止训练）
        early_stopper(val_miou)
        if early_stopper.stop:
            log.info(f"[RL-Train] 早停触发 | Epoch: {epoch:03d} | 最佳 mIOU: {early_stopper.best:.2f}%")
            break

        # 检查是否更新最佳模型（当前 mIOU 高于历史最佳）
        if val_miou > previous_best_miou:
            # 删除旧的最佳模型（避免磁盘冗余）
            if previous_best_miou != 0:
                old_model_path = os.path.join(
                    args.save_path,
                    f"{args.model}_{args.backbone}_{previous_best_miou:.2f}.pth"
                )
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)
                    log.info(f"[模型保存] 删除旧最佳模型: {os.path.basename(old_model_path)}")

            # 保存新的最佳模型（仅保存 model.module 权重，适配多卡训练）
            previous_best_miou = val_miou
            new_model_path = os.path.join(
                args.save_path,
                f"{args.model}_{args.backbone}_{val_miou:.2f}.pth"
            )
            torch.save(model.module.state_dict(), new_model_path)
            best_model = copy.deepcopy(model)  # 更新最佳模型副本
            log.info(
                f"[模型保存] 最佳模型更新 | "
                f"mIOU: {val_miou:.2f}% | "
                f"保存路径: {os.path.basename(new_model_path)}"
            )

    # ---------------- 6. 训练结束，返回最佳模型 ----------------
    log.info(
        f"[RL-Train] 训练完成 | "
        f"总 epoch: {epoch+1} | "
        f"最佳验证 mIOU: {previous_best_miou:.2f}% | "
        f"最佳模型路径: {os.path.join(args.save_path, f'{args.model}_{args.backbone}_{previous_best_miou:.2f}.pth')}"
    )
    return best_model

def main(args):
    """主函数（保留 ST++ 核心流程，调用修改后的 select_reliable/label/train）"""
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)
    if args.plus and args.reliable_id_path is None:
        log.error("[ST++ Error] --reliable-id-path is required for ST++")
        exit(1)

    # 初始化验证集
    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(
        valset,
        batch_size=4 if args.dataset == 'cityscapes' else 1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    base_criterion = CrossEntropyLoss(ignore_index=255)  # 基础分类损失

    # ---------------- 阶段 1：有标签数据监督训练（SupOnly）----------------
    log.info(f"\n[ST++ Stage 1/{'6' if args.plus else '3'}] Supervised Training (Labeled Data Only)")
    global MODE
    MODE = 'train'
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    # 数据增强：若有标签数据过少，重复两次
    if len(trainset.ids) < 200:
        trainset.ids = 2 * trainset.ids
        log.info(f"[Data-Aug] Labeled data duplicated (len < 200) | New len: {len(trainset.ids)}")
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True
    )
    # 初始化模型与优化器
    model, optimizer = init_basic_elems(args)
    log.info(f"[Model-Info] Params: {count_params(model):.1f}M | Model: {args.model} | Backbone: {args.backbone}")
    # 训练
    best_model = train(model, trainloader, valloader, base_criterion, optimizer, args)

    # ---------------- 非 ST++ 模式（ST 基础流程）----------------
    if not args.plus:
        # 阶段 2：给所有无标签数据生成伪标签
        log.info(f"\n[ST Stage 2/3] Pseudo Labeling (All Unlabeled Data)")
        unlabeled_dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            drop_last=False
        )
        label(best_model, unlabeled_dataloader, args)

        # 阶段 3：有标签 + 无标签数据重训练（semi_train 模式，启用 RL）
        log.info(f"\n[ST Stage 3/3] Semi-Supervised Retraining (Labeled + Unlabeled)")
        MODE = 'semi_train'
        semi_trainset = SemiDataset(
            args.dataset, args.data_root, MODE, args.crop_size,
            args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path
        )
        semi_trainloader = DataLoader(
            semi_trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16,
            drop_last=True
        )
        # 重新初始化模型（避免继承之前的梯度）
        model, optimizer = init_basic_elems(args)
        train(model, semi_trainloader, valloader, base_criterion, optimizer, args)
        log.info("[ST] Training Completed!")
        return

    # ---------------- ST++ 模式（选择性重训练，核心流程）----------------
    # 阶段 2：筛选可靠无标签数据
    log.info(f"\n[ST++ Stage 2/6] Reliable Data Selection (mIOU + Entropy)")
    reliable_dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    reliable_dataloader = DataLoader(
        reliable_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    select_reliable([best_model], reliable_dataloader, args)  # 用阶段 1 的最优模型筛选

    # 阶段 3：给可靠无标签数据生成伪标签
    log.info(f"\n[ST++ Stage 3/6] Pseudo Labeling (Reliable Unlabeled Data)")
    reliable_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    reliable_label_dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, reliable_id_path)
    reliable_label_dataloader = DataLoader(
        reliable_label_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    label(best_model, reliable_label_dataloader, args)

    # 阶段 4：第一阶段重训练（有标签 + 可靠无标签）
    log.info(f"\n[ST++ Stage 4/6] 1st Retraining (Labeled + Reliable Unlabeled)")
    MODE = 'semi_train'
    stage4_trainset = SemiDataset(
        args.dataset, args.data_root, MODE, args.crop_size,
        args.labeled_id_path, reliable_id_path, args.pseudo_mask_path
    )
    stage4_trainloader = DataLoader(
        stage4_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True
    )
    model, optimizer = init_basic_elems(args)
    best_model_stage4 = train(model, stage4_trainloader, valloader, base_criterion, optimizer, args)

    # 阶段 5：给不可靠无标签数据生成伪标签（用阶段 4 的最优模型）
    log.info(f"\n[ST++ Stage 5/6] Pseudo Labeling (Unreliable Unlabeled Data)")
    unreliable_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    unreliable_label_dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, unreliable_id_path)
    unreliable_label_dataloader = DataLoader(
        unreliable_label_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )
    label(best_model_stage4, unreliable_label_dataloader, args)

    # 阶段 6：第二阶段重训练（有标签 + 所有无标签）
    log.info(f"\n[ST++ Stage 6/6] 2nd Retraining (Labeled + All Unlabeled)")
    stage6_trainset = SemiDataset(
        args.dataset, args.data_root, MODE, args.crop_size,
        args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path
    )
    stage6_trainloader = DataLoader(
        stage6_trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        drop_last=True
    )
    model, optimizer = init_basic_elems(args)
    train(model, stage6_trainloader, valloader, base_criterion, optimizer, args)
    log.info("[ST++] Training Completed!")


if __name__ == '__main__':
    args = parse_args()
    log.info(f"[Config] {args}")
    main(args)