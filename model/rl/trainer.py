# RL_demo_train.py
import os
import time
import logging
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.cuda import max_memory_allocated
from utils.RLmodel import RLEntropyHighLoss
# 新增：导入动态阈值类
from .dynamic_threshold import DynamicThreshold  # 根据实际路径调整


def RL_demo_train(model, target_model, trainloader,
                  loss_Diloss, compute_entropy_map,
                  PixelSACAgent, SegL,
                  max_epoch=100, log_dir='./logs',
                  RLtrain=RLEntropyHighLoss,
                  device=None,
                  use_ema=False,
                  num_classes=1,
                  lr=1e-4,
                  # 新增：动态阈值相关参数（默认使用动态阈值）
                  use_dynamic_threshold=True,
                  low_percentile=20.0,
                  high_percentile=80.0,
                  min_low=0.1,
                  max_high=0.9,
                  # 兼容原有固定阈值参数（动态模式下会被覆盖）
                  low_ent_thresh=0.4,
                  high_ent_thresh=0.4,
                  reward_scale=0.001,
                  consist_weight=0.5,
                  ema_decay=0.999,
                  random_seed=None,
                  feat_dim=1):
    """
    封装完整的 SFDA + 高熵 RL 训练流程。
    新增参数说明：
    use_dynamic_threshold: 是否启用动态阈值
    low_percentile: 低熵阈值百分位（动态模式）
    high_percentile: 高熵阈值百分位（动态模式）
    min_low: 低熵阈值下限
    max_high: 高熵阈值上限
    """
    # ------- 0. 初始化 -------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d-%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w'),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    model = model.to(device).eval()
    target_model = target_model.to(device).train()

    # 新增：初始化动态阈值计算器
    dynamic_thresh = DynamicThreshold(
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        min_low=min_low,
        max_high=max_high,
        adaptive=use_dynamic_threshold
    )

    # 其余初始化代码不变...
    if use_ema:
        ema_model = type(model)().to(device)
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
    else:
        ema_model = None

    seg_low = SegL(num_classes=num_classes, lambd=0.5)
    explorer = PixelSACAgent(feat_dim=feat_dim).to(device)
    
    optimizer = torch.optim.Adam(target_model.parameters(), 
                               lr=lr,
                               betas=(0.9, 0.999), 
                               weight_decay=0.0005)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
    rl_high_loss = RLtrain(
        # 注意：高阈值将在forward中动态传入，这里仅保留基础参数
        reward_scale=reward_scale,
        consist_weight=consist_weight
    )
    
    if random_seed is not None:
        # 随机种子代码不变...
        seed = random_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info('Random seed for this experiment is {} !'.format(seed))


    # ------- 1. 训练循环 -------
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        running_low = running_high = running_total = running_reward = 0.0
        running_low_thresh = running_high_thresh = 0.0  # 新增：记录阈值
        batch_count = 0

        for images, masks in trainloader:
            images = images.to(device, non_blocking=True)

            # ---- 1.1 教师输出 ----
            with torch.no_grad():
                if use_ema and ema_model is not None:
                    pred_teacher = ema_model(images)
                else:
                    pred_teacher = model(images)
                    
                if num_classes > 1:
                    prob_teacher = F.softmax(pred_teacher, dim=1)
                else:
                    prob_teacher = torch.sigmoid(pred_teacher)
                entropy_t = compute_entropy_map(prob_teacher)  # 熵图计算

            # 新增：计算动态阈值（每个batch更新一次）
            current_low_thresh = dynamic_thresh.compute_low_threshold(entropy_t) if use_dynamic_threshold else low_ent_thresh
            current_high_thresh = dynamic_thresh.compute_high_threshold(entropy_t) if use_dynamic_threshold else high_ent_thresh
            # 计算阈值后立即打印
             # 打印阈值计算详情（用于调试）
            if batch_count % 10 == 0:  # 每10个batch打印一次
                logger.info(f"Batch {batch_count} 熵图统计: "
                            f"min={entropy_t.min().item():.4f}, "
                            f"{low_percentile}%分位={np.percentile(entropy_t.detach().cpu().numpy(), low_percentile):.4f}, "
                            f"{high_percentile}%分位={np.percentile(entropy_t.detach().cpu().numpy(), high_percentile):.4f}, "
                            f"max={entropy_t.max().item():.4f}")
                logger.info(f"计算得到的阈值: low={current_low_thresh:.4f}, high={current_high_thresh:.4f}")
            # ---- 1.2 学生输出 ----
            pred_student = target_model(images)

            # ====== ① 低熵分支 ======
            loss_low = loss_Diloss(
                pred_t=pred_student,
                target_img_prob=prob_teacher.detach(),
                entropy_t=entropy_t,
                low_thresh=current_low_thresh,  # 使用动态低阈值
                SegL=seg_low,
                lambd=0.3
            )

            # ====== ② 高熵分支 ======
            loss_high, reward, prob_aug = rl_high_loss(
                pred_student, 
                prob_teacher, 
                entropy_t,
                images, 
                model, 
                target_model, 
                explorer,
                high_thresh=current_high_thresh  # 新增：传入动态高阈值
            )

            # ====== ③ 统一损失 ======
            total_loss = loss_high + loss_low
            
            # 优化步骤不变...
            optimizer.zero_grad()
            grad_scaler.scale(total_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
                
            if use_ema and ema_model is not None:
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(ema_decay).add_(param.data*(1 - ema_decay))

            # ===== 指标记录 =====
            running_low += loss_low.item()
            running_high += loss_high.item()
            running_total += total_loss.item()
            running_reward += reward.mean().item()
            running_low_thresh += current_low_thresh  # 记录阈值
            running_high_thresh += current_high_thresh
            batch_count += 1

        # ------- 2. 日志 -------
        lr_val = optimizer.param_groups[0]['lr']
        mem = max_memory_allocated(device) / 1e9
        logger.info(f"Epoch {epoch:03d}  "
                    f"loss_low: {running_low/batch_count:.4f}  "
                    f"loss_high: {running_high/batch_count:.4f}  "
                    f"total: {running_total/batch_count:.4f}  "
                    f"avg_reward: {running_reward/batch_count:.4f}  "
                    f"low_thresh: {running_low_thresh/batch_count:.4f}  "  # 新增：输出平均阈值
                    f"high_thresh: {running_high_thresh/batch_count:.4f}  "
                    f"lr: {lr_val:.2e}  "
                    f"mem: {mem:.2f} GB")

        # ------- 3. 保存 ckpt -------
        if (epoch + 1) % 10 == 0:
            save_dict = {
                'epoch': epoch,
                'target_model': target_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sac': explorer.state_dict(),
            }
            if use_ema and ema_model is not None:
                save_dict['ema_model'] = ema_model.state_dict()
                
            torch.save(save_dict, os.path.join(log_dir, f'ckpt_epoch{epoch+1}.pth'))

    logger.info('RL_demo_train finished!')