#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST++ 半监督语义分割主入口（含 RL 高熵探索分支）
用法示例：
    # 原始 ST++
    python main.py --data-root /data/PASCAL --dataset pascal \
                   --labeled-id-path dataset/splits/pascal/labeled.txt \
                   --unlabeled-id-path dataset/splits/pascal/unlabeled.txt \
                   --pseudo-mask-path out/pseudo --save-path out/weights

    # 新增 RL 分支
    python main.py ... --algorithm st++_rl
"""
import argparse
import copy
import datetime
import logging
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# =================  自定义包  =================
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

# ----------  可选 RL  ----------
try:
    from model.rl.dynamic_threshold import DynamicThreshold
    from model.rl.RLmodel import PixelSACAgent, RLEntropyHighLoss
    from model.rl.utilss import compute_entropy_map
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


# =================  全局变量  =================
MODE = None          # 当前训练模式：train / semi_train
GLOBAL_ITERS = 0     # 全局迭代计数（给 tensorboard 用）


# =================  参数解析  =================
def parse_args():
    parser = argparse.ArgumentParser(description='ST++ 半监督框架（支持 RL）')

    # 基础配置
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'],
                        default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None,
                        help='初始学习率，None 则按数据集默认')
    parser.add_argument('--epochs', type=int, default=None,
                        help='总 epoch，None 则按数据集默认')
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str,
                        choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str,
                        choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # 半监督必备路径
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)

    # ST++ 特有
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', action='store_true',
                        help='是否启用 ST++ 两段重训练')

    # ********  新增算法选择  ********
    parser.add_argument('--algorithm', type=str,
                        choices=['st++', 'st++_rl'], default='st++',
                        help='st++: 原流程；st++_rl: 引入 SAC 高熵探索')

    args = parser.parse_args()
    return args


# =================  日志 & TensorBoard 初始化  =================
def init_logger_and_tb():
    os.makedirs('logs', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    log_file = os.path.join(
        'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger('ST++')
    tb_writer = SummaryWriter(
        log_dir=os.path.join('runs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    return logger, tb_writer


# =================  模型 & 优化器初始化  =================
def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus,
                 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone,
                                  21 if args.dataset == 'pascal' else 19)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        # 兼容加载：形状不匹配则自动跳过
    pretrained = torch.load('pretrained/resnet101.pth',
                                map_location='cpu')
    model_dict = model.state_dict()
    filtered = {k: v for k, v in pretrained.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    head_lr_multiple = 1.0
    # 分层学习率
    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': args.lr},
        {'params': [p for n, p in model.named_parameters()
                    if 'backbone' not in n],
         'lr': args.lr * head_lr_multiple}],
        momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()
    return model, optimizer


# =================  RL 组件延迟初始化  =================
def init_rl_components(model, args, device):
    """
    仅当用户指定 --algorithm st++_rl 且进入 semi_train 阶段时调用
    返回 (rl_agent, rl_loss_fn)
    """
    if not RL_AVAILABLE:
        raise RuntimeError('RL 包未安装，无法使用 st++_rl 算法')
    num_classes = 21 if args.dataset == 'pascal' else 19
    rl_agent = PixelSACAgent(feat_dim=256, lr=3e-4, gamma=0.99,
                             tau=0.005, ent_lambda=0.2).to(device)
    rl_loss_fn = RLEntropyHighLoss(reward_scale=0.001,
                                   consist_weight=0.5).to(device)
    rl_loss_fn.init_channel_proj(in_c=num_classes, device=device)
    # 把 channel_proj 参数也交给 RL 优化器
    rl_params = list(rl_agent.parameters()) + \
                list(rl_loss_fn.channel_proj.parameters())
    rl_agent.pi_optim = torch.optim.Adam(rl_params, lr=3e-4)
    return rl_agent, rl_loss_fn


# =================  训练函数  =================
def train(model, trainloader, valloader, criterion, optimizer, args, logger, tb_writer):
    global GLOBAL_ITERS
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    best_model = copy.deepcopy(model)

    # 早停
    patience, best_metric = 10, 0
    for epoch in range(args.epochs):
        # ----------  训练  ----------
        model.train()
        epoch_loss, epoch_rl_loss = 0.0, 0.0
        train_bar = tqdm(trainloader, desc=f'Epoch[{epoch}]')
        for img, mask in train_bar:
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            base_loss = criterion(pred, mask)

            # ********  RL 损失注入  ********
            rl_loss = torch.tensor(0.0, device=img.device)
            if args.algorithm == 'st++_rl' and MODE == 'semi_train':
                # 延迟初始化
                if 'rl_agent' not in locals():
                    rl_agent, rl_loss_fn = init_rl_components(
                        model, args, img.device)
                # 计算熵图、动态阈值 ……（与 main-unet.py 完全一致，此处简化）
                with torch.no_grad():
                    prob = torch.softmax(pred, dim=1)
                    ent_map = -(prob * (prob + 1e-12).log2()).sum(dim=1, keepdim=True)
                    dt = DynamicThreshold(high_percentile=80, adaptive=True)
                    high_thresh = dt.compute_high_threshold(ent_map)
                rl_loss, _, _ = rl_loss_fn(
                    pred_student=pred,
                    prob_teacher=prob.detach(),
                    entropy_t=ent_map,
                    images=img,
                    model=model,
                    target_model=model,
                    explorer=rl_agent,
                    high_thresh=high_thresh)
                epoch_rl_loss += rl_loss.item()

            total_loss = base_loss + rl_loss
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

            GLOBAL_ITERS += 1
            epoch_loss += base_loss.item()
            lr = args.lr * (1 - GLOBAL_ITERS / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10 \
                if args.model != 'deeplabv2' else lr

            train_bar.set_postfix(
                Loss=epoch_loss / (train_bar.n + 1),
                RLLoss=epoch_rl_loss / (train_bar.n + 1))

            # tensorboard
            tb_writer.add_scalar('loss/base', base_loss, GLOBAL_ITERS)
            tb_writer.add_scalar('loss/rl', rl_loss, GLOBAL_ITERS)
            tb_writer.add_scalar('lr', lr, GLOBAL_ITERS)

        # ----------  验证  ----------
        model.eval()
        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        with torch.no_grad():
            for img, mask, _ in tqdm(valloader, desc='Val'):
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
        mIOU = metric.evaluate()[-1] * 100
        logger.info(f'Epoch[{epoch}]  mIOU={mIOU:.2f}%')

        tb_writer.add_scalar('mIOU/val', mIOU, epoch)

        # 最佳模型
        if mIOU > previous_best:
            if previous_best != 0:
                old = os.path.join(
                    args.save_path,
                    f'{args.model}_{args.backbone}_{previous_best:.2f}.pth')
                if os.path.exists(old):
                    os.remove(old)
            previous_best = mIOU
            save_path = os.path.join(
                args.save_path,
                f'{args.model}_{args.backbone}_{mIOU:.2f}.pth')
            torch.save(model.module.state_dict(), save_path)
            best_model = copy.deepcopy(model)
            logger.info(f'保存最佳模型 → {save_path}')

        # 早停
        if mIOU > best_metric + 0.1:
            best_metric, patience = mIOU, 10
        else:
            patience -= 1
            if patience == 0:
                logger.info('早停触发')
                break

    return best_model


# =================  伪标签生成  =================
def label_past(model, dataloader, args, logger):
    model.eval()
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)

    for img, mask, id in tqdm(dataloader, desc='Pseudo-Label'):
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()

        metric.add_batch(pred.numpy(), mask.numpy())
        fname = os.path.basename(id[0].split(' ')[1])
        Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P') \
            .putpalette(cmap) \
            .save(os.path.join(args.pseudo_mask_path, fname))

    logger.info(f'伪标签完成  mIOU={metric.evaluate()[-1]*100:.2f}%')


def label(model, dataloader, args, logger):
    model.eval()
    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)

    for img, mask, id in tqdm(dataloader, desc='Pseudo-Label'):
        img = img.cuda()
        with torch.no_grad():
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()

        metric.add_batch(pred.numpy(), mask.numpy())
        
        # 处理两种格式的ID
        if ' ' in id[0]:
            # 旧格式：包含空格分隔的图像和标签路径
            fname = os.path.basename(id[0].split(' ')[1])
        else:
            # 新格式：只有文件名
            if args.dataset == 'pascal':
                fname = f'{id[0]}.png'
            else:  # cityscapes
                fname = f'{id[0]}_gtFine_labelIds.png'
                
        Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P') \
            .putpalette(cmap) \
            .save(os.path.join(args.pseudo_mask_path, fname))

    logger.info(f'伪标签完成  mIOU={metric.evaluate()[-1]*100:.2f}%')

# =================  可靠/不可靠划分  =================
def select_reliable_past(models, dataloader, args, logger):
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for m in models:
        m.eval()

    id_to_score = []
    for img, mask, id in tqdm(dataloader, desc='Select-Reliable'):
        img = img.cuda()
        with torch.no_grad():
            preds = [torch.argmax(m(img), dim=1).cpu().numpy() for m in models]

        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        metric.add_batch(preds[0], preds[-1])
        score = metric.evaluate()[-1]
        id_to_score.append((id[0], score))

    id_to_score.sort(key=lambda x: x[1], reverse=True)
    split = len(id_to_score) // 2
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for item in id_to_score[:split]:
            f.write(item[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for item in id_to_score[split:]:
            f.write(item[0] + '\n')
    logger.info(f'可靠/不可靠划分完成 → {split} / {len(id_to_score)-split}')


def select_reliable(models, dataloader, args, logger):
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for m in models:
        m.eval()

    id_to_score = []
    for img, mask, id in tqdm(dataloader, desc='Select-Reliable'):
        img = img.cuda()
        with torch.no_grad():
            preds = [torch.argmax(m(img), dim=1).cpu().numpy() for m in models]

        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        metric.add_batch(preds[0], preds[-1])
        score = metric.evaluate()[-1]
        id_to_score.append((id[0], score))

    id_to_score.sort(key=lambda x: x[1], reverse=True)
    split = len(id_to_score) // 2
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for item in id_to_score[:split]:
            f.write(item[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for item in id_to_score[split:]:
            f.write(item[0] + '\n')
    logger.info(f'可靠/不可靠划分完成 → {split} / {len(id_to_score)-split}')

# =================  主流程  =================
def main(args):
    logger, tb_writer = init_logger_and_tb()
    logger.info(f'训练配置 → {args}')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        logger.error('ST++ 必须指定 --reliable-id-path')
        exit(1)

    criterion = CrossEntropyLoss(ignore_index=255)
    valset = SemiDataset(args.dataset, args.data_root, 'val',None)
    valloader = DataLoader(valset,
                           batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4)

    global MODE
    # ----------  阶段 1：纯监督  ----------
    MODE = 'train'
    trainset = SemiDataset(args.dataset, args.data_root, MODE,
                           args.crop_size, args.labeled_id_path)
    if len(trainset.ids) < 200:   # 数据增强
        trainset.ids *= 2
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    logger.info(f'模型参数量 {count_params(model):.1f}M')
    best_model = train(model, trainloader, valloader,
                       criterion, optimizer, args, logger, tb_writer)

    # =================  非 plus 模式：两段结束  =================
    if not args.plus:
        logger.info('==== 进入 ST 标准两段模式 ====')
        # 伪标签
        labelset = SemiDataset(args.dataset, args.data_root, 'label', None,
                               None, args.unlabeled_id_path)
        labelloader = DataLoader(labelset, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=4)
        label(best_model, labelloader, args, logger)
        # 重训练
        MODE = 'semi_train'
        semi_set = SemiDataset(args.dataset, args.data_root, MODE,
                               args.crop_size, args.labeled_id_path,
                               args.unlabeled_id_path, args.pseudo_mask_path)
        semi_loader = DataLoader(semi_set, batch_size=args.batch_size,
                                 shuffle=True, pin_memory=True,
                                 num_workers=16, drop_last=True)
        model, optimizer = init_basic_elems(args)
        train(model, semi_loader, valloader,
              criterion, optimizer, args, logger, tb_writer)
        logger.info('ST 训练完成')
        return

    # =================  ST++ 六段模式  =================
    logger.info('==== 进入 ST++ 六段模式 ====')
    # ① 选可靠
    reliable_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                               None, args.unlabeled_id_path)
    reliable_loader = DataLoader(reliable_set, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=4)
    select_reliable([best_model], reliable_loader, args, logger)
    # ② 给可靠打伪标签
    reliable_txt = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    label_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                            None, reliable_txt)
    label_loader = DataLoader(label_set, batch_size=1, shuffle=False,
                              pin_memory=True, num_workers=4)
    label(best_model, label_loader, args, logger)
    # ③ 第一段重训练
    MODE = 'semi_train'
    semi_set = SemiDataset(args.dataset, args.data_root, MODE,
                           args.crop_size, args.labeled_id_path,
                           reliable_txt, args.pseudo_mask_path)
    semi_loader = DataLoader(semi_set, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=16, drop_last=True)
    model, optimizer = init_basic_elems(args)
    best_model = train(model, semi_loader, valloader,
                       criterion, optimizer, args, logger, tb_writer)
    # ④ 给不可靠打标签
    unreliable_txt = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    unrel_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                            None, unreliable_txt)
    unrel_loader = DataLoader(unrel_set, batch_size=1, shuffle=False,
                              pin_memory=True, num_workers=4)
    label(best_model, unrel_loader, args, logger)
    # ⑤ 第二段重训练
    final_set = SemiDataset(args.dataset, args.data_root, MODE,
                            args.crop_size, args.labeled_id_path,
                            args.unlabeled_id_path, args.pseudo_mask_path)
    final_loader = DataLoader(final_set, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=16, drop_last=True)
    model, optimizer = init_basic_elems(args)
    train(model, final_loader, valloader,
          criterion, optimizer, args, logger, tb_writer)
    logger.info('ST++ 训练完成')


# =================  entry  =================
if __name__ == '__main__':
    args = parse_args()
    # 默认超参
    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]
    main(args)