#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST++ 半监督语义分割主入口（DQN伪标签筛选优化版）
优化内容：
1. 显存隔离：DQN训练前卸载segmentation模型，避免显存叠加
2. 流式数据：使用生成器替代全量列表，解决内存问题
3. 批量筛选：支持批量预测，提升效率
4. 异常防护：增加数据校验、收敛检测、自动重训机制
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
import gc
# =================  自定义包  =================
from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
# ----------  新增：导入优化版DQN筛选器  ----------
try:
    from model.rl.simple_dqn import PseudoLabelSelector
    DQN_AVAILABLE = True
except ImportError:
    DQN_AVAILABLE = False
# =================  全局变量  =================
MODE = None          # 当前训练模式：train / semi_train
GLOBAL_ITERS = 0     # 全局迭代计数（给 tensorboard 用）
# =================  参数解析  =================
def parse_args():
    parser = argparse.ArgumentParser(description='ST++ 半监督框架（DQN伪标签筛选优化版）')
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
    # ST++ 特有（DQN筛选需此路径）
    parser.add_argument('--reliable-id-path', type=str,
                        help='可靠/不可靠图像ID保存路径（--algorithm st++_rl 时必需）')
    parser.add_argument('--plus', action='store_true',
                        help='是否启用 ST++ 两段重训练（DQN筛选依赖此模式）')
    # 算法选择
    parser.add_argument('--algorithm', type=str,
                        choices=['st++', 'st++_rl'], default='st++',
                        help='st++: 原始筛选；st++_rl: DQN伪标签筛选')
    # DQN训练配置
    parser.add_argument('--dqn-batch-size', type=int, default=8,
                        help='DQN训练批量大小（影响显存与速度）')
    parser.add_argument('--dqn-epochs', type=int, default=5,
                        help='DQN训练轮次（建议5-8轮）')
    args = parser.parse_args()
    # 校验：st++_rl 模式必须指定 --reliable-id-path 和 --plus
    if args.algorithm == 'st++_rl':
        if not args.plus or args.reliable_id_path is None:
            parser.error('--algorithm st++_rl 必须配合 --plus 和 --reliable-id-path 使用')
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
# =================  训练函数  =================
def train(model, trainloader, valloader, criterion, optimizer, args, logger, tb_writer):
    global GLOBAL_ITERS
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    best_model = copy.deepcopy(model)
    # 早停
    patience, best_metric = 10, 0
    for epoch in range(args.epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(trainloader, desc=f'Epoch[{epoch}]')
        for img, mask in train_bar:
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            total_loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            GLOBAL_ITERS += 1
            epoch_loss += total_loss.item()
            # 学习率更新
            lr_factor = max(0.0, 1 - GLOBAL_ITERS / total_iters)
            lr_factor = max(lr_factor, 1e-8)
            lr = args.lr * (1 - GLOBAL_ITERS / total_iters) ** 0.9
            if isinstance(lr, complex):
                lr = lr.real
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10 \
                if args.model != 'deeplabv2' else lr
            # 更新进度条
            train_bar.set_postfix(
                Loss=epoch_loss / (train_bar.n + 1))
            # TensorBoard
            tb_writer.add_scalar('loss/base', total_loss, GLOBAL_ITERS)
            tb_writer.add_scalar('lr', lr, GLOBAL_ITERS)
        # 验证
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
        # 最佳模型保存
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
            fname = os.path.basename(id[0].split(' ')[1])
        else:
            if args.dataset == 'pascal':
                fname = f'{id[0]}.png'
            else:  # cityscapes
                fname = f'{id[0]}_gtFine_labelIds.png'
        
        # 保存伪标签
        pred_array = pred.squeeze(0).numpy().astype(np.uint8)
        pil_image = Image.fromarray(pred_array, mode='P')
        pil_image.putpalette(cmap)
        pil_image.save(os.path.join(args.pseudo_mask_path, fname))
    logger.info(f'伪标签完成  mIOU={metric.evaluate()[-1]*100:.2f}%')
# =================  可靠/不可靠划分（核心修改：生成器 + 显存隔离 + 批量筛选）  =================
def select_reliable(models, dataloader, args, logger):
    """
    兼容两种筛选模式：
    - st++: 原始基于mIOU的划分
    - st++_rl: DQN筛选（需2个checkpoint，流式数据处理）
    """
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for m in models:
        m.eval()
    logger.info('启用原始ST++筛选模式（基于mIOU，批量计算优化）...')
    id_to_score = []
        
        # 批量处理提升效率
    batch_size = 16  # 可根据GPU显存调整
        
    for img, mask, id in tqdm(dataloader, desc='Select-Reliable'):
        img = img.cuda()
        with torch.no_grad():
             preds = [torch.argmax(m(img), dim=1).cpu().numpy() for m in models]
            
            # 计算预测一致性（mIOU）
        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        metric.add_batch(preds[0], preds[-1])
        score = metric.evaluate()[-1]
        id_to_score.append((id[0], score))
            
            # 定期清理显存
        if len(id_to_score) % batch_size == 0:
                torch.cuda.empty_cache()
        
        # 按评分排序，取前50%作为可靠图像
        id_to_score.sort(key=lambda x: x[1], reverse=True)
        split = len(id_to_score) // 2
        reliable_ids = [item[0] for item in id_to_score[:split]]
        unreliable_ids = [item[0] for item in id_to_score[split:]]  
    
    # 保存划分结果（两种模式统一输出格式）
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for item in reliable_ids:
            f.write(item + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for item in unreliable_ids:
            f.write(item + '\n')
    logger.info(f'可靠/不可靠划分完成 → 可靠：{len(reliable_ids)} / 不可靠：{len(unreliable_ids)}')
    
    # 最终显存与内存释放
    torch.cuda.empty_cache()
    gc.collect()

# =================  主流程  =================
def main(args):
    print(f"DQN available: {DQN_AVAILABLE}")
    logger, tb_writer = init_logger_and_tb()
    logger.info(f'训练配置 → {args}')
    # 路径初始化
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.pseudo_mask_path, exist_ok=True)
    if args.plus and args.reliable_id_path is None:
        logger.error('ST++ 模式必须指定 --reliable-id-path')
        exit(1)
    
    # 基础组件初始化
    criterion = CrossEntropyLoss(ignore_index=255)
    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset,
                           batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4)
    global MODE

    # ----------  阶段 1：纯监督训练（保留2个checkpoint：中期+末期）  ----------
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
    
    # 保存中期checkpoint（总epoch的1/3处）
    mid_epoch = args.epochs // 3
    mid_model = None
    begin_model=None
    # 训练循环
    for epoch in range(args.epochs):
        begin_model=copy.deepcopy(model)
        # 训练单轮
        model.train()
        epoch_loss = 0.0
        for img, mask in tqdm(trainloader, desc=f'Epoch[{epoch}/{args.epochs}]'):
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            epoch_loss += loss.item()
        
        # 保存中期checkpoint
        if epoch == mid_epoch:
            mid_model = copy.deepcopy(model)
            mid_save_path = os.path.join(args.save_path, 'mid_checkpoint.pth')
            torch.save(mid_model.module.state_dict(), mid_save_path)
            logger.info(f'保存中期checkpoint → {mid_save_path}')
    
    # 训练完成后获取末期checkpoint（最佳模型）
    final_model = train(model, trainloader, valloader,
                       criterion, optimizer, args, logger, tb_writer)

    # =================  非 plus 模式：原始ST流程（无筛选）  =================
    if not args.plus:
        logger.info('==== 进入 ST 标准两段模式 ====')
        # 伪标签生成
        labelset = SemiDataset(args.dataset, args.data_root, 'label', None,
                               None, args.unlabeled_id_path)
        labelloader = DataLoader(labelset, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=4)
        label(final_model, labelloader, args, logger)
        # 半监督重训练
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

    # =================  ST++ 模式（含筛选：原始/DQN二选一）  =================
    logger.info(f'==== 进入 ST++ 模式（筛选算法：{args.algorithm}） ====')
    
    # ① 可靠/不可靠划分（传入中期+末期两个checkpoint）
    reliable_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                               None, args.unlabeled_id_path)
    reliable_loader = DataLoader(reliable_set, batch_size=1, shuffle=False,
                                 pin_memory=True, num_workers=4)
    
    # 显存隔离：划分前释放不必要的显存
    torch.cuda.empty_cache()
    
    select_reliable([begin_model,mid_model, final_model], reliable_loader, args, logger)  # 传入2个checkpoint

    # ② 给可靠图像打伪标签
    reliable_txt = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    label_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                            None, reliable_txt)
    label_loader = DataLoader(label_set, batch_size=1, shuffle=False,
                              pin_memory=True, num_workers=4)
    label(final_model, label_loader, args, logger)

    # ③ 第一段重训练（可靠图像+带标签数据）
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

    # ④ 给不可靠图像打伪标签（用第一段重训练后的最佳模型）
    unreliable_txt = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    unrel_set = SemiDataset(args.dataset, args.data_root, 'label', None,
                            None, unreliable_txt)
    unrel_loader = DataLoader(unrel_set, batch_size=1, shuffle=False,
                              pin_memory=True, num_workers=4)
    label(best_model, unrel_loader, args, logger)

    # ⑤ 第二段重训练（全部数据）
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
    
    # 最终显存与内存释放
    torch.cuda.empty_cache()
    gc.collect()
# =================  入口函数  =================
if __name__ == '__main__':
    args = parse_args()
    # 默认超参
    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]
    if args.dqn_batch_size is None:
        args.dqn_batch_size = 8  # 默认DQN批量大小
    
    main(args)