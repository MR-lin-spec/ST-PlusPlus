#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ST++ 半监督语义分割主入口（仅保留DQN伪标签筛选环节）
用法示例：
    # 原始 ST++
    python main.py --data-root /data/PASCAL --dataset pascal \
                   --labeled-id-path dataset/splits/pascal/labeled.txt \
                   --unlabeled-id-path dataset/splits/pascal/unlabeled.txt \
                   --pseudo-mask-path out/pseudo --save-path out/weights
    # DQN筛选模式（--algorithm st++_rl 生效）
    python main.py ... --algorithm st++_rl --reliable-id-path out/reliable
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
# ----------  新增：导入简化版DQN筛选器  ----------
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
    parser = argparse.ArgumentParser(description='ST++ 半监督框架（仅保留DQN筛选）')
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
    # ********  算法选择（--algorithm st++_rl 启用DQN筛选）  ********
    parser.add_argument('--algorithm', type=str,
                        choices=['st++', 'st++_rl'], default='st++',
                        help='st++: 原始筛选；st++_rl: DQN伪标签筛选')
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
# =================  训练函数（删除RL损失环节）  =================
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
        epoch_loss = 0.0  # 移除RL损失统计
        train_bar = tqdm(trainloader, desc=f'Epoch[{epoch}]')
        for img, mask in train_bar:
            img, mask = img.cuda(), mask.cuda()
            pred = model(img)
            # 仅保留基础交叉熵损失（删除RL损失相关代码）
            total_loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            GLOBAL_ITERS += 1
            epoch_loss += total_loss.item()
            # 学习率更新（保持原逻辑）
            lr_factor = max(0.0, 1 - GLOBAL_ITERS / total_iters)
            lr_factor = max(lr_factor, 1e-8)
            lr = args.lr * (1 - GLOBAL_ITERS / total_iters) ** 0.9
            if isinstance(lr, complex):
                lr = lr.real
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10 \
                if args.model != 'deeplabv2' else lr
            # 更新进度条（移除RL损失显示）
            train_bar.set_postfix(
                Loss=epoch_loss / (train_bar.n + 1))
            # TensorBoard（仅记录基础损失和学习率）
            tb_writer.add_scalar('loss/base', total_loss, GLOBAL_ITERS)
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
        # 最佳模型保存（保持原逻辑）
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
        # 早停（保持原逻辑）
        if mIOU > best_metric + 0.1:
            best_metric, patience = mIOU, 10
        else:
            patience -= 1
            if patience == 0:
                logger.info('早停触发')
                break
    return best_model
# =================  伪标签生成（保持原逻辑）  =================
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
# =================  可靠/不可靠划分（核心修改：增加DQN筛选分支）  =================
def select_reliable(models, dataloader, args, logger):
    """
    兼容两种筛选模式：
    - st++: 原始基于mIOU的划分
    - st++_rl: DQN筛选（需2个checkpoint）
    """
    os.makedirs(args.reliable_id_path, exist_ok=True)
    for m in models:
        m.eval()
    
    # 1. DQN筛选模式（--algorithm st++_rl）
    if args.algorithm == 'st++_rl':
        if not DQN_AVAILABLE:
            raise RuntimeError('DQN模块未找到，请确保 model/rl/simple_dqn.py 存在')
        if len(models) < 2:
            raise RuntimeError('DQN筛选需至少2个checkpoint（中期+末期）')
        
        logger.info('启用DQN伪标签筛选模式...')
        # 1.1 准备DQN训练数据（伪标签+两个checkpoint预测）
        dqn_train_data = []
        id_to_info = []  # 存储(id, 伪标签, 中期预测, 末期预测)
        for img, mask, id in tqdm(dataloader, desc='Prepare DQN Data'):
            img = img.cuda()
            with torch.no_grad():
                # 中期checkpoint预测（models[0]）、末期checkpoint预测（models[1]）
                ckpt1_pred = torch.argmax(models[0](img), dim=1).cpu().numpy()[0]
                ckpt2_pred = torch.argmax(models[1](img), dim=1).cpu().numpy()[0]
                pseudo_mask = ckpt2_pred  # 教师伪标签用末期checkpoint结果
            
            dqn_train_data.append((pseudo_mask, ckpt1_pred, ckpt2_pred))
            id_to_info.append((id[0], pseudo_mask, ckpt1_pred, ckpt2_pred))
        
        # 1.2 初始化并训练DQN（5轮内完成）
        selector = PseudoLabelSelector(dataset_name=args.dataset, device='cuda')
        logger.info('开始训练简化DQN（5轮收敛）...')
        selector.train_dqn(dqn_train_data, epochs=5)
        
        # 1.3 DQN筛选可靠图像
        reliable_ids = []
        unreliable_ids = []
        for id_, pseudo_mask, ckpt1_pred, ckpt2_pred in tqdm(id_to_info, desc='DQN Selection'):
            is_reliable = selector.select_reliable(pseudo_mask, ckpt1_pred, ckpt2_pred)
            if is_reliable:
                reliable_ids.append(id_)
            else:
                unreliable_ids.append(id_)
    
    # 2. 原始ST++筛选模式（--algorithm st++）
    else:
        logger.info('启用原始ST++筛选模式（基于mIOU）...')
        id_to_score = []
        for img, mask, id in tqdm(dataloader, desc='Select-Reliable'):
            img = img.cuda()
            with torch.no_grad():
                preds = [torch.argmax(m(img), dim=1).cpu().numpy() for m in models]
            # 计算预测一致性（mIOU）作为评分
            metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
            metric.add_batch(preds[0], preds[-1])
            score = metric.evaluate()[-1]
            id_to_score.append((id[0], score))
        
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
# =================  主流程（保持原结构，适配DQN筛选）  =================
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
    if len(trainset.ids) < 200:   # 数据增强（原逻辑）
        trainset.ids *= 2
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=16, drop_last=True)
    model, optimizer = init_basic_elems(args)
    logger.info(f'模型参数量 {count_params(model):.1f}M')
    
    # 保存中期checkpoint（总epoch的1/3处）
    mid_epoch = args.epochs // 3
    mid_model = None
    for epoch in range(args.epochs):
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
    select_reliable([mid_model, final_model], reliable_loader, args, logger)  # 传入2个checkpoint

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
# =================  入口函数（保持原逻辑）  =================
if __name__ == '__main__':
    args = parse_args()
    # 默认超参（原逻辑）
    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]
    main(args)