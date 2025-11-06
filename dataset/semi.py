#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
半监督数据集封装
支持 PASCAL VOC / Cityscapes 两套数据
mode 说明：
    train       – 仅用带标签数据做纯监督训练
    label       – 给无标签图片生成伪标签
    semi_train  – 半监督训练（带标签 + 伪标签）
    val         – 验证
"""
import math
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
# 引入自定义增强
from dataset.transform import crop, hflip, normalize, resize, blur, cutout


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size,
                 labeled_id_paths=None, unlabeled_id_paths=None, pseudo_mask_path=None):
        """
        参数
        ----
        name : str
            数据集名称，支持 'pascal' 或 'cityscapes'
        root : str
            数据集根目录
        mode : str
            运行模式，详见文件顶部说明
        size : int
            训练时随机裁剪尺寸
        labeled_id_path : str, optional
            存放“带标签图片 ID”的 txt 文件路径，train / semi_train 需要
        unlabeled_id_path : str, optional
            存放“无标签图片 ID”的 txt 文件路径，label / semi_train 需要
        pseudo_mask_path : str, optional
            存放伪标签的目录，semi_train 需要
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path
        if labeled_id_paths is not None:
            labeled_id_path = labeled_id_paths
        if unlabeled_id_paths is not None:
            unlabeled_id_path = unlabeled_id_paths

        # 半监督模式下，需要把“带标签”与“无标签”两份 ID 合并
        if mode == 'semi_train':

            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()

            # 为了保证每个 epoch 都能遍历到全部无标签数据，
            # 将带标签数据重复 ceil(N_unlabeled / N_labeled) 次
            repeat = math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids))
            self.ids = self.labeled_ids * repeat + self.unlabeled_ids
        else:
            # 其他模式只需读取对应 ID 文件
            if mode == 'val':
                if labeled_id_paths is not None:
                    id_path = labeled_id_path
                else:
                    id_path = f'dataset/splits/{name}/val.txt'
            elif mode == 'label':
                id_path = unlabeled_id_paths
            elif mode == 'train':
                id_path = labeled_id_paths
            else:
                raise ValueError(f'未知 mode: {mode}')

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # 读取图片 - 添加后缀检查
        if ' ' in id_:
            # 如果ID中包含空格，说明是完整路径格式（兼容旧格式）
            img_path = os.path.join(self.root, id_.split(' ')[0])
        else:
            # 如果ID中不包含空格，说明是只有文件名的格式，需要添加路径和后缀
            if self.name == 'pascal':
                img_path = os.path.join(self.root, 'JPEGImages', f'{id_}.jpg')
            else:  # cityscapes
                img_path = os.path.join(self.root, 'leftImg8bit', 'train', id_)  # 需要根据实际目录结构调整

        img = Image.open(img_path)

        # ---------- 验证 / 纯伪标签生成模式 ----------
        if self.mode in ('val', 'label'):
            # 构建mask路径 - 添加后缀检查
            if ' ' in id_:
                # 兼容旧格式
                mask_path = os.path.join(self.root, id_.split(' ')[1])
            else:
                # 新格式，只有文件名
                if self.name == 'pascal':
                    mask_path = os.path.join(self.root, 'SegmentationClass', f'{id_}.png')
                else:  # cityscapes
                    # 需要根据实际目录结构调整
                    mask_path = os.path.join(self.root, 'gtFine', 'train', f'{id_}_gtFine_labelIds.png')
            
            mask = Image.open(mask_path)
            img, mask = normalize(img, mask)   # 仅归一化，无增强
            return img, mask, id_

        # ---------- 训练模式（train / semi_train） ----------
        # 1. 决定读取哪份 mask
        if self.mode == 'train' or (self.mode == 'semi_train' and id_ in self.labeled_ids):
            # 构建mask路径 - 添加后缀检查
            if ' ' in id_:
                # 兼容旧格式
                mask_path = os.path.join(self.root, id_.split(' ')[1])
            else:
                # 新格式，只有文件名
                if self.name == 'pascal':
                    mask_path = os.path.join(self.root, 'SegmentationClass', f'{id_}.png')
                else:  # cityscapes
                    # 需要根据实际目录结构调整
                    mask_path = os.path.join(self.root, 'gtFine', 'train', f'{id_}_gtFine_labelIds.png')
            
            mask = Image.open(mask_path)
        else:
            # semi_train 且当前样本为无标签数据 → 读取伪标签
            if ' ' in id_:
                fname = os.path.basename(id_.split(' ')[1])
            else:
                fname = f'{id_}.png'  # 假设伪标签都是PNG格式
                
            mask_path = os.path.join(self.pseudo_mask_path, fname)
            mask = Image.open(mask_path)

        # 2. 基础增强（带标签 / 无标签都会做）
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))  # 随机缩放
        img, mask = crop(img, mask, self.size)                # 随机裁剪
        img, mask = hflip(img, mask, p=0.5)                   # 随机水平翻转

        # 3. 强增强（仅无标签数据）
        # 在 semi.py 文件中找到调用 cutout 的部分，修改为：
        # 3. 强增强（仅无标签数据）
        if self.mode == 'semi_train' and id_ in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            # 安全调用 cutout
            try:
                img, mask = cutout(img, mask, p=0.5)
            except ValueError:
                # 如果 cutout 失败（如擦除区域大于图像），则跳过此增强
                pass

        # 4. 归一化
        img, mask = normalize(img, mask)
        return img, mask