#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强工具集
所有函数均返回 PIL 对象（除 normalize 外最终返回 tensor）
"""
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms


def crop(img, mask, size):
    """
    随机裁剪到固定 size
    若原图小于 size，先补 0（图像）或 255（mask）
    """
    w, h = img.size
    padw = max(size - w, 0)
    padh = max(size - h, 0)
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))
    return img, mask


def hflip(img, mask, p=0.5):
    """随机水平翻转"""
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    """
    归一化到 ImageNet 均值方差
    返回 (img_tensor, mask_tensor) 或 img_tensor
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize(img, mask, base_size, ratio_range):
    """
    随机长边缩放
    ratio_range: (min_ratio, max_ratio)
    """
    w, h = img.size
    long_side = random.randint(
        int(base_size * ratio_range[0]),
        int(base_size * ratio_range[1])
    )
    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    """随机高斯模糊"""
    if random.random() < p:
        sigma = random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5,
           size_min=0.02, size_max=0.4,
           ratio_1=0.3, ratio_2=1 / 0.3,
           value_min=0, value_max=255,
           pixel_level=True):
    """
    Random Cutout / Random Erasing
    默认参数来自原文实现
    """
    if random.random() > p:
        return img, mask

    img = np.array(img)
    mask = np.array(mask)
    h, w = img.shape[:2]

    # 随机面积与长宽比
    while True:
        size = random.uniform(size_min, size_max) * h * w
        ratio = random.uniform(ratio_1, ratio_2)
        erase_h = int(np.sqrt(size * ratio))
        erase_w = int(np.sqrt(size / ratio))
        y = random.randint(0, h - erase_h)
        x = random.randint(0, w - erase_w)
        if y + erase_h <= h and x + erase_w <= w:
            break

    # 随机填充值
    if pixel_level:
        value = np.random.uniform(value_min, value_max,
                                  (erase_h, erase_w, img.shape[2]))
    else:
        value = np.random.uniform(value_min, value_max)

    img[y:y + erase_h, x:x + erase_w] = value
    mask[y:y + erase_h, x:x + erase_w] = 255   # 255 视为忽略区域

    img = Image.fromarray(img.astype(np.uint8))
    mask = Image.fromarray(mask.astype(np.uint8))
    return img, mask