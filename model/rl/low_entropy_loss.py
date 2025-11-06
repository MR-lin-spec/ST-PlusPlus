import torch
import torch.nn as nn
import torch.nn.functional as F 
class SegL(nn.Module):
    def __init__(self, num_classes=1, lambd=0.5, smooth=1e-5):
        super().__init__()
        self.n = num_classes
        self.lambd = lambd
        self.smooth = smooth

    # ---------- 一次前向 = 逐类求 Dice + CE，再平均 ----------
    def forward(self, pred_stu, pred_tea, low_mask):
        """
        pred_stu: (N,C,H,W)  学生 logit
        pred_tea: (N,C,H,W)  教师概率（已detach）
        low_mask:(N,1,H,W)   低熵掩膜
        return: 标量
        """
        N, C, H, W = pred_stu.shape
        # 1. 生成伪标签（one-hot 形式）
        with torch.no_grad():
            if self.n == 1:                      # 二分类
                pseudo = (pred_tea > 0.5).float()
            else:                                # 多分类
                pseudo = F.one_hot(pred_tea.argmax(1), C).permute(0,3,1,2).float()

        # 2. 掩膜广播到 (N,C,H,W)
        mask = low_mask.expand_as(pred_stu)        # 0/1

        # 3. 逐类 Dice
        inter = (pred_tea * pseudo * mask).sum(dim=(2,3))   # (N,C)
        union = (pred_tea + pseudo).sum(dim=(2,3))
        dice  = (2. * inter + self.smooth) / (union + self.smooth)
        loss_dice = 1. - dice.mean()                        # 类维平均

        # 4. 逐类 CE（二分类 BCE / 多分类 CE）
        if self.n == 1:
            ce = F.binary_cross_entropy_with_logits(pred_stu, pseudo, reduction='none')
        else:
            ce = F.cross_entropy(pred_stu, pseudo.argmax(1), reduction='none').unsqueeze(1)
        loss_ce = (ce * mask).sum() / (mask.sum() + 1e-8)

        return (1 - self.lambd) * loss_dice + self.lambd * loss_ce


class DiceLoss(nn.Module):
        """
    通用 DiceLoss
    mode='binary'  -> 输入 pred (N,1,H,W)  target (N,H,W)  0/1
    mode='multiclass'-> pred (N,C,H,W)  target (N,H,W)  0~C-1
        """
        def __init__(self, mode='binary', smooth=1e-5, ignore_index=-100):
         super().__init__()
         self.mode = mode
         self.smooth = smooth
         self.ignore_index = ignore_index

        def forward(self, pred, target):
            if self.mode == 'binary':
                return self._binary_forward(pred, target)
            else:
                return self._multiclass_forward(pred, target)

        # ---------- 二分支 ----------
        def _binary_forward(self, pred, target):
            pred = torch.sigmoid(pred).squeeze(1)          # (N,H,W)
            target = target.float()
            inter = (pred * target).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            dice  = (2. * inter + self.smooth) / (union + self.smooth)
            return 1. - dice.mean()

        # ---------- 多分支 ----------
        def _multiclass_forward(self, pred, target):
            C = pred.shape[1]
            prob = F.softmax(pred, dim=1)                # (N,C,H,W)
            target = target.long()
            if self.ignore_index >= 0:
                mask = (target != self.ignore_index)
                target = target * mask
            else:
                mask = 1.0

            # one-hot
            target_oh = F.one_hot(target.clamp(min=0), C)  # (N,H,W,C)
            target_oh = target_oh.permute(0, 3, 1, 2).float()  # (N,C,H,W)

            inter = (prob * target_oh * mask).sum(dim=(2, 3))  # (N,C)
            union = (prob + target_oh).sum(dim=(2, 3))
            dice  = (2. * inter + self.smooth) / (union + self.smooth)
            return 1. - dice.mean()

@torch.no_grad()
def calc_dice(model, data_loader, mode='binary', device='cuda'):
    model.eval()
    dice_sum = cnt = 0
    for img, msk in data_loader:
        img, msk = img.to(device), msk.to(device)
        pred = model(img)
        if mode == 'binary':
            pred = torch.sigmoid(pred).squeeze(1)
            pred = (pred > 0.5).float()
            inter = (pred * msk).sum(dim=(1, 2))
            union = pred.sum(dim=(1, 2)) + msk.sum(dim=(1, 2))
            dice  = (2. * inter + 1e-8) / (union + 1e-8)
        else:  # multiclass
            pred = pred.argmax(1)          # (N,H,W)
            C = pred.shape[1] if pred.dim() == 4 else pred.max().item() + 1
            pred_oh = F.one_hot(pred, C).permute(0, 3, 1, 2).float()
            msk_oh  = F.one_hot(msk.long(), C).permute(0, 3, 1, 2).float()
            inter = (pred_oh * msk_oh).sum(dim=(2, 3))
            union = pred_oh.sum(dim=(2, 3)) + msk_oh.sum(dim=(2, 3))
            dice  = (2. * inter + 1e-8) / (union + 1e-8)   # (N,C)
            dice  = dice.mean()                            # 先样本后类别平均

        dice_sum += dice.sum().item()
        cnt += dice.numel()

    model.train()
    return dice_sum / cnt


def loss_Diloss(pred_t, target_img_prob, entropy_t, low_thresh, SegL, lambd=1.0):
    """
    pred_t: (N,C,H,W)         目标域 logit（学生）
    target_img_prob: (N,C,H,W) 目标图像概率图（教师①，已detach）
    entropy_t: (N,1,H,W)      目标域熵图
    low_thresh: float
    SegL: 已实例化的 SegL（内部按类求和）
    lambd: 自蒸馏权重
    return: 标量
    """
    low_mask1 =entropy_t.where(entropy_t <=low_thresh,
                                 torch.tensor(float('0')).to(entropy_t))
    low_mask= low_mask = (entropy_t <= low_thresh).float()   # (N,1,H,W)
    # 1. 图像概率图 → 目标域输出
    loss_img = SegL(pred_stu=pred_t,
                    pred_tea=target_img_prob.detach(),
                    low_mask=low_mask)

    # 2. 目标域自蒸馏（自己拟合自己低熵区域）
    with torch.no_grad():
        if pred_t.shape[1] == 1:
            prob_t = torch.sigmoid(pred_t)
        else:
            prob_t = torch.softmax(pred_t, dim=1)
    loss_self = SegL(pred_stu=pred_t,
                     pred_tea=prob_t.detach(),
                     low_mask=low_mask)

    # 3. 两次损失相加（类维已在 SegL 内平均）
    return (1-lambd)*loss_img + lambd * loss_self
