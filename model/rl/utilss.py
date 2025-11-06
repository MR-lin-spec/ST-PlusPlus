import torch
import matplotlib.pyplot as plt
from typing import Optional
# --------- 1. 纯计算 ---------
def compute_entropy_map(prob_map: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    二分类熵图计算器
    Args:
        prob_map: HW 或 NCHW 的预测概率 (0~1)
        eps: 防 log0
    Returns:
        与 prob_map 同 shape 的熵图，值域 0~1
    """
    p = prob_map.clamp(eps, 1 - eps)
    entropy = -(p * p.log2() + (1 - p) * (1 - p).log2())
    return entropy
# --------- 2. 纯可视化 ---------
def vis_entropy_map(entropy_map: torch.Tensor,
                    save_path: Optional[str] = None,
                    show: bool = False,
                    cmap: str = 'jet',
                    title: str='entropy map'):
    """
    可视化熵图（HW）
    """
    plt.imshow(entropy_map.cpu(), cmap=cmap)
    #plt.colorbar(label='entropy')
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()