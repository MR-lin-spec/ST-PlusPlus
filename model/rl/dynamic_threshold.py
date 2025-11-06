# dynamic_threshold.py  (无裁剪版)
import torch
import numpy as np

class DynamicThreshold:
    def __init__(self,
                 low_percentile: float = 20.0,
                 high_percentile: float = 80.0,
                 min_low: float = 0.0,      # 宽松下限
                 max_low: float = 1.0,      # 宽松上限，实际不再生效
                 min_high: float = 0.0,
                 max_high: float = 1.0,
                 adaptive: bool = True):
        # 参数校验（可选保留，防止百分位写反）
        if not (0 <= low_percentile <= 100 and 0 <= high_percentile <= 100):
            raise ValueError("百分位必须在 [0,100] 之间")
        if low_percentile >= high_percentile:
            raise ValueError("low_percentile 必须小于 high_percentile")

        self.low_p = low_percentile
        self.high_p = high_percentile
        self.adaptive = adaptive
        # 边界保留，但后面不再 clip，仅作接口兼容
        self.min_low = min_low
        self.max_low = max_low
        self.min_high = min_high
        self.max_high = max_high

    # ------------- 内部工具 -------------
    def _get_flat_entropy(self, entropy_map: torch.Tensor) -> np.ndarray:
        """展平并返回 numpy 数组"""
        return entropy_map.detach().cpu().numpy().flatten()

    def _calc_percentile(self, flat_entropy: np.ndarray, percentile: float) -> float:
        """直接返回百分位数值，不做 clip"""
        if flat_entropy.size == 0:
            raise ValueError("熵图为空，无法计算阈值")
        return float(np.percentile(flat_entropy, percentile))

    # ------------- 对外接口 -------------
   # dynamic_threshold.py
    def compute_low_threshold(self, entropy_map: torch.Tensor) -> float:
        """计算低熵区域动态阈值（基于低百分位）+ 绝对值兜底"""
        if not self.adaptive:
            return self.min_low          # 非自适应模式仍返回预设下限

        flat_entropy = self._get_flat_entropy(entropy_map)
        raw = self._calc_percentile(flat_entropy, self.low_p)
        return max(raw, 0.4)            # 关键兜底：不让阈值 < 0.4

    def compute_high_threshold(self, entropy_map: torch.Tensor) -> float:
        if not self.adaptive:
            return self.max_high
        flat_entropy = self._get_flat_entropy(entropy_map)
        raw = self._calc_percentile(flat_entropy, self.high_p)
        return max(raw, 0.7)          # 兜底不让它过小