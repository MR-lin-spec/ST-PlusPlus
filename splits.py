#!/usr/bin/env python3
# splits.py  —— 仅输出 basename，无后缀
# /root/ST-PlusPlus/splits.py
import os, random
from pathlib import Path

# -------------------------------------------------
# 1. 扫描真实 jpg-png 对，返回 basename 列表
# -------------------------------------------------
def scan_real_pairs_basename(img_root: Path, img_sub: str, ann_sub: str):
    img_dir = img_root / img_sub
    ann_dir = img_root / ann_sub
    basenames = []
    for jpg in img_dir.glob("*.jpg"):
        png = ann_dir / f"{jpg.stem}.png"
        if png.exists():
            basenames.append(jpg.stem)
    return sorted(basenames)

# -------------------------------------------------
# 2. 划分训练子集 + 独立测试集
# -------------------------------------------------
def split_and_write_basenames(valid_list, ratios, out_base: Path, tag: str):
    random.seed(2022)
    shuffled = random.sample(valid_list, len(valid_list))

    # 留出 10 % 作为测试集
    val_num = max(1, int(len(shuffled) * 0.1))
    val_list   = shuffled[:val_num]
    train_list = shuffled[val_num:]

    # 写 val.txt（与比例子目录同级）
    (out_base / "val.txt").write_text("\n".join(val_list) + "\n")
    print(f"[val] 测试集数量：{len(val_list)}")

    # 对 train_list 再做比例划分
    for ratio in ratios:
        n_label = max(1, int(len(train_list) * ratio)) if ratio <= 1.0 else len(train_list)
        labeled   = train_list[:n_label]
        unlabeled = train_list[n_label:]

        ratio_str = f"{ratio}".replace("/", "_")
        split_dir = out_base / f"{tag}_{ratio_str}"
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "labeled.txt").write_text("\n".join(labeled) + "\n")
        (split_dir / "unlabeled.txt").write_text("\n".join(unlabeled) + "\n")

        print(f"[{tag}] ratio={ratio} -> labeled={len(labeled)}  unlabeled={len(unlabeled)}")

# -------------------------------------------------
# 3. 主入口
# -------------------------------------------------
if __name__ == "__main__":
 # --------------- 用户路径（保持不动） ---------------
    VOC2012_ROOT    = Path("/DeepLearning_linux/Projects/ST-PlusPlus/dataset/splits/VOC2012")
    BASE_OUTPUT_DIR = Path(r"\\DeepLearning_linux\Projects\ST-PlusPlus\dataset\pascal")
    # ----------------------------------------------------

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    basenames = scan_real_pairs_basename(VOC2012_ROOT, "JPEGImages", "SegmentationClass")
    print(f"磁盘上真实存在的图片-标签对：{len(basenames)} 组")

    ratios = [1, 1/2, 1/4, 1/8, 1/16]

    print("=== 开始划分（含测试集）===")
    split_and_write_basenames(basenames, ratios, BASE_OUTPUT_DIR, "original")

    print("\n=== 划分完成！文件清单 ===")
    print(BASE_OUTPUT_DIR / "val.txt")          # 测试集
    for root, _, files in os.walk(BASE_OUTPUT_DIR):
        for f in files:
            if f in {"label.txt", "unlabeled.txt"}:
                print(Path(root) / f)
