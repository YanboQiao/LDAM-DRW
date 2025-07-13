# balanced_augmentation.py  ----------------------------------------------------
import random
from collections import defaultdict
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# -------------------- 全局增强参数 --------------------
AUG_CONF = {
    "head": {  # Head 类：几乎不变，只做翻转 & 裁剪
        "rotate_deg": 30,
        "rotate_step": 30,
        "gaussian_p": 0.1,
        "hflip_p": 0.5,
        "multi_rotate": 1,  # 单次旋转
    },
    "tail": {  # Tail 类：强旋转 ±180°、更高概率模糊、多次旋转
        "rotate_deg": 180,
        "rotate_step": 30,   # 更小步长，更细粒度的角度
        "gaussian_p": 0.3,
        "hflip_p": 0.7,
        "multi_rotate": 4,   # 最多旋转3次，让模型学会形状而非角度
    },
}


# -------------------- 数据增强组件 --------------------
class DiscreteRotate:
    """离散随机旋转：±rotate_deg 内取 n·rotate_step°"""
    def __init__(self, rotate_deg: int, rotate_step: int, multi_rotate: int = 1):
        self.rotate_deg, self.rotate_step = rotate_deg, rotate_step
        self.multi_rotate = multi_rotate  # 多次旋转次数

    def __call__(self, img):
        if self.rotate_deg == 0:
            return img
            
        n_range = self.rotate_deg // self.rotate_step
        result = img
        
        # 进行多次旋转
        rotations = min(self.multi_rotate, 1) if n_range == 0 else self.multi_rotate
        for _ in range(rotations):
            k = random.randint(-n_range, n_range)
            result = T.functional.rotate(result, k * self.rotate_step, expand=False)
            
        return result


class SelectiveAugmentation:
    """
    - 必选：RandomCrop(32, pad=4)
    - 可选（最多 1 种，40 % 概率不加任何增强）：
        · RandomHorizontalFlip
        · DiscreteRotate
        · GaussianBlur
    - 末尾 ToTensor + Normalize
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.crop = T.RandomCrop(32, padding=4)
        self.hflip = (T.RandomHorizontalFlip(p=1.0)
                      if cfg["hflip_p"] > 0 else None)
        self.rotate = (DiscreteRotate(
                        cfg["rotate_deg"], 
                        cfg["rotate_step"],
                        cfg.get("multi_rotate", 1))  # 获取多旋转次数，默认为1
                       if cfg["rotate_deg"] > 0 else None)
        self.blur = (T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                     if cfg["gaussian_p"] > 0 else None)

        self.to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, img):
        img = self.crop(img)

        # 组装候选增强
        cand = []
        if self.hflip: cand.append(("hflip", self.hflip, self.cfg["hflip_p"]))
        if self.rotate: cand.append(("rotate", self.rotate, 1.0))
        if self.blur:   cand.append(("blur", self.blur, self.cfg["gaussian_p"]))

        # 对于多旋转配置，提高应用概率 (如果是尾部类别则提高应用增强概率)
        apply_prob = 0.8 if self.cfg.get("multi_rotate", 1) > 1 else 0.6
        if cand and random.random() < apply_prob:  # 提高概率应用增强
            name, op, p = random.choice(cand)
            if random.random() < p:
                img = op(img)

        return self.to_tensor_norm(img)


def _build_tf(cfg):  # 简化入口
    return SelectiveAugmentation(cfg)


# ----------------- 主数据集包装 -----------------
class BalancedAugmentedCIFAR(Dataset):
    """
    • Head 类   →  1.1 × n   （最多取整）
    • Tail 类   →  2   × n
    • 若总量仍超 2 × 原始，则再按比例压缩 Tail 类扩充量
    """
    def __init__(self, base_ds, cls_num_list: List[int]):
        self.base_data   = base_ds.data
        self.base_labels = base_ds.targets
        self.orig_counts = cls_num_list

        self.max_orig = max(cls_num_list)
        self.orig_total = len(base_ds)
        self.num_classes = len(cls_num_list)

        # ---------------- 1. 计算目标样本数 ----------------
        tgt = []
        head_mask = []
        for n in cls_num_list:
            if n >= 0.7 * self.max_orig:  # 最头部类别（70%以上）
                tgt.append(int(round(n * 1.1)))
                head_mask.append(True)
            elif n >= 0.5 * self.max_orig:  # 头部类别（50%-70%）
                tgt.append(int(round(n * 1.3)))
                head_mask.append(True)
            elif n >= 0.3 * self.max_orig:  # 中上部类别（30%-50%）
                tgt.append(int(round(n * 2.0)))
                head_mask.append(False)
            elif n >= 0.15 * self.max_orig:  # 中部类别（15%-30%）
                tgt.append(int(round(n * 3.0)))
                head_mask.append(False)
            elif n >= 0.05 * self.max_orig:  # 中下部类别（5%-15%）
                tgt.append(int(round(n * 4.5)))
                head_mask.append(False)
            else:  # 极尾部类别（<5%）
                tgt.append(int(round(n * 6.5)))
                head_mask.append(False)

        # -------------- 2. 若超总量 2×，缩减 Tail 扩充 ------
        max_total = self.orig_total * 2
        new_total = sum(tgt)
        if new_total > max_total:
            excess = new_total - max_total
            tail_extra = sum(tgt[i] - cls_num_list[i]
                             for i, is_head in enumerate(head_mask) if not is_head)
            # 计算缩减比例
            ratio = max(0.0, 1 - excess / max(tail_extra, 1))
            for i, is_head in enumerate(head_mask):
                if not is_head:
                    add = tgt[i] - cls_num_list[i]
                    tgt[i] = cls_num_list[i] + int(round(add * ratio))

        self.tgt_counts = tgt
        self.final_total = sum(tgt)

        # ---------------- 3. 构造索引池 -------------------
        per_cls_idx = defaultdict(list)
        for idx, lab in enumerate(self.base_labels):
            per_cls_idx[lab].append(idx)

        self.indices: List[Tuple[int, int]] = []
        for c in range(self.num_classes):
            idxs = per_cls_idx[c]
            need = self.tgt_counts[c]
            if need <= len(idxs):
                self.indices.extend([(i, c) for i in random.sample(idxs, need)])
            else:
                self.indices.extend([(i, c) for i in idxs])
                extra = random.choices(idxs, k=need - len(idxs))
                self.indices.extend([(i, c) for i in extra])

        # ---------------- 4. 预生成 transform -------------
        self.head_tf = _build_tf(AUG_CONF["head"])
        self.tail_tf = _build_tf(AUG_CONF["tail"])

        self.targets = [lab for _, lab in self.indices]

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx, lab = self.indices[idx]
        img = Image.fromarray(self.base_data[base_idx])
        tf = self.tail_tf if self.orig_counts[lab] < 0.5 * self.max_orig else self.head_tf
        return tf(img), lab

    def get_cls_num_list(self):
        return self.tgt_counts


# ---------------- Quick Sanity Check ----------------
if __name__ == "__main__":
    from imbalance_cifar import IMBALANCECIFAR10
    import numpy as np

    raw = IMBALANCECIFAR10(root="./data", imb_type="exp",
                           imb_factor=0.01, train=True,
                           download=True, transform=T.ToTensor())

    bal = BalancedAugmentedCIFAR(raw, raw.get_cls_num_list())

    print("原始总样本:", len(raw))
    print("增强后总样本:", len(bal))
    print("\n头部前 5 类目标数:", bal.get_cls_num_list()[:5])
    print("尾部后 5 类目标数:", bal.get_cls_num_list()[-5:])
