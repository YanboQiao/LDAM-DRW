#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_cifar100_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~
• 载入 PaCo/MoCo‑LDAM checkpoint
• 推断 CIFAR‑100 测试集
• 绘制:
  1) 100×100 混淆矩阵
  2) 每类 Top‑1 准确率柱状
  3) 训练样本数 vs. 准确率 散点图
• 终端输出整体 Top‑1 Accuracy
"""

import argparse
import os
import pathlib
import warnings
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torchvision.datasets as dsets
from sklearn.metrics import confusion_matrix

# ========= 根据你的工程结构导入 =========
from PaCoModels import resnet_cifar, resnet_big           # backbone
import moco.builder                                       # MoCo/PaCo wrapper
# ======================================


def build_model(arch: str,
                num_classes: int = 100,
                moco_dim: int = 128,
                moco_k: int = 1024,
                moco_m: float = 0.999,
                moco_t: float = 0.07,
                mlp: bool = True,
                feat_dim: int = 64) -> torch.nn.Module:
    """按训练脚本方式重建模型"""
    backbone = getattr(resnet_cifar if arch == "resnet32" else resnet_big, arch)
    model = moco.builder.MoCo(
        backbone, moco_dim, moco_k, moco_m, moco_t,
        mlp=mlp, feat_dim=feat_dim, num_classes=num_classes
    )
    return model


@torch.no_grad()
def inference(model: torch.nn.Module,
              loader: torch.utils.data.DataLoader,
              device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """测试集推断，返回真值与预测"""
    model.eval()
    y_true, y_pred = [], []
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        pred = model(imgs).argmax(dim=1)
        y_true.append(targets.numpy())
        y_pred.append(pred.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion(cm: np.ndarray, path: str) -> None:
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm,
                cmap="Blues",
                square=True,
                cbar_kws={"label": "Number of Samples"},
                vmax=cm.max())
    plt.title("Confusion Matrix (100×100) – CIFAR‑100")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_per_class_acc(cm: np.ndarray, path: str) -> np.ndarray:
    correct = np.diag(cm)
    total = cm.sum(axis=1)
    acc = correct / total
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(100), acc * 100.0)
    plt.ylim(0, 100)
    plt.xlabel("Class Index (0–99)")
    plt.ylabel("Accuracy (%)")
    plt.title("Per‑Class Accuracy on CIFAR‑100")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return acc


def plot_sample_vs_acc(train_counts: np.ndarray,
                       acc: np.ndarray,
                       path: str) -> None:
    """散点图: 训练样本数 ↔︎ 测试准确率"""
    x = train_counts
    y = acc * 100.0
    # 拟合 log10(x) 与 y 的线性关系
    coeff = np.polyfit(np.log10(x + 1e-6), y, 1)  # 加小量防 0
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = coeff[0] * np.log10(x_line) + coeff[1]

    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=y, cmap="viridis", s=60, edgecolor="k", alpha=0.8)
    plt.plot(x_line, y_line, "r--", linewidth=2, label="Trend Line")
    plt.xscale("log")
    plt.xlabel("Training Samples per Class")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs. Training Sample Count")
    cbar = plt.colorbar(sc)
    cbar.set_label("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    # -------------- 必要参数 --------------
    parser.add_argument("--ckpt", required=True,
                        help="模型 checkpoint (*.pth.tar)")
    # -------------- 可选参数 --------------
    parser.add_argument("--data-root", default="./data",
                        help="CIFAR‑100 数据根目录")
    parser.add_argument("--output-dir", default="./figs",
                        help="输出图片目录")
    parser.add_argument("--imb-factor", type=float, default=1.0,
                        help="ImbalanceCIFAR100 指数系数 (1.0 = 平衡)")
    # ----------- 网络超参数(可选) ----------
    parser.add_argument("--moco-dim", type=int, default=128)
    parser.add_argument("--moco-k", type=int, default=1024)
    parser.add_argument("--moco-m", type=float, default=0.999)
    parser.add_argument("--moco-t", type=float, default=0.07)
    parser.add_argument("--feat-dim", type=int, default=64)
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # ========== 1. 重建网络并加载权重 ==========
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "resnet32")

    model = build_model(
        arch,
        moco_dim=args.moco_dim,
        moco_k=args.moco_k,
        moco_m=args.moco_m,
        moco_t=args.moco_t,
        feat_dim=args.feat_dim,
    )
    new_state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        warnings.warn(f"Missing keys: {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys: {unexpected}")
    model = model.to(device)

    # ========== 2. 数据集 ==========
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    trans = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # --- 测试集 ---
    test_set = dsets.CIFAR100(root=args.data_root,
                              train=False, download=True, transform=trans)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # --- 训练集(仅统计样本数) ---
    if args.imb_factor != 1.0:
        from dataset.imbalance_cifar import ImbalanceCIFAR100
        train_set = ImbalanceCIFAR100(root=args.data_root, imb_type='exp',
                                      imb_factor=args.imb_factor, rand_number=0,
                                      train=True, download=True, transform=trans)
    else:
        train_set = dsets.CIFAR100(root=args.data_root,
                                   train=True, download=True, transform=trans)

    train_counts = np.bincount(train_set.targets, minlength=100)

    # ========== 3. 推断 & 评估 ==========
    print("==> 推断 CIFAR‑100 测试集 …")
    y_true, y_pred = inference(model, test_loader, device)
    overall_acc = (y_true == y_pred).mean() * 100.0
    print(f"[Overall Top‑1 Accuracy] {overall_acc:.2f}%")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(100)))

    # ========== 4. 绘图 ==========
    conf_path = os.path.join(args.output_dir, "confusion_matrix.png")
    bar_path = os.path.join(args.output_dir, "per_class_accuracy.png")
    scatter_path = os.path.join(args.output_dir, "sample_vs_accuracy.png")

    plot_confusion(cm, conf_path)
    per_class_acc = plot_per_class_acc(cm, bar_path)
    plot_sample_vs_acc(train_counts, per_class_acc, scatter_path)

    print(f"✓ 图片已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
