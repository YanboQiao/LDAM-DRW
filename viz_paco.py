#!/usr/bin/env python3
# viz_paco.py – 计算 per‑class accuracy 并绘制 100×100 混淆矩阵

import argparse
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.datasets as tvds
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- 1. 构建模型 ----------
def build_model(arch: str, num_classes: int = 100):
    from PaCoModels import resnet_cifar, resnet_big
    fn = getattr(resnet_cifar, arch) if arch == 'resnet32' else getattr(resnet_big, arch)
    return fn(num_classes=num_classes)


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--arch', default='resnet32', choices=['resnet32', 'resnet50'])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--num-workers', default=0, type=int)          # 为避免 Win 多进程问题默认设 0
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading checkpoint: {args.ckpt}')
    state = torch.load(args.ckpt, map_location='cpu')
    state_dict = state['state_dict'] if 'state_dict' in state else state
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = build_model(args.arch).to(args.device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    val_set = tvds.CIFAR100(root=args.data_root, train=False, download=True, transform=val_tf)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Infer'):
            imgs = imgs.to(args.device)
            logits = model(imgs)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(100))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(18, 4))
    plt.bar(np.arange(100), per_class_acc * 100)
    plt.xlabel('Class index (0‑99)')
    plt.ylabel('Accuracy (%)')
    plt.title('Per‑class accuracy on CIFAR‑100')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (CIFAR‑100)')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
