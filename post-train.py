import argparse
import os
import random
import time
import warnings
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

import ldamModels
from pacoUtils import *

def eval_and_pick_worst(val_loader, model, criterion) -> List[int]:
    """评估一次，返回分类准确率最低的 20% 类别 id 列表"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for img, target in val_loader:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(img)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    cf = confusion_matrix(all_targets, all_preds).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    num_cls = len(cls_acc)
    k = max(1, int(num_cls * 0.2))
    worst_classes = np.argsort(cls_acc)[:k].tolist()
    return worst_classes
