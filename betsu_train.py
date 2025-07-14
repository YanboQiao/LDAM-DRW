# train.py
import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import LDAMLoss, PaCoLoss
from balanced_augmentation import BalancedAugmentedCIFAR

# -------------------- CLI -------------------- #
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)
parser = argparse.ArgumentParser(description='PaCo‑LDAM two‑stage training')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('-a', '--arch', default='resnet32', choices=model_names)
parser.add_argument('--imb_type', default='exp')
parser.add_argument('--imb_factor', default=0.01, type=float)
parser.add_argument('--train_rule', default='None')
parser.add_argument('--rand_number', default=0, type=int)
parser.add_argument('--exp_str', default='0')
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--wd', default=2e-4, type=float)
parser.add_argument('--print-freq', default=10, type=int)
parser.add_argument('--resume', default='')
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--gpu', default=None, type=int)
parser.add_argument('--root_log', default='log')
parser.add_argument('--root_model', default='checkpoint')

best_acc1 = 0


# -------------------- helper -------------------- #
def _is_classifier(name):
    return any(k in name for k in ['fc', 'classifier', 'linear', 'head'])


def set_phase(model, phase):
    """
    phase ∈ {'rep', 'cls'}.
      rep : freeze classifier
      cls : freeze backbone
    """
    if isinstance(model, nn.DataParallel):
        model = model.module
    for n, p in model.named_parameters():
        p.requires_grad = not _is_classifier(n) if phase == 'rep' else _is_classifier(n)


def build_optimizer(model, args):
    """依据当前可训练参数构建优化器"""
    return torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd      # ← 修正字段名
    )



def lr_scheduler(epoch_rel):
    """epoch_rel starts from 0 inside its own 100‑epoch stage"""
    e = epoch_rel + 1
    if e <= 5:
        return 0.02 + (0.1 - 0.02) * (e - 1) / 4
    if e <= 30:
        return 0.1
    if e <= 50:
        return 0.05
    if e <= 70:
        return 0.01
    if e <= 90:
        return 0.001
    return 0.0001


# ---------- rep_validate (多指标版本) ----------
@torch.no_grad()
def rep_validate(backbone, train_loader, val_loader, device):
    """
    评估表征器质量，返回多项指标:
        proto_acc   : 基于类别原型的 top-1 准确率 (%)           – 越高越好
        intra_d     : 类内均方距离 (类内散度)                    – 越低越好
        inter_min   : 各类中心最小距离                          – 越高越好
        inter_mean  : 各类中心平均距离                          – 越高越好
        fisher      : Fisher 判别率 = S_B / S_W                 – 越高越好

    说明:
        • 原型 μ_c 取自训练集特征均值，可直接替代 PaCo learnable center 做评估。
        • 该函数只 forward 一遍 train_loader、val_loader，
          计算量与原实现基本一致，不依赖额外库。
        • 返回 dict，方便在训练循环里记录到 tensorboard，
          或将其中指标(如 intra_d, fisher) 加权写入损失函数。
    """
    backbone.eval()

    # -------- 1) 收集训练特征，用于估计类中心 / 统计 --------
    feats_list, labels_list = [], []
    for x, y in train_loader:
        feats_list.append(backbone(x.to(device)).cpu())
        labels_list.append(y.cpu())
    feats  = torch.cat(feats_list, 0)          # (N, d)
    labels = torch.cat(labels_list, 0)         # (N, )

    num_classes = int(labels.max()) + 1
    feat_dim    = feats.size(1)

    # 类中心 μ_c
    proto = torch.zeros(num_classes, feat_dim)
    cnt   = torch.zeros(num_classes)
    proto.index_add_(0, labels, feats)
    cnt.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    proto = proto / cnt.unsqueeze(1).clamp_min(1.0)     # (C, d)

    # -------- 2) 类内散度 (S_W) --------
    intra_d2 = ((feats - proto[labels]) ** 2).sum(1)    # 每样本的平方距离
    intra_d  = intra_d2.mean().item()                   # 均值
    S_W      = intra_d2.sum().item()                    # 总和

    # -------- 3) 类间散度 / 距离 --------
    dist_mat = torch.cdist(proto, proto, p=2)           # (C, C) pairwise L2
    mask     = torch.eye(num_classes, dtype=torch.bool)
    dist_mat[mask] = float('inf')
    inter_min  = dist_mat.min().item()
    inter_mean = dist_mat[~mask].mean().item()

    # Fisher 判别率 S_B / S_W
    mu_g   = feats.mean(0, keepdim=True)                # 全局均值
    sb_vec = ((proto - mu_g) ** 2).sum(1) * cnt         # 加权类间散布
    S_B    = sb_vec.sum().item()
    fisher = S_B / (S_W + 1e-12)

    # -------- 4) 使用原型做 top-1 分类评估 --------
    correct, total = 0, 0
    for x, y in val_loader:
        f   = backbone(x.to(device)).cpu()              # (B,d)
        logit = f @ proto.t()                           # 余弦/点积相似度
        pred  = logit.argmax(1)
        correct += (pred == y.cpu()).sum().item()
        total   += y.size(0)
    proto_acc = 100. * correct / total if total else 0.

    return {
        'proto_acc':  round(proto_acc, 4),
        'intra_d':    round(intra_d,   6),
        'inter_min':  round(inter_min, 6),
        'inter_mean': round(inter_mean,6),
        'fisher':     round(fisher,    6),
    }

# -------------------- main -------------------- #
def main():
    args = parser.parse_args()
    args.store_name = '_'.join([
        args.dataset, args.arch, 'PaCo‑LDAM', args.train_rule,
        args.imb_type, str(args.imb_factor), args.exp_str
    ])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if torch.cuda.is_available():
        args.device = torch.device('cuda' if args.gpu is None else f'cuda:{args.gpu}')
    elif torch.backends.mps.is_available():
        args.device = torch.device('mps')
    else:
        args.device = torch.device('cpu')

    main_worker(args)


def main_worker(args):
    global best_acc1
    print(f"=> creating model '{args.arch}'")
    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=True)

    if args.device.type == 'cuda':
        if args.gpu is None:
            model = nn.DataParallel(model).cuda()
        else:
            torch.cuda.set_device(args.device)
            model = model.cuda()
    else:
        model = model.to(args.device)

    # ------------ data ------------ #
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    DS = IMBALANCECIFAR10 if args.dataset == 'cifar10' else IMBALANCECIFAR100
    train_set = DS('./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                   rand_number=args.rand_number, train=True, download=True,
                   transform=tf_train)
    val_set = (datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100)(
        './data', train=False, download=True, transform=tf_val)

    cls_num_list = train_set.get_cls_num_list()
    paco_loss = PaCoLoss(cls_num_list, temperature=0.2, base_temperature=0.2).to(args.device)
    ldam_loss = LDAMLoss(cls_num_list, max_m=0.5, s=30).to(args.device)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    log_tr = open(os.path.join(args.root_log, args.store_name, 'train.csv'), 'w')
    log_te = open(os.path.join(args.root_log, args.store_name, 'test.csv'), 'w')
    tf_writer = SummaryWriter(os.path.join(args.root_log, args.store_name))

    # ------------ phase‑0 (representation) ------------ #
    set_phase(model, 'rep')
    optimizer = build_optimizer(model, args)

    for epoch in range(args.epochs):
        phase = 'rep' if epoch < 100 else 'cls'
        epoch_rel = epoch if phase == 'rep' else epoch - 100
        lr = lr_scheduler(epoch_rel)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if epoch == 100:                # switch to classification stage
            set_phase(model, 'cls')
            optimizer = build_optimizer(model, args)
            # 重计算损失以引入 DRW 权重后续再覆盖
            ldam_loss = LDAMLoss(cls_num_list, max_m=0.5, s=30).to(args.device)

        # BalancedAug from epoch 50 (index 50)
        if epoch == 0:
            print(">>> BalancedAugmentedCIFAR enabled")
            train_set = BalancedAugmentedCIFAR(train_set, cls_num_list)
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True,
            )
            cls_num_list = train_set.get_cls_num_list()
            paco_loss = PaCoLoss(cls_num_list, temperature=0.2,
                                 base_temperature=0.2).to(args.device)

        # DRW weights (epoch 180‑199): only affect LDAM
        if epoch >= 180:
            beta = 0.9999
            eff = 1. - np.power(beta, cls_num_list)
            w = (1. - beta) / eff
            w = w / w.sum() * len(cls_num_list)
            ldam_loss = LDAMLoss(
                cls_num_list, max_m=0.5, s=30,
                weight=torch.FloatTensor(w).to(args.device)
            )

        criterion = paco_loss if phase == 'rep' else ldam_loss
        use_paco = phase == 'rep'

        print(f"\nEpoch {epoch:03d} | Phase: {'PaCo‑Rep' if use_paco else 'LDAM‑Cls'} "
              f"| LR: {lr:.5f} | Sampler: {args.train_rule}")

        train_one_epoch(
            train_loader, model, criterion, optimizer, epoch,
            args, log_tr, tf_writer, use_paco,
            print_acc=(phase == 'cls')
        )

        if phase == 'rep':
            # 多指标评估
            rep_metrics = rep_validate(model, train_loader, val_loader, args.device)

            # 1) 记录到 TensorBoard：一次写入全部指标
            tf_writer.add_scalars('rep_metrics', rep_metrics, epoch)

            # 2) 打印关键信息到终端
            print(
                "Rep-stage metrics | "
                f"proto_acc: {rep_metrics['proto_acc']:.2f}%  "
                f"intra_d: {rep_metrics['intra_d']:.4f}  "
                f"inter_min: {rep_metrics['inter_min']:.4f}  "
                f"fisher: {rep_metrics['fisher']:.4f}"
            )
            # 3) 兼容旧日志字段（只存 prototype top-1）
            tf_writer.add_scalar('rep_acc/val', rep_metrics['proto_acc'], epoch)
        else:
            acc1 = validate(val_loader, model, ldam_loss, epoch,
                            args, log_te, tf_writer)
            best_acc1 = max(best_acc1, acc1)
            tf_writer.add_scalar('acc/best_top1', best_acc1, epoch)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, acc1 == best_acc1)


# -------------------- train / validate -------------------- #
def train_one_epoch(loader, model, criterion, optimizer, epoch,
                    args, log_f, tf_writer, use_paco, print_acc=True):
    meter_t = AverageMeter('Time', ':6.3f')
    meter_d = AverageMeter('Data', ':6.3f')
    meter_l = AverageMeter('Loss', ':.4e')
    meter_a1 = AverageMeter('Acc@1', ':6.2f')
    meter_a5 = AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()
    for i, (inp, tgt) in enumerate(loader):
        meter_d.update(time.time() - end)
        inp, tgt = inp.to(args.device), tgt.to(args.device)

        if use_paco:
            logit, feat = model(inp, return_feat=True)
            loss = criterion(feat, logit, tgt)
            out = logit.detach()
        else:
            out = model(inp)
            loss = criterion(out, tgt)

        if print_acc:
            acc1, acc5 = accuracy(out, tgt, topk=(1, 5))
            meter_a1.update(acc1[0], inp.size(0))
            meter_a5.update(acc5[0], inp.size(0))
        meter_l.update(loss.item(), inp.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter_t.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if print_acc:
                msg = ('Epoch: [{0}][{1}/{2}] lr {lr:.5f}\t'
                       'Time {t.val:.3f} ({t.avg:.3f})\t'
                       'Data {d.val:.3f} ({d.avg:.3f})\t'
                       'Loss {l.val:.4f} ({l.avg:.4f})\t'
                       'Acc@1 {a1.val:.2f} ({a1.avg:.2f})'.format(
                    epoch, i, len(loader), lr=optimizer.param_groups[0]['lr'],
                    t=meter_t, d=meter_d, l=meter_l, a1=meter_a1))
            else:
                msg = ('Epoch: [{0}][{1}/{2}] lr {lr:.5f}\t'
                       'Time {t.val:.3f} ({t.avg:.3f})\t'
                       'Data {d.val:.3f} ({d.avg:.3f})\t'
                       'Loss {l.val:.4f} ({l.avg:.4f})'.format(
                    epoch, i, len(loader), lr=optimizer.param_groups[0]['lr'],
                    t=meter_t, d=meter_d, l=meter_l))
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()

    tf_writer.add_scalar('loss/train', meter_l.avg, epoch)
    if print_acc:
        tf_writer.add_scalar('acc/train_top1', meter_a1.avg, epoch)


def validate(loader, model, criterion, epoch, args, log_f, tf_writer):
    meter_t = AverageMeter('Time', ':6.3f')
    meter_l = AverageMeter('Loss', ':.4e')
    meter_a1 = AverageMeter('Acc@1', ':6.2f')
    meter_a5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (inp, tgt) in enumerate(loader):
            inp, tgt = inp.to(args.device), tgt.to(args.device)
            out = model(inp)
            loss = criterion(out, tgt)

            acc1, acc5 = accuracy(out, tgt, topk=(1, 5))
            meter_l.update(loss.item(), inp.size(0))
            meter_a1.update(acc1[0], inp.size(0))
            meter_a5.update(acc5[0], inp.size(0))
            meter_t.update(time.time() - end)
            end = time.time()

    msg = f' * Val Acc@1 {meter_a1.avg:.3f}  Acc@5 {meter_a5.avg:.3f}'
    print(msg)
    log_f.write(msg + '\n')
    log_f.flush()

    tf_writer.add_scalar('loss/val', meter_l.avg, epoch)
    tf_writer.add_scalar('acc/val_top1', meter_a1.avg, epoch)
    return meter_a1.avg


if __name__ == '__main__':
    main()
