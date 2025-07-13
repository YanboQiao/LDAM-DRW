# losses.py  ── 2025-07-13 修订
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------ #
# 1. LDAMLoss
# ------------------------------------------------------------------ #
class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1. / np.sqrt(np.maximum(cls_num_list, 1))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                self.register_buffer('class_weight', weight.float())
            else:
                self.register_buffer('class_weight',
                                     torch.tensor(weight, dtype=torch.float32))
        else:
            self.class_weight = None

    def forward(self, logits, target):
        margins = self.m_list[target]
        one_hot = torch.zeros_like(logits, dtype=torch.uint8)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        logits_adj = logits - one_hot * margins.unsqueeze(1) * self.s
        weight = getattr(self, 'class_weight', None)
        return F.cross_entropy(logits_adj, target, weight=weight)

# ------------------------------------------------------------------ #
# 2. PaCoLoss  (单 GPU 版，带 base_temperature 兼容形参)
# ------------------------------------------------------------------ #
class PaCoLoss(nn.Module):
    """
    关键实现点：
      • learnable class centers C  (d × C)
      • recent negatives queue  (K × d)
      • sup_logits + log p_c / supt  拼入对比矩阵左侧
      • α / β / γ 掩码
    """
    def __init__(self,
                 cls_num_list,
                 feat_dim=64,
                 num_classes=None,
                 K=128,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 supt=1.0,
                 temperature=0.1,
                 base_temperature=None):          # ← 新增, 仅为兼容
        super().__init__()
        if num_classes is None:
            num_classes = len(cls_num_list)
        self.num_classes = num_classes
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.temperature = temperature
        # 若调用端传入 base_temperature，则覆盖 temperature 用于公式
        self.base_temperature = base_temperature or temperature

        # 先验 log p_c
        cls_arr = np.maximum(np.array(cls_num_list, dtype=np.float32), 1)
        prior = cls_arr / cls_arr.sum()
        self.register_buffer('log_prior', torch.log(torch.tensor(prior)))

        # learnable class centers
        self.C = nn.Parameter(torch.randn(feat_dim, self.num_classes) * 0.01)

        # negatives queue
        self.register_buffer('queue', torch.randn(K, feat_dim))
        self.queue_ptr = 0

        # class-balanced weight (1/n_i)
        inv_freq = 1. / cls_arr
        inv_freq = inv_freq / inv_freq.sum() * self.num_classes
        self.register_buffer('class_weight', torch.tensor(inv_freq, dtype=torch.float32))

    # ---------- helper ----------
    @torch.no_grad()
    def _enqueue(self, feats):
        k = self.queue.shape[0]
        b = min(feats.size(0), k)
        idx = int(self.queue_ptr)
        end = idx + b
        if end <= k:
            self.queue[idx:end] = feats[:b]
        else:
            first = k - idx
            self.queue[idx:] = feats[:first]
            self.queue[:b - first] = feats[first:b]
        self.queue_ptr = (idx + b) % k

    # ---------- forward ----------
    def forward(self, feats, logits, labels):
        """
        feats  : (N, d) 背骨特征
        logits : (N, C) 分类器输出
        labels : (N,)
        """
        device = feats.device
        N, d = feats.size()

        # ---- 1) supervised CE branch ----
        sup_logits = (logits + self.log_prior) / self.supt
        ce_loss = F.cross_entropy(sup_logits, labels, weight=self.class_weight)

        # ---- 2) contrastive branch ----
        feats = F.normalize(feats, dim=1)
        self._enqueue(feats.detach())                       # 更新负样本队列

        pos_centers = self.C[:, labels].t()                 # (N,d)
        all_centers = F.normalize(self.C, dim=0).t()        # (C,d)
        neg_center_sim = feats @ all_centers.t()            # (N,C)
        neg_queue_sim = feats @ self.queue.t()              # (N,K)

        left_logits = sup_logits.detach()                   # (N,C)
        right_logits = torch.cat([neg_center_sim, neg_queue_sim], dim=1)
        anchor_dot = torch.cat([left_logits, right_logits], dim=1) / self.temperature

        # mask  (β·one-hot | α·正 | 0·负)
        mask_sup = F.one_hot(labels, self.num_classes).float()
        mask_cent = (labels[:, None] == torch.arange(self.num_classes,
                                                    device=device)[None, :]).float()
        mask = torch.cat([mask_sup * self.beta,
                          mask_cent * self.alpha,
                          torch.zeros(N, self.K, device=device)], dim=1)

        # logits mask  (对负样本乘 γ)
        logits_mask = torch.cat([torch.ones_like(mask_sup),
                                 torch.ones_like(mask_cent),
                                 torch.full((N, self.K), self.gamma, device=device)], dim=1)

        # 数值稳定
        anchor_dot = anchor_dot - anchor_dot.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(anchor_dot) * logits_mask
        log_prob = anchor_dot - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        contrast_loss = -mean_log_prob_pos.mean() * (self.temperature / self.base_temperature)

        return ce_loss + contrast_loss

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)