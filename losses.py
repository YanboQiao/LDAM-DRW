# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1. / np.sqrt(np.maximum(cls_num_list, 1))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.class_weight = None
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                self.register_buffer('class_weight', weight.float())
            else:
                self.register_buffer('class_weight',
                                     torch.tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        margins = self.m_list[target]
        one_hot = torch.zeros_like(logits, dtype=torch.uint8)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        logits_adj = logits - one_hot * margins.unsqueeze(1) * self.s
        return F.cross_entropy(logits_adj, target, weight=self.class_weight)


class PaCoLoss(nn.Module):
    def __init__(self,
                 cls_num_list,
                 feat_dim: int = 64,
                 num_classes: int | None = None,
                 K: int = 8192,
                 alpha: float = 0.02,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 supt: float = 1.0,
                 temperature: float = 0.05,
                 base_temperature: float | None = None,
                 lambda_intra: float = 1.0,
                 lambda_inter_mean: float = 0.1,
                 lambda_inter_min: float = 0.05,
                 lambda_fisher: float = 0.1,
                 eps: float = 1e-6):
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
        self.base_temperature = base_temperature or temperature
        self.lambda_intra = lambda_intra
        self.lambda_inter_mean = lambda_inter_mean
        self.lambda_inter_min = lambda_inter_min
        self.lambda_fisher = lambda_fisher
        self.eps = eps

        cls_arr = np.maximum(np.array(cls_num_list, dtype=np.float32), 1)
        prior = cls_arr / cls_arr.sum()
        self.register_buffer('log_prior', torch.log(torch.tensor(prior)))

        self.C = nn.Parameter(torch.randn(feat_dim, self.num_classes) * 0.01)
        self.register_buffer('queue', torch.randn(K, feat_dim))
        self.queue_ptr = 0

        inv_freq = 1. / cls_arr
        inv_freq = inv_freq / inv_freq.sum() * self.num_classes
        self.register_buffer('class_weight', torch.tensor(inv_freq, dtype=torch.float32))

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

    def forward(self, feats, logits, labels):
        device = feats.device
        N, _ = feats.size()

        # ---------- supervised branch ----------
        sup_logits = (logits + self.log_prior) / self.supt
        ce_loss = F.cross_entropy(sup_logits, labels, weight=self.class_weight)

        # ---------- contrastive branch ----------
        feats = F.normalize(feats, dim=1)
        self._enqueue(feats.detach())

        centers = F.normalize(self.C, dim=0).t()                 # (C,d)
        neg_center_sim = feats @ centers.t()                     # (N,C)
        neg_queue_sim  = feats @ self.queue.t()                  # (N,K)

        left_logits  = sup_logits.detach()                       # (N,C)
        right_logits = torch.cat([neg_center_sim, neg_queue_sim], dim=1)
        anchor_dot   = torch.cat([left_logits, right_logits], dim=1) / self.temperature

        mask_sup  = F.one_hot(labels, self.num_classes).float()
        mask_cent = (labels.unsqueeze(1) == torch.arange(
                     self.num_classes, device=device).unsqueeze(0)).float()
        mask = torch.cat([
            mask_sup * self.beta,
            mask_cent * self.alpha,
            torch.zeros(N, self.K, device=device)
        ], dim=1)

        logits_mask = torch.cat([
            torch.ones_like(mask_sup),
            torch.ones_like(mask_cent),
            torch.full((N, self.K), self.gamma, device=device)
        ], dim=1)

        anchor_dot = anchor_dot - anchor_dot.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(anchor_dot) * logits_mask
        log_prob   = anchor_dot - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        contrast_loss = -mean_log_prob_pos.mean() * (self.temperature / self.base_temperature)

        # ---------- representation regularizers ----------
        # 1) inter-class distances (avoid in-place ops)
        dist_mat   = torch.cdist(centers, centers, p=2)          # (C,C)
        eye_mask   = torch.eye(self.num_classes, device=device, dtype=torch.bool)
        dist_no_dg = torch.where(eye_mask,
                                 torch.full_like(dist_mat, 1e6),
                                 dist_mat)
        inter_min  = dist_no_dg.min()
        inter_mean = dist_no_dg[~eye_mask].mean()

        # 2) intra-class mean square distance
        intra_d = (feats - centers[labels]).pow(2).sum(1).mean()

        # 3) Fisher ratio
        mu_g   = feats.mean(0, keepdim=True)
        cls_cnt = torch.bincount(labels, minlength=self.num_classes).float().to(device)
        S_W = (feats - centers[labels]).pow(2).sum()
        S_B = ((centers - mu_g).pow(2).sum(1) * cls_cnt).sum()
        fisher = S_B / (S_W + self.eps)

        reg = (self.lambda_intra       * intra_d +
               self.lambda_inter_mean  / (inter_mean + self.eps) +
               self.lambda_inter_min   / (inter_min  + self.eps) +
               self.lambda_fisher      / (fisher     + self.eps))

        return ce_loss + contrast_loss + reg


def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none',
                                          weight=self.weight), self.gamma)
