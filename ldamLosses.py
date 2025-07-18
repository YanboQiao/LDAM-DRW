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
        """
        兼容不同设备：
          • 保证 self.m_list 与 logits / target 在同一 device
        """
        device  = logits.device
        m_list  = self.m_list.to(device)          # ← 保证 buffer 在同一设备
        margins = m_list[target]

        # one-hot 与 logits 位于同一设备
        one_hot = torch.zeros_like(logits, dtype=torch.bool, device=device)
        one_hot.scatter_(1, target.view(-1, 1), True)

        logits_adj = logits - one_hot.float() * margins.unsqueeze(1) * self.s
        weight = getattr(self, 'class_weight', None)
        if weight is not None:
            weight = weight.to(device)
        return F.cross_entropy(logits_adj, target, weight=weight)