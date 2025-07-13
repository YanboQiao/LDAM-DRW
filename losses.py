import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss (用于长尾分类)"""
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        """
        参数:
        - cls_num_list: 各类别样本数量的列表，用于计算每类的margin大小
        - max_m: 最大margin值（长尾情况下一般为0.5）
        - weight: 每个类别的可选权重（tensor类型），用于再平衡（默认为None）
        - s: logits缩放因子（默认30，提升损失梯度的可分性）
        """
        super(LDAMLoss, self).__init__()
        # 计算每个类别的 margin 大小 m_i = max_m * (1/√n_i) / max_j(1/√n_j)
        cls_num_array = np.array(cls_num_list)
        cls_num_array = np.maximum(cls_num_array, 1)  # 防止除零
        m_list = 1.0 / np.sqrt(cls_num_array)
        m_list = m_list * (max_m / np.max(m_list))
        # 将 margin 列表注册为buffer，随模型保存/移动
        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.s = s  # logits缩放系数
        if weight is not None:
            # 注册类别权重buffer（如使用DRW后期阶段的1/n_i权重）
            self.register_buffer('class_weight', torch.FloatTensor(weight))
        else:
            self.class_weight = None

    def forward(self, logits, target):
        """
        前向计算:
        - logits: 模型输出的 logits 张量，形状 [N, C]
        - target: 对应的标签张量，形状 [N]
        返回:
        - 根据LDAM调整后的交叉熵损失（标量）
        """
        # 从buffer获取 margin 列表
        m_list = self.m_list
        # 根据 target 提取每个样本对应类别的 margin
        margins = m_list[target]              # 张量形状 [N]
        # 创建 one-hot 掩码，将 margin 只应用到正确分类的 logit 上
        one_hot = torch.zeros_like(logits, dtype=torch.uint8)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        # 调整 logits：对于真实类别的位置减去 margin*s
        logits_adjusted = logits - one_hot * margins.unsqueeze(1) * self.s
        # 计算交叉熵损失（如果提供了权重则使用）
        weight = self.class_weight if hasattr(self, 'class_weight') else None
        loss = F.cross_entropy(logits_adjusted, target, weight=weight)
        return loss

class PaCoLoss(nn.Module):
    """Parametric Contrastive Loss (结合交叉熵和对比学习的损失)"""
    def __init__(self, cls_num_list, alpha=1.0, beta=1.0, gamma=1.0, 
                 supt=1.0, temperature=0.2, base_temperature=0.2, K=128):
        """
        参数:
        - cls_num_list: 各类别样本数量列表，用于计算类别权重
        - alpha, beta: 分别为监督交叉熵损失和对比损失项的权重系数
        - gamma: Focal Loss中的γ参数（调节易分类样本的损失，默认1.0）
        - supt: （可选）监督损失缩放系数，默认1.0
        - temperature: 对比损失温度参数τ，默认0.2
        - base_temperature: 对比损失基准温度，默认0.2（一般与τ相同）
        - K: （可选）参数，对应论文中队列长度或其他常数超参数，默认128
        """
        super(PaCoLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.temperature = temperature
        self.base_temperature = base_temperature
        # 计算每类的权重（采用类频次反比或Effective Number策略）
        cls_num_array = np.array(cls_num_list)
        cls_num_array = np.maximum(cls_num_array, 1)
        if beta < 1.0:
            # Effective number 权重: w_i = (1 - beta) / (1 - beta^n_i)
            effective_num = 1.0 - np.power(beta, cls_num_array)
            weights = (1.0 - beta) / effective_num
        else:
            # 若 beta=1，退化为简单的 1/n_i 权重
            weights = 1.0 / cls_num_array
        weights = weights / np.sum(weights) * len(cls_num_array)  # 归一化权重总和=类别数
        # 注册类别权重为 buffer
        self.register_buffer('class_weight', torch.FloatTensor(weights))

    def forward(self, features, sup_logits, labels):
        """
        前向计算:
        - features: 模型输出的特征向量（未经过最后全连接分类层），形状 [N, d]
        - sup_logits: 模型输出的分类 logits（经过全连接层），形状 [N, C]
        - labels: 样本标签张量，形状 [N]
        返回:
        - PaCo组合损失值（标量）
        """
        # ------- 1. 监督交叉熵损失（含Focal调整和类别重权） -------
        log_probs = F.log_softmax(sup_logits, dim=1)           # 计算每类的对数概率
        log_prob_target = log_probs[torch.arange(labels.size(0)), labels]  # 取正确类别的log概率 [N]
        # 基础交叉熵损失项
        ce_loss = - log_prob_target
        # 如果 gamma>0，则应用 Focal Loss 的调制因子
        if self.gamma > 0:
            # 计算 pt（真实类别的概率），根据 pt 调整损失
            pt = log_prob_target.exp()
            ce_loss = ((1 - pt) ** self.gamma) * ce_loss
        # 乘以类别权重
        ce_loss = self.class_weight[labels] * ce_loss
        # 按照加权后样本总权值求平均
        ce_loss = ce_loss.sum() / self.class_weight[labels].sum()

        # ------- 2. 对比学习损失（Supervised Contrastive Loss） -------
        # 特征向量归一化（提高数值稳定性）
        features = F.normalize(features, dim=1)
        # 计算两两样本之间的相似度矩阵 (cosine similarity / 温度)
        sim_matrix = torch.div(features @ features.t(), self.temperature)  # [N, N]
        # 为数值稳定性减去每行最大值
        logits = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        # 构造正样本掩码：mask_{ij}=1表示样本i与样本j属于同一类别
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # 除去自身对比（对角线位置不参与计算）
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask
        # 计算对比项的 log_prob
        exp_logits = torch.exp(logits) * logits_mask  # [N, N]
        log_prob_contrast = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        # 计算每个样本对所有正样本的平均 log-likelihood 
        pos_per_sample = mask.sum(1)  # 每个样本的正样本数量
        mean_log_prob_pos = (mask * log_prob_contrast).sum(1) / (pos_per_sample + 1e-12)
        contrast_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos  # [N]
        contrast_loss[pos_per_sample == 0] = 0  # 若某样本在批次中无正样本，则损失置0
        # 按类别权重加权每个样本的对比损失
        contrast_loss = self.class_weight[labels] * contrast_loss
        contrast_loss = contrast_loss.sum() / self.class_weight[labels].sum()

        # ------- 3. 组合总损失 -------
        total_loss = self.alpha * ce_loss + self.beta * contrast_loss
        return total_loss