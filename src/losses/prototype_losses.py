import torch
import torch.nn.functional as F

def contrastive_prototype_loss(features, labels, prototypes, temperature=0.07):
    """
    对比学习损失,增强原型质量
    参考论文: "Supervised Contrastive Learning"

    Args:
        features: (N, D) 特征
        labels: (N,) 标签
        prototypes: dict {class_id: prototype_tensor}
        temperature: 温度参数

    Returns:
        loss: 对比学习损失
    """
    features = F.normalize(features, dim=1)

    # 构建原型矩阵
    unique_labels = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[label] for label in unique_labels])
    proto_matrix = F.normalize(proto_matrix, dim=1)

    # 计算相似度
    logits = features @ proto_matrix.T / temperature  # (N, num_classes)

    # 构建标签索引
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    targets = torch.tensor([label_to_idx[label.item()] for label in labels],
                          device=features.device)

    # 对比学习损失
    loss = F.cross_entropy(logits, targets)

    return loss

def compute_prototype_diversity_loss(prototypes_list):
    """
    计算原型多样性损失,鼓励同一类的多个原型保持多样性

    Args:
        prototypes_list: list of tensors, 每个类的多个原型

    Returns:
        diversity_loss: 多样性损失 (越小越好,表示原型越多样)
    """
    if len(prototypes_list) <= 1:
        return torch.tensor(0.0, device=prototypes_list[0].device if prototypes_list else 'cpu')

    # 归一化原型
    prototypes = torch.stack(prototypes_list)
    prototypes = F.normalize(prototypes, dim=1)

    # 计算原型间的相似度
    similarity_matrix = prototypes @ prototypes.T

    # 去除对角线 (自己和自己的相似度)
    mask = ~torch.eye(len(prototypes_list), dtype=torch.bool, device=similarity_matrix.device)
    similarities = similarity_matrix[mask]

    # 多样性损失 = 平均相似度 (我们希望这个值小,即原型不相似)
    diversity_loss = similarities.mean()

    return diversity_loss

def contrastive_enhancement_loss(features, labels, prototypes, margin=0.5, temperature=0.07, hard_mining=True):
    """
    Phase 3: 对比学习增强损失
    增强类间距离,减少类内方差

    Args:
        features: (N, D) 特征
        labels: (N,) 标签
        prototypes: dict {class_id: prototype_tensor}
        margin: 对比学习边界
        temperature: 温度参数
        hard_mining: 是否使用困难样本挖掘

    Returns:
        loss: 对比学习增强损失
    """
    features = F.normalize(features, dim=1)

    # 构建原型矩阵
    unique_labels = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[label] for label in unique_labels])
    proto_matrix = F.normalize(proto_matrix, dim=1)

    # 计算相似度
    logits = features @ proto_matrix.T / temperature  # (N, num_classes)

    # 构建标签索引
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    targets = torch.tensor([label_to_idx[label.item()] for label in labels],
                          device=features.device)

    # 基础对比损失
    base_loss = F.cross_entropy(logits, targets)

    # 困难样本挖掘 (可选)
    if hard_mining:
        # 计算每个样本的置信度
        probs = F.softmax(logits, dim=1)
        confidences = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 选择低置信度样本 (困难样本)
        hard_mask = confidences < 0.7
        if hard_mask.sum() > 0:
            hard_logits = logits[hard_mask]
            hard_targets = targets[hard_mask]
            hard_loss = F.cross_entropy(hard_logits, hard_targets)
            base_loss = 0.7 * base_loss + 0.3 * hard_loss

    # 边界损失 (增强类间距离)
    margin_loss = 0.0
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            # 计算类i和类j之间的距离
            dist_ij = 1.0 - (proto_matrix[i] @ proto_matrix[j])
            # 如果距离小于margin,增加损失
            if dist_ij < margin:
                margin_loss += (margin - dist_ij) ** 2

    if len(unique_labels) > 1:
        margin_loss = margin_loss / (len(unique_labels) * (len(unique_labels) - 1) / 2)

    # 组合损失
    total_loss = base_loss + 0.5 * margin_loss

    return total_loss
