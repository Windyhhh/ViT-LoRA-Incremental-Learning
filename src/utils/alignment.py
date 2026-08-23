import torch
import torch.nn.functional as F
import numpy as np

def procrustes_alignment(source_features, target_features):
    """
    Procrustes对齐: 找到最优旋转矩阵R,使得 source @ R 最接近 target
    参考论文: "Geometric Prototype Alignment for Class-Incremental Learning"

    Args:
        source_features: (N, D) 源特征
        target_features: (N, D) 目标特征

    Returns:
        R: (D, D) 最优旋转矩阵
    """
    try:
        # 中心化
        source_mean = source_features.mean(dim=0, keepdim=True)
        target_mean = target_features.mean(dim=0, keepdim=True)

        source_centered = source_features - source_mean
        target_centered = target_features - target_mean

        # 计算协方差矩阵
        H = source_centered.T @ target_centered

        # 添加正则化以改善条件数
        H = H + 1e-6 * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

        # SVD分解 (使用full_matrices=False以提高稳定性)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)

        # 最优旋转矩阵
        R = Vh.T @ U.T

        # 确保是旋转矩阵 (det(R) = 1)
        if torch.det(R) < 0:
            Vh_corrected = Vh.clone()
            Vh_corrected[-1, :] *= -1
            R = Vh_corrected.T @ U.T

        return R

    except Exception as e:
        # 如果SVD失败,返回单位矩阵 (无对齐)
        print(f"[WARNING] Procrustes alignment failed: {e}")
        return torch.eye(source_features.shape[1], device=source_features.device, dtype=source_features.dtype)

def gradient_reassignment(gradients, importance_weights):
    """
    梯度重分配: 根据重要性权重重新分配梯度
    参考CL-LoRA论文

    Args:
        gradients: 原始梯度
        importance_weights: 重要性权重 (L2 norm of previous weights)

    Returns:
        reassigned_gradients: 重分配后的梯度
    """
    # 归一化重要性权重
    importance_weights = torch.softmax(importance_weights, dim=0)

    # 重分配梯度
    reassigned_gradients = gradients * importance_weights.unsqueeze(1)

    return reassigned_gradients

def prototype_self_calibration(model, features, labels, prototypes, old_prototypes=None,
                              iterations=5, momentum=0.9, threshold=0.95, device='cuda'):
    """
    Phase 4: 原型自校准
    在merged空间中重新计算原型,使用高置信度伪标签

    Args:
        model: 模型
        features: (N, D) 特征
        labels: (N,) 标签
        prototypes: dict {class_id: prototype_tensor}
        old_prototypes: 旧原型 (用于平滑)
        iterations: 校准迭代次数
        momentum: 校准动量
        threshold: 伪标签置信度阈值
        device: 设备

    Returns:
        calibrated_prototypes: 校准后的原型
    """
    features = F.normalize(features, dim=1)
    calibrated_prototypes = {k: v.clone() for k, v in prototypes.items()}

    for iteration in range(iterations):
        # 构建原型矩阵
        unique_labels = sorted(calibrated_prototypes.keys())
        proto_matrix = torch.stack([calibrated_prototypes[label] for label in unique_labels])
        proto_matrix = F.normalize(proto_matrix, dim=1)

        # 计算相似度
        logits = features @ proto_matrix.T  # (N, num_classes)
        probs = F.softmax(logits, dim=1)

        # 获取预测标签和置信度
        confidences, pred_labels = probs.max(dim=1)

        # 使用高置信度样本更新原型
        for class_id in unique_labels:
            # 找到该类的高置信度样本
            class_mask = pred_labels == unique_labels.index(class_id)
            high_conf_mask = confidences > threshold
            selected_mask = class_mask & high_conf_mask

            if selected_mask.sum() > 0:
                # 计算新原型
                new_proto = features[selected_mask].mean(dim=0)
                new_proto = F.normalize(new_proto.unsqueeze(0), p=2, dim=1).squeeze(0)

                # 使用动量平滑
                if old_prototypes is not None and class_id in old_prototypes:
                    old_proto = F.normalize(old_prototypes[class_id].unsqueeze(0), p=2, dim=1).squeeze(0)
                    new_proto = momentum * new_proto + (1 - momentum) * old_proto

                calibrated_prototypes[class_id] = new_proto.detach()

    return calibrated_prototypes
