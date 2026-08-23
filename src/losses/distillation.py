import torch
import torch.nn.functional as F

def multi_layer_distillation_loss(current_features, old_features, temperature=4.0):
    """
    多层次蒸馏损失: 保持中间层特征与旧模型的一致性

    Args:
        current_features: 当前模型的特征 [batch, hidden_dim]
        old_features: 旧模型的特征 [batch, hidden_dim]
        temperature: 蒸馏温度

    Returns:
        蒸馏损失
    """
    # 归一化特征
    current_norm = F.normalize(current_features, p=2, dim=1)
    old_norm = F.normalize(old_features, p=2, dim=1)

    # 计算相似度矩阵
    similarity = current_norm @ old_norm.T / temperature

    # 目标: 对角线上的相似度应该最高 (同一样本)
    batch_size = current_features.shape[0]
    target = torch.arange(batch_size, device=current_features.device)

    # 交叉熵损失
    loss = F.cross_entropy(similarity, target)

    return loss

def compute_multi_layer_distillation(current_dict, old_dict, temperature=4.0, weights=None):
    """
    计算多层次蒸馏总损失

    Args:
        current_dict: 当前模型的多层特征 {'layer_3': [...], 'layer_6': [...], ...}
        old_dict: 旧模型的多层特征
        temperature: 蒸馏温度
        weights: 各层的权重 (如果为None,使用均匀权重)

    Returns:
        总蒸馏损失
    """
    if not current_dict or not old_dict:
        # 返回零张量,使用第一个可用的设备
        device = next(iter(current_dict.values())).device if current_dict else 'cpu'
        return torch.tensor(0.0, device=device)

    total_loss = 0.0
    num_layers = 0

    if weights is None:
        # 默认权重: 后层权重更高
        weights = {
            'layer_3': 0.1,
            'layer_6': 0.2,
            'layer_9': 0.3,
            'layer_11': 0.4
        }

    for layer_name in current_dict.keys():
        if layer_name in old_dict:
            layer_loss = multi_layer_distillation_loss(
                current_dict[layer_name],
                old_dict[layer_name],
                temperature=temperature
            )
            weight = weights.get(layer_name, 1.0 / len(current_dict))
            total_loss += weight * layer_loss
            num_layers += 1

    if num_layers > 0:
        return total_loss / num_layers
    else:
        device = next(iter(current_dict.values())).device if current_dict else 'cpu'
        return torch.tensor(0.0, device=device)

def procrustes_alignment_loss(current_features, old_features, alignment_matrix=None):
    """
    Procrustes对齐损失: 鼓励当前特征与对齐后的旧特征相近

    Args:
        current_features: 当前模型特征 [batch, hidden_dim]
        old_features: 旧模型特征 [batch, hidden_dim]
        alignment_matrix: 预计算的对齐矩阵 (如果为None,则计算)

    Returns:
        对齐损失
    """
    try:
        if alignment_matrix is None:
            # 计算对齐矩阵
            from src.utils.alignment import procrustes_alignment
            alignment_matrix = procrustes_alignment(old_features, current_features)

        # 对齐旧特征
        aligned_old_features = old_features @ alignment_matrix

        # 计算对齐损失 (MSE)
        loss = F.mse_loss(current_features, aligned_old_features)

        return loss
    except Exception as e:
        # 如果计算失败,返回零损失
        print(f"[WARNING] Procrustes alignment loss failed: {e}")
        return torch.tensor(0.0, device=current_features.device, requires_grad=True)

def compute_procrustes_alignment_loss(current_features, old_features,
                                     alignment_matrices=None, temperature=4.0):
    """
    计算Procrustes对齐总损失 (Phase 2核心)

    Args:
        current_features: 当前特征 [batch, hidden_dim]
        old_features: 旧特征 [batch, hidden_dim]
        alignment_matrices: 预计算的对齐矩阵字典
        temperature: 温度参数

    Returns:
        对齐损失
    """
    if old_features is None:
        return torch.tensor(0.0, device=current_features.device)

    # 方法1: 直接MSE损失
    alignment_loss = procrustes_alignment_loss(current_features, old_features)

    # 方法2: 相似度匹配 (可选)
    current_norm = F.normalize(current_features, p=2, dim=1)
    old_norm = F.normalize(old_features, p=2, dim=1)

    # 计算相似度矩阵
    similarity = current_norm @ old_norm.T / temperature

    # 目标: 对角线上的相似度应该最高
    batch_size = current_features.shape[0]
    target = torch.arange(batch_size, device=current_features.device)

    # 相似度损失
    similarity_loss = F.cross_entropy(similarity, target)

    # 组合损失 (50% MSE + 50% 相似度)
    total_loss = 0.5 * alignment_loss + 0.5 * similarity_loss

    return total_loss
