import torch
import torch.nn.functional as F
from configs.config import Config

class PrototypeClassifier:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.prototypes = {}
        self.task_classes = {}
        self.multi_prototypes = {}  # 多原型存储
        self.alignment_matrices = {}  # Procrustes对齐矩阵

    def extract_prototypes(self, task_id, model, data_loader, class_ids, device):
        model.eval()
        class_features = {cid: [] for cid in class_ids}

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                features = model(images)
                features_normalized = F.normalize(features, p=2, dim=1)

                for class_id in class_ids:
                    mask = labels == class_id
                    if mask.any():
                        class_features[class_id].append(features_normalized[mask])

        for class_id, feat_list in class_features.items():
            if feat_list:
                all_features = torch.cat(feat_list, dim=0)
                prototype = all_features.mean(dim=0)
                prototype = F.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
                self.prototypes[class_id] = prototype.cpu()

        self.task_classes[task_id] = class_ids

    def refine_prototypes_in_merged_space(self, native_model, merged_model,
                                         data_loader, class_ids, device, refinement_iterations=None):
        """
        在融合空间中精炼原型,使用Procrustes对齐
        这是解决原型错位问题的关键!
        """
        if not Config.ENABLE_PROTOTYPE_REFINEMENT:
            return

        native_model.eval()
        merged_model.eval()

        # 收集native和merged空间的特征
        native_features_dict = {cid: [] for cid in class_ids}
        merged_features_dict = {cid: [] for cid in class_ids}

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)

                native_features = native_model(images)
                merged_features = merged_model(images)

                native_features_norm = F.normalize(native_features, p=2, dim=1)
                merged_features_norm = F.normalize(merged_features, p=2, dim=1)

                for class_id in class_ids:
                    mask = labels == class_id
                    if mask.any():
                        native_features_dict[class_id].append(native_features_norm[mask])
                        merged_features_dict[class_id].append(merged_features_norm[mask])

        # 对每个类别进行Procrustes对齐
        for class_id in class_ids:
            if native_features_dict[class_id] and merged_features_dict[class_id]:
                native_feats = torch.cat(native_features_dict[class_id], dim=0)
                merged_feats = torch.cat(merged_features_dict[class_id], dim=0)

                if Config.USE_PROCRUSTES_ALIGNMENT:
                    # 计算Procrustes对齐矩阵
                    from src.utils.alignment import procrustes_alignment
                    R = procrustes_alignment(native_feats, merged_feats)
                    self.alignment_matrices[class_id] = R.cpu()

                    # 对齐原型
                    if class_id in self.prototypes:
                        proto = self.prototypes[class_id].to(device)
                        aligned_proto = proto @ R
                        aligned_proto = F.normalize(aligned_proto.unsqueeze(0), p=2, dim=1).squeeze(0)
                        self.prototypes[class_id] = aligned_proto.cpu()

                # 迭代精炼
                iters = refinement_iterations if refinement_iterations is not None else Config.PROTOTYPE_REFINEMENT_ITERATIONS
                for iteration in range(iters):
                    # 在merged空间重新计算原型
                    new_proto = merged_feats.mean(dim=0)
                    new_proto = F.normalize(new_proto.unsqueeze(0), p=2, dim=1).squeeze(0)

                    # 动量更新
                    if class_id in self.prototypes:
                        old_proto = self.prototypes[class_id].to(device)
                        refined_proto = Config.REFINEMENT_MOMENTUM * old_proto + \
                                      (1 - Config.REFINEMENT_MOMENTUM) * new_proto
                        refined_proto = F.normalize(refined_proto.unsqueeze(0), p=2, dim=1).squeeze(0)
                        self.prototypes[class_id] = refined_proto.cpu()
                    else:
                        self.prototypes[class_id] = new_proto.cpu()

    def predict(self, features, device):
        if not self.prototypes:
            return torch.zeros(len(features), dtype=torch.long, device=device)

        features_normalized = F.normalize(features, p=2, dim=1)

        proto_ids = sorted(self.prototypes.keys())
        proto_matrix = torch.stack([self.prototypes[cid] for cid in proto_ids])
        proto_matrix = proto_matrix.to(device)

        similarities = features_normalized @ proto_matrix.T
        scaled_similarities = similarities / self.temperature

        max_indices = scaled_similarities.argmax(dim=1)
        predictions = torch.tensor(proto_ids, device=device)[max_indices]

        return predictions
