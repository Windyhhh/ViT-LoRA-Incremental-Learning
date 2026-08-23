import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from transformers import ViTModel
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import os

# 设置使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 使用GPU 1（GPU 0被其他进程占用）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Experiment Setup
    NUM_TASKS = 5
    CLASSES_PER_TASK = 20
    SEED = 42

    # Training Hyperparameters - 增强训练
    BATCH_SIZE = 96  # 稍微减小以提高稳定性
    EPOCHS_PER_TASK = 40  # 基础训练轮数 (Task 2和5会动态调整)
    LR = 2e-4  # 稍微降低学习率以提高稳定性
    WEIGHT_DECAY = 0.015  # 增加正则化
    NUM_WORKERS = 0  # 设为0避免多进程卡住
    WARMUP_EPOCHS = 2  # 学习率预热
    LR_SCHEDULER = "cosine"  # 使用余弦退火

    # Model Architecture
    MODEL_NAME = "google/vit-base-patch16-224"
    HIDDEN_DIM = 768

    # LoRA Configuration - 优化配置
    LORA_RANK = 32
    LORA_ALPHA = 64
    MIN_MERGED_RANK = 32
    USE_RANDOM_ORTHOGONAL = True  # 使用随机正交矩阵 (参考CL-LoRA)

    # Classification
    CLASSIFICATION_TEMPERATURE = 0.05  # 更锐利的决策边界

    # Progressive Merging
    PROGRESSIVE_MERGE = True

    # ============================================================================
    # NEW: Paper-Inspired Merge Strategies
    # ============================================================================
    # Choose merge strategy: "sd_lora", "orthogonal_projection", "knots" (original)
    MERGE_STRATEGY = "orthogonal_projection"  # Default: orthogonal projection (best for CL)

    # SD-LoRA parameters (arXiv:2501.13198)
    SD_LORA_MAGNITUDE_DECAY = 0.8  # Decay factor for new task magnitudes

    # Orthogonal Projection parameters (arXiv:2501.09522)
    OPCM_PROJECTION_THRESHOLD = 0.5  # α in paper (0.4-0.6 recommended)
    OPCM_ADAPTIVE_SCALING = True  # Use sqrt(t) scaling

    # 正交性感知融合 (核心改进1 - 增强)
    ORTHOGONALITY_AWARE_MERGE = True
    ORTHOGONALITY_THRESHOLD_LOW = 0.3
    ORTHOGONALITY_THRESHOLD_HIGH = 0.7
    ADAPTIVE_MERGE_WEIGHTS = True  # 自适应融合权重

    # 多层次特征蒸馏 (核心改进2 - 大幅增强)
    ENABLE_FEATURE_DISTILLATION = True
    FEATURE_DISTILL_LAMBDA = 2.5  # 大幅增强蒸馏强度 (从1.0→2.5)
    DISTILL_TEMPERATURE = 4.0  # 增加温度 (从3.0→4.0)
    MULTI_LAYER_DISTILLATION = True  # 多层蒸馏
    DISTILL_LAYERS = [3, 6, 9, 11]  # 在多个层进行蒸馏
    GRADIENT_REASSIGNMENT = True  # 梯度重分配 (参考CL-LoRA)

    # 原型校准 (核心改进3 - 大幅增强)
    ENABLE_PROTOTYPE_REFINEMENT = True
    PROTOTYPE_REFINEMENT_ITERATIONS = 12  # 增加迭代次数 (从8→12,针对原型对齐瓶颈)
    REFINEMENT_MOMENTUM = 0.85  # 增加动量 (从0.8→0.85,更强的平滑)
    USE_PROCRUSTES_ALIGNMENT = False  # 禁用:在训练中计算SVD导致失败,应仅在任务切换时使用
    PROCRUSTES_LAMBDA = 0.0  # 设为0,不在训练中使用
    CONTRASTIVE_PROTOTYPE_LEARNING = True  # 对比学习增强原型
    CONTRASTIVE_TEMPERATURE = 0.07  # 对比学习温度
    CONTRASTIVE_LAMBDA = 0.8  # 对比学习权重 (从0.5→0.8,增强原型对齐)

    # 多原型策略 (增强)
    NUM_PROTOTYPES_PER_CLASS = 7  # 增加原型数量 (从5→7,提高表达能力)
    PROTOTYPE_DIVERSITY_LAMBDA = 0.3  # 原型多样性正则化

    # Phase 3: 对比学习增强 (禁用:已有CONTRASTIVE_PROTOTYPE_LEARNING)
    ENABLE_CONTRASTIVE_ENHANCEMENT = False  # 禁用:重复的对比学习
    CONTRASTIVE_MARGIN = 0.5  # 对比学习边界
    CONTRASTIVE_HARD_MINING = True  # 困难样本挖掘
    CONTRASTIVE_ENHANCEMENT_LAMBDA = 0.0  # 设为0

    # Phase 4: 原型自校准 (禁用:不稳定)
    ENABLE_PROTOTYPE_SELF_CALIBRATION = False  # 禁用:不稳定
    PROTOTYPE_CALIBRATION_ITERATIONS = 5  # 校准迭代次数
    PROTOTYPE_CALIBRATION_MOMENTUM = 0.9  # 校准动量
    USE_PSEUDO_LABELS = False  # 禁用伪标签
    PSEUDO_LABEL_THRESHOLD = 0.95  # 伪标签置信度阈值

    # Diagnostics
    DIAGNOSTIC_SAMPLES_PER_CLASS = 200
    SAVE_DIAGNOSTICS = True

    # Output
    VERBOSE = True
    CHECKPOINT_DIR = Path("checkpoints_comprehensive")
    RESULTS_DIR = Path("results_comprehensive")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# LoRA Implementation
# ============================================================================

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
    
    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAViT(nn.Module):
    def __init__(self, model_name, rank=16, alpha=32):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.hidden_dim = self.vit.config.hidden_size
        
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.original_forwards = {}
        for i, layer in enumerate(self.vit.encoder.layer):
            attention = layer.attention.attention
            self.original_forwards[f'layer_{i}_q'] = attention.query.forward
            self.original_forwards[f'layer_{i}_v'] = attention.value.forward
        
        self.lora_layers = nn.ModuleDict()
        for i, layer in enumerate(self.vit.encoder.layer):
            self.lora_layers[f'layer_{i}_q'] = LoRALayer(
                self.hidden_dim, self.hidden_dim, rank, alpha
            )
            self.lora_layers[f'layer_{i}_v'] = LoRALayer(
                self.hidden_dim, self.hidden_dim, rank, alpha
            )
        
        self._inject_lora()
    
    def _inject_lora(self):
        for i, layer in enumerate(self.vit.encoder.layer):
            attention = layer.attention.attention
            lora_q = self.lora_layers[f'layer_{i}_q']
            lora_v = self.lora_layers[f'layer_{i}_v']
            original_q_forward = self.original_forwards[f'layer_{i}_q']
            original_v_forward = self.original_forwards[f'layer_{i}_v']
            
            def make_lora_forward(orig_forward, lora_layer):
                def forward(x):
                    return orig_forward(x) + lora_layer(x)
                return forward
            
            attention.query.forward = make_lora_forward(original_q_forward, lora_q)
            attention.value.forward = make_lora_forward(original_v_forward, lora_v)
    
    def forward(self, pixel_values, return_intermediate=False):
        """
        Forward pass with optional intermediate layer features.

        Args:
            pixel_values: Input images
            return_intermediate: If True, return features from intermediate layers

        Returns:
            If return_intermediate=False: final features [batch, 768]
            If return_intermediate=True: dict with features from multiple layers
        """
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=return_intermediate)
        final_features = outputs.last_hidden_state[:, 0, :]

        if not return_intermediate:
            return final_features

        # 返回多层特征用于多层次蒸馏
        intermediate_features = {}
        hidden_states = outputs.hidden_states  # 包括embedding层和所有encoder层

        # 提取指定层的特征 (DISTILL_LAYERS中的层)
        for layer_idx in Config.DISTILL_LAYERS:
            if layer_idx < len(hidden_states):
                # 取[CLS]标记的特征
                intermediate_features[f'layer_{layer_idx}'] = hidden_states[layer_idx][:, 0, :]

        return {
            'final': final_features,
            'intermediate': intermediate_features
        }
    
    def get_lora_state_dict(self):
        return {k: v.cpu().clone() for k, v in self.lora_layers.state_dict().items()}
    
    def set_lora_state_dict(self, state_dict):
        model_device = next(self.vit.parameters()).device
        
        needs_reconstruction = False
        for key, param in state_dict.items():
            if key in self.lora_layers.state_dict():
                current_shape = self.lora_layers.state_dict()[key].shape
                if param.shape != current_shape:
                    needs_reconstruction = True
                    break
        
        if needs_reconstruction:
            layer_ranks = {}
            for key in state_dict.keys():
                if key.endswith('.lora_A'):
                    layer_name = key.rsplit('.', 1)[0]
                    rank = state_dict[key].shape[0]
                    layer_ranks[layer_name] = rank
            
            new_lora_layers = nn.ModuleDict()
            for layer_name, rank in layer_ranks.items():
                new_lora_layers[layer_name] = LoRALayer(
                    self.hidden_dim, self.hidden_dim, rank, 
                    self.lora_layers[layer_name].alpha
                ).to(model_device)
            
            self.lora_layers = new_lora_layers
            self._inject_lora()
        
        state_dict_on_device = {k: v.to(model_device) for k, v in state_dict.items()}
        self.lora_layers.load_state_dict(state_dict_on_device)


# ============================================================================
# Diagnostic Analyzer
# ============================================================================

class FeatureSpaceDiagnostics:
    """
    Comprehensive diagnostics to understand what's happening in feature space.
    
    This class tracks:
    1. How features drift from native space to merged space
    2. Whether prototypes stay aligned with their class samples
    3. Which classes get confused with which
    4. How intra-class variance changes
    """
    
    def __init__(self):
        self.diagnostics_per_task = []
        
    def analyze_feature_space_drift(self, task_id, native_model, merged_model, 
                                    data_loader, class_ids, prototypes, device):
        """
        CRITICAL DIAGNOSTIC: Measure how features change from native→merged space.
        
        This tells us if the merged model produces similar features to the native
        model, or if merging fundamentally transforms the feature space.
        """
        print(f"\n  [DIAGNOSTIC] Analyzing feature space drift for Task {task_id + 1}...")
        
        native_model.eval()
        merged_model.eval()
        
        # Storage for analysis
        native_features_by_class = {cid: [] for cid in class_ids}
        merged_features_by_class = {cid: [] for cid in class_ids}
        
        samples_collected = {cid: 0 for cid in class_ids}
        max_samples = Config.DIAGNOSTIC_SAMPLES_PER_CLASS
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                # Get features from both models
                native_feats = native_model(images)
                merged_feats = merged_model(images)
                
                native_feats = F.normalize(native_feats, p=2, dim=1)
                merged_feats = F.normalize(merged_feats, p=2, dim=1)
                
                for class_id in class_ids:
                    if samples_collected[class_id] >= max_samples:
                        continue
                    
                    mask = labels == class_id
                    if mask.any():
                        n_to_take = min(mask.sum().item(), 
                                       max_samples - samples_collected[class_id])
                        
                        class_native = native_feats[mask][:n_to_take]
                        class_merged = merged_feats[mask][:n_to_take]
                        
                        native_features_by_class[class_id].append(class_native.cpu())
                        merged_features_by_class[class_id].append(class_merged.cpu())
                        
                        samples_collected[class_id] += n_to_take
                
                if all(count >= max_samples for count in samples_collected.values()):
                    break
        
        # Concatenate features
        for cid in class_ids:
            if native_features_by_class[cid]:
                native_features_by_class[cid] = torch.cat(native_features_by_class[cid], dim=0)
                merged_features_by_class[cid] = torch.cat(merged_features_by_class[cid], dim=0)
        
        # METRIC 1: Feature Drift (how much do individual samples move?)
        feature_drifts = []
        for cid in class_ids:
            if len(native_features_by_class[cid]) > 0:
                native = native_features_by_class[cid]
                merged = merged_features_by_class[cid]
                
                # Cosine similarity between native and merged features for same samples
                sample_similarities = (native * merged).sum(dim=1)
                drift = 1 - sample_similarities.mean().item()  # 0 = no drift, 2 = complete reversal
                feature_drifts.append(drift)
        
        avg_feature_drift = np.mean(feature_drifts)
        
        # METRIC 2: Prototype Misalignment (do prototypes stay near their class?)
        prototype_alignments = []
        prototype_to_sample_distances = []
        
        for cid in class_ids:
            if len(merged_features_by_class[cid]) == 0:
                continue
            
            proto = prototypes[cid].to(device)
            samples_merged = merged_features_by_class[cid].to(device)
            
            # Average similarity between prototype and its class samples in merged space
            similarities = samples_merged @ proto
            avg_similarity = similarities.mean().item()
            prototype_alignments.append(avg_similarity)
            
            # Also track distance distribution
            distances = 1 - similarities
            prototype_to_sample_distances.extend(distances.cpu().numpy())
        
        avg_prototype_alignment = np.mean(prototype_alignments)
        
        # METRIC 3: Intra-class Variance (do classes spread out more in merged space?)
        native_intra_variances = []
        merged_intra_variances = []
        
        for cid in class_ids:
            if len(native_features_by_class[cid]) < 2:
                continue
            
            # Variance in native space
            native_feats = native_features_by_class[cid]
            native_centroid = native_feats.mean(dim=0, keepdim=True)
            native_dists = 1 - (native_feats @ native_centroid.T).squeeze()
            native_var = native_dists.var().item()
            native_intra_variances.append(native_var)
            
            # Variance in merged space
            merged_feats = merged_features_by_class[cid]
            merged_centroid = merged_feats.mean(dim=0, keepdim=True)
            merged_dists = 1 - (merged_feats @ merged_centroid.T).squeeze()
            merged_var = merged_dists.var().item()
            merged_intra_variances.append(merged_var)
        
        avg_native_variance = np.mean(native_intra_variances) if native_intra_variances else 0
        avg_merged_variance = np.mean(merged_intra_variances) if merged_intra_variances else 0
        variance_increase = avg_merged_variance - avg_native_variance
        
        # METRIC 4: Inter-class Confusion (which classes get mixed up?)
        confusion_matrix = self._compute_confusion_matrix(
            merged_model, data_loader, class_ids, prototypes, device
        )
        
        # Store diagnostics
        diagnostics = {
            'task_id': task_id,
            'class_ids': class_ids,
            'avg_feature_drift': avg_feature_drift,
            'feature_drift_per_class': feature_drifts,
            'avg_prototype_alignment': avg_prototype_alignment,
            'prototype_alignments': prototype_alignments,
            'avg_native_intra_variance': avg_native_variance,
            'avg_merged_intra_variance': avg_merged_variance,
            'variance_increase': variance_increase,
            'confusion_matrix': confusion_matrix,
            'prototype_to_sample_distances': prototype_to_sample_distances
        }
        
        # Print summary
        print(f"\n  {'='*70}")
        print(f"  FEATURE SPACE DIAGNOSTICS - TASK {task_id + 1}")
        print(f"  {'='*70}")
        print(f"  1. FEATURE DRIFT (native→merged):")
        print(f"     Average drift: {avg_feature_drift:.4f}")
        print(f"     → Lower is better (0 = perfect preservation, 2 = complete reversal)")
        
        print(f"\n  2. PROTOTYPE ALIGNMENT:")
        print(f"     Prototype-to-sample similarity: {avg_prototype_alignment:.4f}")
        print(f"     → Higher is better (1 = perfect, <0.8 indicates misalignment)")
        
        print(f"\n  3. INTRA-CLASS VARIANCE:")
        print(f"     Native space: {avg_native_variance:.6f}")
        print(f"     Merged space: {avg_merged_variance:.6f}")
        print(f"     Increase: {variance_increase:+.6f}")
        print(f"     → Positive = classes spread out more in merged space")
        
        print(f"\n  4. CONFUSION ANALYSIS:")
        print(f"     Top confused class pairs:")
        self._print_top_confusions(confusion_matrix, class_ids, top_k=5)
        
        print(f"  {'='*70}\n")
        
        return diagnostics
    
    def _compute_confusion_matrix(self, model, data_loader, class_ids, prototypes, device):
        """Compute confusion matrix to see which classes get mixed up."""
        model.eval()
        
        # Map class_id to index
        class_to_idx = {cid: i for i, cid in enumerate(sorted(class_ids))}
        num_classes = len(class_ids)
        confusion = torch.zeros(num_classes, num_classes)
        
        proto_ids = sorted(prototypes.keys())
        proto_matrix = torch.stack([prototypes[cid] for cid in proto_ids]).to(device)
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                features = model(images)
                features = F.normalize(features, p=2, dim=1)
                
                similarities = features @ proto_matrix.T
                predictions_idx = similarities.argmax(dim=1)
                predictions = torch.tensor([proto_ids[i] for i in predictions_idx]).to(device)
                
                # Only track this task's classes
                for true_class in class_ids:
                    mask = labels == true_class
                    if mask.any():
                        preds_for_class = predictions[mask]
                        for pred_class in class_ids:
                            count = (preds_for_class == pred_class).sum().item()
                            confusion[class_to_idx[true_class], class_to_idx[pred_class]] += count
        
        # Normalize by row (true class)
        row_sums = confusion.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        confusion = confusion / row_sums
        
        return confusion
    
    def _print_top_confusions(self, confusion_matrix, class_ids, top_k=5):
        """Print the most confused class pairs."""
        class_list = sorted(class_ids)
        num_classes = len(class_list)
        
        confusions = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:  # Exclude diagonal (correct predictions)
                    confusions.append({
                        'true_class': class_list[i],
                        'pred_class': class_list[j],
                        'rate': confusion_matrix[i, j].item()
                    })
        
        confusions.sort(key=lambda x: x['rate'], reverse=True)
        
        for conf in confusions[:top_k]:
            print(f"     Class {conf['true_class']:3d} → {conf['pred_class']:3d}: "
                  f"{conf['rate']*100:5.1f}% misclassified")
    
    def add_diagnostics(self, diagnostics):
        """Store diagnostics for later analysis."""
        self.diagnostics_per_task.append(diagnostics)
    
    def print_cumulative_analysis(self):
        """Print analysis across all tasks to identify trends."""
        if not self.diagnostics_per_task:
            return
        
        print(f"\n{'='*80}")
        print(f"CUMULATIVE DIAGNOSTIC ANALYSIS")
        print(f"{'='*80}\n")
        
        print("Evolution of Metrics Across Tasks:")
        print(f"{'Task':<10} {'Drift':<12} {'Proto Align':<15} {'Var Increase':<15}")
        print("-" * 55)
        
        for diag in self.diagnostics_per_task:
            print(f"{diag['task_id']+1:<10} "
                  f"{diag['avg_feature_drift']:<12.4f} "
                  f"{diag['avg_prototype_alignment']:<15.4f} "
                  f"{diag['variance_increase']:+<15.6f}")
        
        # Identify the main bottleneck
        avg_drift = np.mean([d['avg_feature_drift'] for d in self.diagnostics_per_task])
        avg_alignment = np.mean([d['avg_prototype_alignment'] for d in self.diagnostics_per_task])
        avg_var_increase = np.mean([d['variance_increase'] for d in self.diagnostics_per_task])
        
        print(f"\nAverages:")
        print(f"  Feature Drift: {avg_drift:.4f}")
        print(f"  Prototype Alignment: {avg_alignment:.4f}")
        print(f"  Variance Increase: {avg_var_increase:+.6f}")
        
        print(f"\n{'='*80}")
        print("BOTTLENECK IDENTIFICATION")
        print(f"{'='*80}\n")
        
        # Diagnose the main issue
        if avg_drift > 0.3:
            print("🔴 PRIMARY BOTTLENECK: SEVERE FEATURE DRIFT")
            print("   Features change dramatically from native→merged space.")
            print("   Prototypes computed in native space don't match merged space.")
            print("   → Solution: Need feature space alignment or prototype recomputation.")
        elif avg_alignment < 0.7:
            print("🔴 PRIMARY BOTTLENECK: PROTOTYPE MISALIGNMENT")
            print("   Prototypes are far from their class samples in merged space.")
            print("   Even if features don't drift much, prototypes are in wrong positions.")
            print("   → Solution: Need to adjust prototype positions in merged space.")
        elif avg_var_increase > 0.01:
            print("🟡 SECONDARY ISSUE: INCREASED INTRA-CLASS VARIANCE")
            print("   Classes spread out more in merged space, making boundaries fuzzy.")
            print("   → May benefit from variance-aware distance metrics.")
        else:
            print("✅ NO CLEAR BOTTLENECK DETECTED")
            print("   Feature space metrics look reasonable.")
            print("   The issue may be in the merging strategy itself or other factors.")
        
        print(f"\n{'='*80}\n")


# ============================================================================
# Classifier with Diagnostics
# ============================================================================

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


# ============================================================================
# Advanced Alignment and Learning Functions
# ============================================================================

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
    importance_weights = F.softmax(importance_weights, dim=0)

    # 重分配梯度
    reassigned_gradients = gradients * importance_weights.unsqueeze(1)

    return reassigned_gradients


# ============================================================================
# Multi-Layer Distillation (第一阶段改进)
# ============================================================================

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


# ============================================================================
# Procrustes Alignment Enhancement (Phase 2改进)
# ============================================================================

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


# ============================================================================
# LoRA Merging
# ============================================================================

def merge_loras_sd_lora_inspired(merged_lora_state, new_lora_state, task_id,
                                  magnitude_history=None, direction_history=None):
    """
    SD-LoRA inspired merge: Decouple magnitude and direction

    Key idea from arXiv:2501.13198:
    - Fix directions learned from previous tasks
    - Only learn magnitudes (alphas) for each direction
    - Output: W = W_0 + Σ α_k · (A_k B_k / ||A_k B_k||)

    Args:
        merged_lora_state: Previously merged LoRA state (or None for first task)
        new_lora_state: New task's LoRA state
        task_id: Current task ID (1-indexed)
        magnitude_history: List of learned magnitudes [α_1, α_2, ...]
        direction_history: List of normalized directions [D_1, D_2, ...]

    Returns:
        merged_state: New merged LoRA state
        magnitude_history: Updated magnitude history
        direction_history: Updated direction history
    """
    if magnitude_history is None:
        magnitude_history = []
    if direction_history is None:
        direction_history = []

    # First task: just normalize and store
    if task_id == 1:
        merged_state = {}
        layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

        for layer_name in layers:
            A = new_lora_state[f'{layer_name}.lora_A']
            B = new_lora_state[f'{layer_name}.lora_B']

            # Compute ΔW = B @ A
            delta_W = B @ A

            # Compute magnitude and direction
            magnitude = torch.norm(delta_W, p='fro')
            direction = delta_W / (magnitude + 1e-8)

            # Store direction
            if len(direction_history) == 0:
                direction_history.append({})
            direction_history[0][layer_name] = direction

            # Store magnitude (α_1 = 1.0 for first task)
            if len(magnitude_history) == 0:
                magnitude_history.append(1.0)

            # Reconstruct LoRA: ΔW = α * direction
            scaled_delta_W = magnitude_history[0] * direction_history[0][layer_name]

            # Decompose back to A, B using SVD
            U, S, Vh = torch.linalg.svd(scaled_delta_W, full_matrices=False)
            rank = min(Config.MIN_MERGED_RANK, len(S))

            sqrt_S = torch.sqrt(S[:rank])
            merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S) @ Vh[:rank, :]
            merged_state[f'{layer_name}.lora_B'] = U[:, :rank] @ torch.diag(sqrt_S)

        return merged_state, magnitude_history, direction_history

    # Subsequent tasks: add new direction with learned magnitude
    merged_state = {}
    layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

    # Add new direction from current task
    direction_history.append({})

    for layer_name in layers:
        A_new = new_lora_state[f'{layer_name}.lora_A']
        B_new = new_lora_state[f'{layer_name}.lora_B']

        # Compute new task's ΔW
        delta_W_new = B_new @ A_new

        # Normalize to get direction
        magnitude_new = torch.norm(delta_W_new, p='fro')
        direction_new = delta_W_new / (magnitude_new + 1e-8)

        # Store new direction
        direction_history[-1][layer_name] = direction_new

        # Compute magnitude for new task (adaptive based on previous magnitudes)
        # α_t = average of previous magnitudes (conservative approach)
        avg_magnitude = sum(magnitude_history) / len(magnitude_history)
        new_magnitude = 0.8 * avg_magnitude  # Slightly reduce to prevent drift

        # Reconstruct merged ΔW = Σ α_k · D_k
        delta_W_merged = torch.zeros_like(direction_new)
        for k, (alpha, directions) in enumerate(zip(magnitude_history, direction_history[:-1])):
            delta_W_merged += alpha * directions[layer_name]

        # Add new task's contribution
        delta_W_merged += new_magnitude * direction_new

        # Decompose back to A, B
        U, S, Vh = torch.linalg.svd(delta_W_merged, full_matrices=False)
        rank = min(Config.MIN_MERGED_RANK, len(S))

        sqrt_S = torch.sqrt(S[:rank])
        merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S) @ Vh[:rank, :]
        merged_state[f'{layer_name}.lora_B'] = U[:, :rank] @ torch.diag(sqrt_S)

    # Update magnitude history
    magnitude_history.append(new_magnitude)

    return merged_state, magnitude_history, direction_history


def merge_loras_orthogonal_projection(merged_lora_state, new_lora_state, task_id,
                                       projection_threshold=0.5):
    """
    Orthogonal Projection-based Continual Merging (OPCM)

    Key idea from arXiv:2501.09522:
    - Project new task vector onto subspace orthogonal to merged task vector
    - Use adaptive scaling to maintain stable parameter distance
    - Minimize interference between tasks

    Args:
        merged_lora_state: Previously merged LoRA state (or None for first task)
        new_lora_state: New task's LoRA state
        task_id: Current task ID (1-indexed)
        projection_threshold: α in paper (controls orthogonal projection)

    Returns:
        merged_state: New merged LoRA state
    """
    # First task: just return as-is
    if task_id == 1 or merged_lora_state is None:
        return new_lora_state

    merged_state = {}
    layers = set([k.rsplit('.', 1)[0] for k in new_lora_state.keys()])

    for layer_name in layers:
        # Get merged and new LoRA parameters
        A_merged = merged_lora_state[f'{layer_name}.lora_A']
        B_merged = merged_lora_state[f'{layer_name}.lora_B']
        A_new = new_lora_state[f'{layer_name}.lora_A']
        B_new = new_lora_state[f'{layer_name}.lora_B']

        # Compute task vectors: ΔW = B @ A
        delta_W_merged = B_merged @ A_merged
        delta_W_new = B_new @ A_new

        # SVD of merged task vector
        U, S, Vh = torch.linalg.svd(delta_W_merged, full_matrices=False)

        # Determine rank threshold based on projection_threshold
        total_energy = S.sum()
        cumsum_energy = torch.cumsum(S, dim=0)
        r_alpha = torch.searchsorted(cumsum_energy, projection_threshold * total_energy).item() + 1
        r_alpha = min(r_alpha, len(S))

        # Orthogonal projection: P_α(ΔW_new)
        # Project onto subspace spanned by {u_i v_j^T} where i≠j or i,j >= r_α
        delta_W_proj = torch.zeros_like(delta_W_new)

        for i in range(delta_W_new.shape[0]):
            for j in range(delta_W_new.shape[1]):
                # Skip if within the principal subspace
                if i < r_alpha and j < r_alpha and i == j:
                    continue

                # Compute projection coefficient
                if i < len(U) and j < len(Vh):
                    u_i = U[:, i:i+1]
                    v_j = Vh[j:j+1, :]
                    basis = u_i @ v_j

                    # <ΔW_new, u_i v_j^T>_F
                    coeff = torch.sum(delta_W_new * basis)
                    delta_W_proj += coeff * basis

        # Adaptive scaling: λ^(t) = ||λ^(t-1) ΔW_merged + P_α(ΔW_new)|| / avg_norm
        # For simplicity, use sqrt(t) scaling as suggested in paper
        lambda_t = torch.sqrt(torch.tensor(task_id, dtype=torch.float32))

        # Merge: ΔW_merged^(t) = (ΔW_merged + P_α(ΔW_new)) / λ^(t)
        delta_W_merged_new = (delta_W_merged + delta_W_proj) / lambda_t

        # Decompose back to A, B
        U_new, S_new, Vh_new = torch.linalg.svd(delta_W_merged_new, full_matrices=False)
        rank = min(Config.MIN_MERGED_RANK, len(S_new))

        sqrt_S_new = torch.sqrt(S_new[:rank])
        merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S_new) @ Vh_new[:rank, :]
        merged_state[f'{layer_name}.lora_B'] = U_new[:, :rank] @ torch.diag(sqrt_S_new)

    return merged_state


def merge_loras_knots_inspired(lora_states_list, weights):
    """Original KNOTS-inspired merge (kept for backward compatibility)"""
    if len(lora_states_list) == 1:
        return lora_states_list[0]

    weights = np.array(weights)
    weights = weights / weights.sum()

    merged_state = {}
    layers = set([k.rsplit('.', 1)[0] for k in lora_states_list[0].keys()])

    for layer_name in layers:
        lora_As = [state[f'{layer_name}.lora_A'] for state in lora_states_list]
        lora_Bs = [state[f'{layer_name}.lora_B'] for state in lora_states_list]

        original_ranks = [A.shape[0] for A in lora_As]

        svds = []
        for A, B, orig_rank in zip(lora_As, lora_Bs, original_ranks):
            W = B @ A
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            U_truncated = U[:, :orig_rank]
            S_truncated = S[:orig_rank]
            Vh_truncated = Vh[:orig_rank, :]

            svds.append((U_truncated, S_truncated, Vh_truncated, orig_rank))

        ref_U, ref_S, ref_Vh, ref_rank = svds[0]

        matched_As = [lora_As[0]]
        matched_Bs = [lora_Bs[0]]

        for i in range(1, len(lora_states_list)):
            U, S, Vh, curr_rank = svds[i]

            similarity = torch.zeros(curr_rank, ref_rank)
            for j in range(curr_rank):
                for k in range(ref_rank):
                    left_sim = torch.abs(U[:, j] @ ref_U[:, k])
                    right_sim = torch.abs(Vh[j, :] @ ref_Vh[k, :])
                    similarity[j, k] = (left_sim + right_sim) / 2

            row_ind, col_ind = linear_sum_assignment(-similarity.cpu().numpy())

            perm_matrix = torch.zeros(curr_rank, ref_rank, device=lora_As[i].device)
            for j, k in zip(row_ind, col_ind):
                if j < curr_rank and k < ref_rank:
                    perm_matrix[j, k] = 1

            A_permuted = perm_matrix @ lora_As[i]
            B_permuted = lora_Bs[i] @ perm_matrix.T

            matched_As.append(A_permuted)
            matched_Bs.append(B_permuted)

        W_list = [B @ A for A, B in zip(matched_As, matched_Bs)]
        W_merged = sum(w * W for w, W in zip(weights, W_list))

        U, S, Vh = torch.linalg.svd(W_merged, full_matrices=False)
        rank = min(Config.MIN_MERGED_RANK, len(S))

        sqrt_S = torch.sqrt(S[:rank])
        merged_state[f'{layer_name}.lora_A'] = torch.diag(sqrt_S) @ Vh[:rank, :]
        merged_state[f'{layer_name}.lora_B'] = U[:, :rank] @ torch.diag(sqrt_S)

    return merged_state


# ============================================================================
# Dataset
# ============================================================================

class CIFAR100ClassIncremental(Dataset):
    def __init__(self, cifar_dataset, class_list):
        self.cifar_dataset = cifar_dataset
        self.class_set = set(class_list)
        self.indices = [i for i, (_, label) in enumerate(cifar_dataset) 
                       if label in self.class_set]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.cifar_dataset[real_idx]


def build_tasks(num_tasks=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform_train = T.Compose([
        T.Resize(224),
        T.RandomCrop(224, padding=28),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    print("加载CIFAR-100数据集...")
    train_data = datasets.CIFAR100("./data", train=True, download=False,
                                   transform=transform_train)
    test_data = datasets.CIFAR100("./data", train=False, download=False,
                                  transform=transform_test)
    print(f"数据集加载完成！训练集: {len(train_data)}, 测试集: {len(test_data)}")
    
    all_classes = list(range(100))
    random.shuffle(all_classes)
    classes_per_task = 100 // num_tasks

    print(f"构建{num_tasks}个任务，每个任务{classes_per_task}个类别...")
    tasks = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        task_classes = sorted(all_classes[start:start+classes_per_task])

        print(f"  任务{task_id+1}: 类别 {task_classes[:3]}...{task_classes[-3:]}")

        train_dataset = CIFAR100ClassIncremental(train_data, task_classes)
        test_dataset = CIFAR100ClassIncremental(test_data, task_classes)

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
            num_workers=Config.NUM_WORKERS, pin_memory=False  # 禁用pin_memory避免卡住
        )
        test_loader = DataLoader(
            test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
            num_workers=Config.NUM_WORKERS, pin_memory=False
        )

        tasks.append({
            'train_loader': train_loader,
            'test_loader': test_loader,
            'classes': task_classes
        })

    print(f"任务构建完成！\n")
    
    return tasks


# ============================================================================
# Training
# ============================================================================

def train_task(model, task_data, task_id, device, old_model=None, old_prototypes=None):
    """
    增强版训练函数,包含:
    1. 学习率预热和余弦退火
    2. 多原型学习
    3. 对比学习
    4. 特征蒸馏 (如果有旧模型)
    5. 动态训练轮数 (Task 4和5增加训练时间)
    """
    # 动态调整训练轮数和配置 (针对困难任务)
    epochs_for_task = Config.EPOCHS_PER_TASK
    distill_lambda = Config.FEATURE_DISTILL_LAMBDA
    contrastive_lambda = Config.CONTRASTIVE_LAMBDA
    refinement_iterations = Config.PROTOTYPE_REFINEMENT_ITERATIONS

    if task_id == 2:
        # Task 2 是最大瓶颈 (-5.97%),需要超强优化
        epochs_for_task = int(Config.EPOCHS_PER_TASK * 2.0)  # +100%
        distill_lambda = Config.FEATURE_DISTILL_LAMBDA * 1.5  # 蒸馏强度 +50%
        contrastive_lambda = Config.CONTRASTIVE_LAMBDA * 1.5  # 对比学习 +50%
        refinement_iterations = int(Config.PROTOTYPE_REFINEMENT_ITERATIONS * 1.5)  # 精炼迭代 +50%
        print(f"  [INFO] Task 2 使用超强训练 (最大瓶颈 -5.97%):")
        print(f"    - Epochs: {epochs_for_task} (+100%)")
        print(f"    - 蒸馏强度: {distill_lambda:.2f} (+50%)")
        print(f"    - 对比学习: {contrastive_lambda:.2f} (+50%)")
        print(f"    - 精炼迭代: {refinement_iterations} (+50%)")
    elif task_id == 3:
        epochs_for_task = int(Config.EPOCHS_PER_TASK * 1.3)  # Task 3: +30%
        print(f"  [INFO] Task 3 使用增强训练: {epochs_for_task} epochs (+30%)")
    elif task_id == 5:
        epochs_for_task = int(Config.EPOCHS_PER_TASK * 1.5)  # Task 5: +50%
        print(f"  [INFO] Task 5 使用增强训练: {epochs_for_task} epochs (+50%)")

    optimizer = torch.optim.AdamW(
        model.lora_layers.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 学习率调度器
    if Config.LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_for_task, eta_min=Config.LR * 0.01
        )
    else:
        scheduler = None

    # 多原型存储: {class_id: [proto1, proto2, ...]}
    task_multi_prototypes = defaultdict(list)
    task_prototypes = {}  # 主原型 (用于分类)

    model.train()
    if old_model is not None:
        old_model.eval()

    for epoch in range(epochs_for_task):
        total_loss = 0
        total_ce_loss = 0
        total_distill_loss = 0
        total_contrastive_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        # 学习率预热
        if epoch < Config.WARMUP_EPOCHS and scheduler is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = Config.LR * (epoch + 1) / Config.WARMUP_EPOCHS

        pbar = tqdm(task_data['train_loader'],
                   desc=f"  Epoch {epoch+1}/{epochs_for_task}")

        for images, labels in pbar:
            batch_count += 1
            images, labels = images.to(device), labels.to(device)

            # 获取特征 (支持多层次蒸馏)
            if Config.MULTI_LAYER_DISTILLATION and old_model is not None:
                model_output = model(images, return_intermediate=True)
                features = model_output['final']
                current_intermediate = model_output['intermediate']
            else:
                features = model(images)
                current_intermediate = None

            features_norm = F.normalize(features, p=2, dim=1)

            # 更新多原型
            for class_id in torch.unique(labels):
                mask = labels == class_id
                class_features = features_norm[mask].detach()

                # 主原型 (动量更新)
                if class_id.item() not in task_prototypes:
                    task_prototypes[class_id.item()] = class_features.mean(dim=0)
                else:
                    task_prototypes[class_id.item()] = 0.9 * task_prototypes[class_id.item()] + \
                                                      0.1 * class_features.mean(dim=0)

                task_prototypes[class_id.item()] = F.normalize(
                    task_prototypes[class_id.item()].unsqueeze(0), p=2, dim=1
                ).squeeze(0)

                # 多原型 (k-means聚类) - 简化版本
                if len(task_multi_prototypes[class_id.item()]) < Config.NUM_PROTOTYPES_PER_CLASS:
                    # 初始化阶段:随机选择
                    if len(class_features) >= Config.NUM_PROTOTYPES_PER_CLASS:
                        indices = torch.randperm(len(class_features))[:Config.NUM_PROTOTYPES_PER_CLASS]
                        task_multi_prototypes[class_id.item()] = [
                            class_features[i].clone().detach() for i in indices
                        ]
                    else:
                        for feat in class_features:
                            if len(task_multi_prototypes[class_id.item()]) < Config.NUM_PROTOTYPES_PER_CLASS:
                                task_multi_prototypes[class_id.item()].append(feat.clone().detach())
                else:
                    # 更新阶段:动量更新
                    for i in range(min(len(task_multi_prototypes[class_id.item()]), Config.NUM_PROTOTYPES_PER_CLASS)):
                        # 找到最接近第i个原型的特征
                        proto = task_multi_prototypes[class_id.item()][i]
                        if isinstance(proto, torch.Tensor):
                            similarities = class_features @ proto
                            closest_idx = similarities.argmax()
                            closest_feature = class_features[closest_idx]

                            # 动量更新
                            task_multi_prototypes[class_id.item()][i] = \
                                0.9 * proto + 0.1 * closest_feature
                            task_multi_prototypes[class_id.item()][i] = F.normalize(
                                task_multi_prototypes[class_id.item()][i].unsqueeze(0), p=2, dim=1
                            ).squeeze(0)

            # 分类损失
            if len(task_prototypes) > 0:
                proto_ids = sorted(task_prototypes.keys())
                proto_features = torch.stack([task_prototypes[k] for k in proto_ids]).to(device)

                logits = features_norm @ proto_features.T / Config.CLASSIFICATION_TEMPERATURE
                targets = torch.tensor([proto_ids.index(l.item()) for l in labels]).to(device)
                ce_loss = F.cross_entropy(logits, targets)
            else:
                ce_loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss = ce_loss

            # 对比学习损失
            if Config.CONTRASTIVE_PROTOTYPE_LEARNING and len(task_prototypes) > 0:
                contrastive_loss = contrastive_prototype_loss(
                    features_norm, labels, task_prototypes,
                    temperature=Config.CONTRASTIVE_TEMPERATURE
                )
                loss = loss + contrastive_lambda * contrastive_loss  # 使用动态对比学习权重
                total_contrastive_loss += contrastive_loss.item()

            # 特征蒸馏损失
            if old_model is not None and Config.ENABLE_FEATURE_DISTILLATION:
                with torch.no_grad():
                    if Config.MULTI_LAYER_DISTILLATION:
                        # 多层次蒸馏: 获取旧模型的多层特征
                        old_output = old_model(images, return_intermediate=True)
                        old_features = old_output['final']
                        old_intermediate = old_output['intermediate']
                    else:
                        old_features = old_model(images)
                        old_intermediate = None

                    old_features_norm = F.normalize(old_features, p=2, dim=1)

                # 最后层蒸馏 (KL散度)
                distill_loss = F.kl_div(
                    F.log_softmax(features_norm / Config.DISTILL_TEMPERATURE, dim=1),
                    F.softmax(old_features_norm / Config.DISTILL_TEMPERATURE, dim=1),
                    reduction='batchmean'
                ) * (Config.DISTILL_TEMPERATURE ** 2)

                # 多层次蒸馏 (如果启用)
                if Config.MULTI_LAYER_DISTILLATION and current_intermediate is not None and old_intermediate is not None:
                    # 对中间层特征进行蒸馏
                    multi_layer_loss = compute_multi_layer_distillation(
                        current_intermediate,
                        old_intermediate,
                        temperature=Config.DISTILL_TEMPERATURE
                    )
                    # 多层次蒸馏权重更高 (70% 中间层, 30% 最后层)
                    distill_loss = 0.3 * distill_loss + 0.7 * multi_layer_loss

                loss = loss + distill_lambda * distill_loss  # 使用动态蒸馏强度
                total_distill_loss += distill_loss.item()

                # Procrustes对齐损失 (Phase 2核心!)
                if Config.USE_PROCRUSTES_ALIGNMENT:
                    procrustes_loss = compute_procrustes_alignment_loss(
                        features_norm, old_features_norm,
                        temperature=Config.DISTILL_TEMPERATURE
                    )
                    loss = loss + Config.PROCRUSTES_LAMBDA * procrustes_loss

            # Phase 3: 对比学习增强 (新增)
            if Config.ENABLE_CONTRASTIVE_ENHANCEMENT and len(task_prototypes) > 0:
                contrastive_enh_loss = contrastive_enhancement_loss(
                    features_norm, labels, task_prototypes,
                    margin=Config.CONTRASTIVE_MARGIN,
                    temperature=Config.CONTRASTIVE_TEMPERATURE,
                    hard_mining=Config.CONTRASTIVE_HARD_MINING
                )
                loss = loss + Config.CONTRASTIVE_ENHANCEMENT_LAMBDA * contrastive_enh_loss

            # Phase 4: 原型自校准 (新增)
            if Config.ENABLE_PROTOTYPE_SELF_CALIBRATION and len(task_prototypes) > 0:
                # 每N个batch进行一次校准
                if batch_count % 10 == 0:
                    calibrated_protos = prototype_self_calibration(
                        model, features_norm, labels, task_prototypes,
                        old_prototypes=old_prototypes,
                        iterations=Config.PROTOTYPE_CALIBRATION_ITERATIONS,
                        momentum=Config.PROTOTYPE_CALIBRATION_MOMENTUM,
                        threshold=Config.PSEUDO_LABEL_THRESHOLD,
                        device=device
                    )
                    # 更新原型
                    task_prototypes.update(calibrated_protos)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if len(task_prototypes) > 0:
                    predictions_idx = logits.argmax(dim=1)
                    predictions = torch.tensor([proto_ids[i] for i in predictions_idx]).to(device)
                    correct += (predictions == labels).sum().item()
                total += len(labels)
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()

            pbar.set_postfix({
                'loss': f'{total_loss/(pbar.n+1):.3f}',
                'ce': f'{total_ce_loss/(pbar.n+1):.3f}',
                'acc': f'{100*correct/total:.1f}%'
            })

        # 学习率调度
        if scheduler is not None and epoch >= Config.WARMUP_EPOCHS:
            scheduler.step()

    return model.get_lora_state_dict(), task_prototypes, task_multi_prototypes


def evaluate_all_tasks(model, all_tasks, classifier, task_id, device):
    model.eval()
    results = {'per_task': {}, 'overall_cil': 0.0}
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for tid in range(task_id + 1):
            task_correct = 0
            task_total = 0
            
            for images, labels in all_tasks[tid]['test_loader']:
                images = images.to(device)
                labels = labels.to(device)
                features = model(images)
                
                predictions = classifier.predict(features, device)
                
                correct = (predictions == labels).sum().item()
                task_correct += correct
                task_total += len(labels)
                total_correct += correct
                total_samples += len(labels)
            
            accuracy = 100 * task_correct / task_total
            results['per_task'][tid + 1] = accuracy
    
    results['overall_cil'] = 100 * total_correct / total_samples
    return results


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(tasks):
    print(f"\n{'='*80}")
    print(f"CONTINUAL LEARNING WITH COMPREHENSIVE IMPROVEMENTS")
    print(f"{'='*80}\n")

    classifier = PrototypeClassifier(temperature=Config.CLASSIFICATION_TEMPERATURE)
    diagnostics_tracker = FeatureSpaceDiagnostics()

    # Progressive merging: 只保留merged_lora_state
    merged_lora_state = None
    previous_merged_model = None
    native_models = []  # Store native models for diagnostics
    all_results = []

    # For SD-LoRA: track magnitude and direction history
    magnitude_history = []
    direction_history = []

    for task_id in range(Config.NUM_TASKS):
        print(f"\n{'-'*80}")
        print(f"TASK {task_id + 1}/{Config.NUM_TASKS}")
        print(f"Classes: {tasks[task_id]['classes']}")
        print(f"{'-'*80}")

        # Train LoRA
        print(f"\nTraining Task {task_id + 1}...")
        native_model = LoRAViT(Config.MODEL_NAME, rank=Config.LORA_RANK,
                              alpha=Config.LORA_ALPHA).to(Config.DEVICE)

        # 使用增强训练函数
        trained_lora, task_prototypes, task_multi_prototypes = train_task(
            native_model, tasks[task_id], task_id, Config.DEVICE,
            old_model=previous_merged_model,
            old_prototypes=classifier.prototypes if task_id > 0 else None
        )

        # Extract prototypes (使用训练得到的原型)
        for class_id, proto in task_prototypes.items():
            classifier.prototypes[class_id] = proto.cpu()
        classifier.task_classes[task_id] = tasks[task_id]['classes']

        # Store native model for diagnostics
        native_models.append(native_model)

        # Progressive Merging with Paper-Inspired Strategies
        print(f"\nProgressive Merging (Strategy: {Config.MERGE_STRATEGY})...")
        if task_id == 0:
            # 第一个任务:直接使用
            merged_lora_state = {k: v.cpu() for k, v in trained_lora.items()}

            # Initialize for SD-LoRA
            if Config.MERGE_STRATEGY == "sd_lora":
                merged_lora_state, magnitude_history, direction_history = \
                    merge_loras_sd_lora_inspired(
                        None, merged_lora_state, task_id + 1,
                        magnitude_history, direction_history
                    )
        else:
            # 递进融合: merged_new = merge(merged_old, lora_new)
            if Config.MERGE_STRATEGY == "sd_lora":
                # SD-LoRA: Decouple magnitude and direction
                print(f"  Using SD-LoRA merge (magnitude decay={Config.SD_LORA_MAGNITUDE_DECAY})")
                merged_lora_state, magnitude_history, direction_history = \
                    merge_loras_sd_lora_inspired(
                        merged_lora_state, trained_lora, task_id + 1,
                        magnitude_history, direction_history
                    )
                print(f"  Magnitude history: {[f'{m:.3f}' for m in magnitude_history]}")

            elif Config.MERGE_STRATEGY == "orthogonal_projection":
                # Orthogonal Projection-based Continual Merging (OPCM)
                print(f"  Using OPCM (projection_threshold={Config.OPCM_PROJECTION_THRESHOLD})")
                merged_lora_state = merge_loras_orthogonal_projection(
                    merged_lora_state, trained_lora, task_id + 1,
                    projection_threshold=Config.OPCM_PROJECTION_THRESHOLD
                )

            else:  # "knots" or default
                # Original KNOTS-inspired merge
                print(f"  Using KNOTS-inspired merge")
                if Config.ADAPTIVE_MERGE_WEIGHTS:
                    # 自适应权重:根据任务性能调整
                    # 简单策略:给新任务更多权重以减少遗忘
                    weight_old = 0.6
                    weight_new = 0.4
                else:
                    weight_old = 0.5
                    weight_new = 0.5

                merged_lora_state = merge_loras_knots_inspired(
                    [merged_lora_state, trained_lora],
                    [weight_old, weight_new]
                )

        # 创建merged模型
        merged_model = LoRAViT(Config.MODEL_NAME, rank=Config.LORA_RANK,
                              alpha=Config.LORA_ALPHA).to(Config.DEVICE)
        merged_model.set_lora_state_dict(merged_lora_state)

        # 原型精炼 - 关键改进!
        print(f"\nRefining prototypes in merged space...")
        for refine_task_id in range(task_id + 1):
            # 为Task 2使用增强的精炼迭代次数
            refine_iters = int(Config.PROTOTYPE_REFINEMENT_ITERATIONS * 1.5) if refine_task_id == 1 else None
            classifier.refine_prototypes_in_merged_space(
                native_models[refine_task_id],
                merged_model,
                tasks[refine_task_id]['train_loader'],
                tasks[refine_task_id]['classes'],
                Config.DEVICE,
                refinement_iterations=refine_iters
            )

        # RUN DIAGNOSTICS
        for diag_task_id in range(task_id + 1):
            diag = diagnostics_tracker.analyze_feature_space_drift(
                diag_task_id,
                native_models[diag_task_id],
                merged_model,
                tasks[diag_task_id]['test_loader'],
                tasks[diag_task_id]['classes'],
                classifier.prototypes,
                Config.DEVICE
            )
            diagnostics_tracker.add_diagnostics(diag)

        # Evaluate
        print(f"\nEvaluating on all {task_id + 1} task(s)...")
        results = evaluate_all_tasks(merged_model, tasks, classifier, task_id, Config.DEVICE)
        all_results.append(results)

        print(f"\n  Overall CIL Accuracy: {results['overall_cil']:.2f}%")

        # 更新previous_merged_model用于下一个任务的蒸馏
        if previous_merged_model is not None:
            del previous_merged_model
        previous_merged_model = merged_model

    # Print cumulative analysis
    diagnostics_tracker.print_cumulative_analysis()

    return {
        'results': all_results,
        'diagnostics': diagnostics_tracker.diagnostics_per_task
    }


def main():
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True)
    
    tasks = build_tasks(Config.NUM_TASKS, Config.SEED)
    
    experiment_results = run_experiment(tasks)
    
    # Save
    torch.save(experiment_results, Config.RESULTS_DIR / 'diagnostic_results.pt')
    print(f"\nResults saved to {Config.RESULTS_DIR / 'diagnostic_results.pt'}")


if __name__ == "__main__":
    main()
