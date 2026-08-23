import torch
import torch.nn.functional as F
import numpy as np
from configs.config import Config

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
