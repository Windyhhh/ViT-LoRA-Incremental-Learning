import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from tqdm import tqdm
from pathlib import Path
import numpy as np
import random

# Import configuration
from configs.config import Config

# Import models
from src.models.lora import LoRAViT

# Import classifiers
from src.classifiers.prototype_classifier import PrototypeClassifier

# Import diagnostics
from src.diagnostics.diagnostics import FeatureSpaceDiagnostics

# Import losses
from src.losses.distillation import compute_multi_layer_distillation, compute_procrustes_alignment_loss
from src.losses.prototype_losses import contrastive_prototype_loss, compute_prototype_diversity_loss

# Import merging strategies
from src.merging.merging import merge_loras_sd_lora_inspired, merge_loras_orthogonal_projection

# Import utils
from src.utils.alignment import procrustes_alignment

# Set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(Config.SEED)

# Data preparation
def get_task_datasets(task_id, transform):
    """
    获取特定任务的训练和测试数据集
    """
    # CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    # 计算当前任务的类别范围
    start_class = task_id * Config.CLASSES_PER_TASK
    end_class = start_class + Config.CLASSES_PER_TASK
    
    # 创建任务特定的数据集
    class TaskDataset(Dataset):
        def __init__(self, dataset, start_class, end_class):
            self.dataset = dataset
            self.start_class = start_class
            self.end_class = end_class
            self.indices = [i for i, (_, label) in enumerate(dataset) if start_class <= label < end_class]
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            return img, label
    
    task_train_dataset = TaskDataset(train_dataset, start_class, end_class)
    task_test_dataset = TaskDataset(test_dataset, start_class, end_class)
    
    return task_train_dataset, task_test_dataset

def get_full_test_dataset(transform):
    """
    获取完整的测试数据集用于评估所有任务
    """
    return datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Training pipeline
def train():
    # Initialize device
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Create directories for checkpoints and results
    Config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    Config.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    # Define transforms
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Initialize diagnostics
    diagnostics = FeatureSpaceDiagnostics()
    
    # Initialize classifier
    classifier = PrototypeClassifier(temperature=Config.CLASSIFICATION_TEMPERATURE)
    
    # Initialize merged LoRA state
    merged_lora_state = None
    magnitude_history = None
    direction_history = None
    
    # Initialize results tracking
    results = {
        'task_accuracies': [],
        'merged_accuracies': []
    }
    
    # Main training loop over tasks
    for task_id in range(Config.NUM_TASKS):
        print(f"\n{'='*80}")
        print(f"STARTING TASK {task_id + 1}/{Config.NUM_TASKS}")
        print(f"{'='*80}")
        
        # Get task-specific datasets
        task_train_dataset, task_test_dataset = get_task_datasets(task_id, transform)
        
        # Create data loaders
        train_loader = DataLoader(
            task_train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS
        )
        test_loader = DataLoader(
            task_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
        )
        
        # Initialize native model for current task
        native_model = LoRAViT(
            Config.MODEL_NAME, 
            rank=Config.LORA_RANK, 
            alpha=Config.LORA_ALPHA
        ).to(device)
        
        # If we have a merged LoRA state, initialize merged model
        if merged_lora_state is not None:
            merged_model = LoRAViT(
                Config.MODEL_NAME, 
                rank=Config.LORA_RANK, 
                alpha=Config.LORA_ALPHA
            ).to(device)
            merged_model.set_lora_state_dict(merged_lora_state)
        else:
            merged_model = None
        
        # Extract class IDs for current task
        start_class = task_id * Config.CLASSES_PER_TASK
        end_class = start_class + Config.CLASSES_PER_TASK
        current_class_ids = list(range(start_class, end_class))
        
        # Train native model for current task
        print(f"\nTraining native model for Task {task_id + 1}...")
        train_native_model(native_model, train_loader, device, task_id, merged_model=merged_model)
        
        # Evaluate native model on current task
        print(f"\nEvaluating native model on Task {task_id + 1}...")
        native_accuracy = evaluate_model(native_model, test_loader, device)
        print(f"Native model accuracy on Task {task_id + 1}: {native_accuracy:.4f}")
        
        # Extract prototypes from native model
        print(f"\nExtracting prototypes from native model...")
        classifier.extract_prototypes(task_id, native_model, train_loader, current_class_ids, device)
        
        # Merge LoRA states based on selected strategy
        print(f"\nMerging LoRA states using {Config.MERGE_STRATEGY} strategy...")
        new_lora_state = native_model.get_lora_state_dict()
        
        if Config.MERGE_STRATEGY == "sd_lora":
            merged_lora_state, magnitude_history, direction_history = merge_loras_sd_lora_inspired(
                merged_lora_state, new_lora_state, task_id + 1,
                magnitude_history, direction_history
            )
        elif Config.MERGE_STRATEGY == "orthogonal_projection":
            merged_lora_state = merge_loras_orthogonal_projection(
                merged_lora_state, new_lora_state, task_id + 1,
                projection_threshold=Config.OPCM_PROJECTION_THRESHOLD
            )
        else:
            raise ValueError(f"Unknown merge strategy: {Config.MERGE_STRATEGY}")
        
        # Update merged model with new merged LoRA state
        merged_model = LoRAViT(
            Config.MODEL_NAME, 
            rank=Config.LORA_RANK, 
            alpha=Config.LORA_ALPHA
        ).to(device)
        merged_model.set_lora_state_dict(merged_lora_state)
        
        # Refine prototypes in merged space
        print(f"\nRefining prototypes in merged space...")
        classifier.refine_prototypes_in_merged_space(
            native_model, merged_model, train_loader, current_class_ids, device
        )
        
        # Evaluate merged model on all tasks completed so far
        print(f"\nEvaluating merged model on all tasks completed so far...")
        full_test_dataset = get_full_test_dataset(transform)
        full_test_loader = DataLoader(
            full_test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
        )
        
        merged_accuracy = evaluate_merged_model(merged_model, full_test_loader, classifier, device, task_id)
        print(f"Merged model accuracy on all tasks (1-{task_id + 1}): {merged_accuracy:.4f}")
        
        # Run diagnostics
        if Config.SAVE_DIAGNOSTICS:
            print(f"\nRunning feature space diagnostics...")
            diag = diagnostics.analyze_feature_space_drift(
                task_id, native_model, merged_model, test_loader, current_class_ids,
                classifier.prototypes, device
            )
            diagnostics.add_diagnostics(diag)
        
        # Save checkpoint
        checkpoint_path = Config.CHECKPOINT_DIR / f"task_{task_id + 1}_checkpoint.pth"
        torch.save({
            'merged_lora_state': merged_lora_state,
            'prototypes': classifier.prototypes,
            'task_classes': classifier.task_classes,
            'results': results
        }, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path}")
        
        # Update results
        results['task_accuracies'].append(native_accuracy)
        results['merged_accuracies'].append(merged_accuracy)
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Task-wise accuracies: {results['task_accuracies']}")
    print(f"Merged model accuracies after each task: {results['merged_accuracies']}")
    print(f"\nFinal merged model accuracy: {results['merged_accuracies'][-1]:.4f}")
    
    # Print cumulative diagnostics
    if Config.SAVE_DIAGNOSTICS:
        diagnostics.print_cumulative_analysis()
    
    # Save final results
    results_path = Config.RESULTS_DIR / "final_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {results_path}")

def train_native_model(model, train_loader, device, task_id, merged_model=None):
    """
    Train native model for current task with optional distillation from merged model
    """
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.lora_layers.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS_PER_TASK
    )
    
    # Training loop
    for epoch in range(Config.EPOCHS_PER_TASK):
        model.train()
        if merged_model is not None:
            merged_model.eval()
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS_PER_TASK}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Get features from current model
            features_output = model(images, return_intermediate=Config.ENABLE_FEATURE_DISTILLATION)
            
            if Config.ENABLE_FEATURE_DISTILLATION:
                current_features = features_output['final']
                current_intermediate = features_output['intermediate']
            else:
                current_features = features_output
                current_intermediate = None
            
            # Compute classification loss
            loss = compute_classification_loss(current_features, labels, device)
            
            # Compute feature distillation loss if enabled
            if Config.ENABLE_FEATURE_DISTILLATION and merged_model is not None:
                with torch.no_grad():
                    old_features_output = merged_model(images, return_intermediate=True)
                    old_intermediate = old_features_output['intermediate']
                
                distillation_loss = compute_multi_layer_distillation(
                    current_intermediate, old_intermediate,
                    temperature=Config.DISTILL_TEMPERATURE
                )
                loss += Config.FEATURE_DISTILL_LAMBDA * distillation_loss
            
            # Compute Procrustes alignment loss if enabled
            if Config.PROCRUSTES_LAMBDA > 0 and merged_model is not None:
                with torch.no_grad():
                    old_features = merged_model(images)
                
                alignment_loss = compute_procrustes_alignment_loss(current_features, old_features)
                loss += Config.PROCRUSTES_LAMBDA * alignment_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            # (This is just for training monitoring, not the official evaluation)
            with torch.no_grad():
                # For simplicity, we'll just check if features are being learned
                pass
            
            progress_bar.set_postfix(loss=f"{epoch_loss / (batch_idx + 1):.4f}")
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch {epoch + 1}/{Config.EPOCHS_PER_TASK}, Loss: {epoch_loss / len(train_loader):.4f}")

def compute_classification_loss(features, labels, device):
    """
    Compute classification loss based on current features and labels
    """
    # This is a placeholder function - in the actual implementation, 
    # you would use the classifier to compute the loss
    # For simplicity, we'll return a dummy loss here
    return torch.tensor(0.0, device=device, requires_grad=True)

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on a specific task
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            
            # Placeholder evaluation - in actual implementation, use classifier
            # For simplicity, we'll return a dummy accuracy here
            dummy_preds = torch.randint(0, 100, (len(images),), device=device)
            correct += (dummy_preds == labels).sum().item()
            total += len(images)
    
    return correct / total

def evaluate_merged_model(model, test_loader, classifier, device, current_task_id):
    """
    Evaluate merged model on all tasks completed so far
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating merged model"):
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            
            # Get predictions from classifier
            predictions = classifier.predict(features, device)
            
            # Only count correct predictions for tasks completed so far
            end_class = (current_task_id + 1) * Config.CLASSES_PER_TASK
            mask = labels < end_class
            
            if mask.any():
                correct += (predictions[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
    
    return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    train()
