import torch
from pathlib import Path

# 设置使用国内镜像
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 使用GPU 1（GPU 0被其他进程占用）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
