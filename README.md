# 🧠 ViT-LoRA Incremental Learning | 基于 ViT+LoRA 渐进式融合的类增量持续学习系统

> **Learn new tasks without forgetting old ones. Vision Transformer + LoRA + Progressive Merging achieves 87.58% on 5-task CIFAR-100.**
>
> 学习新任务而不遗忘旧任务。Vision Transformer + LoRA + 渐进式融合，在 CIFAR-100 五任务上达到 87.58% 准确率。

---

## 🌟 Why This Project? | 项目亮点

Class-incremental learning (CIL) faces the **stability-plasticity dilemma**: models must learn new classes while retaining knowledge of old ones. This project tackles it with a powerful combination of **Vision Transformer (ViT) backbone**, **LoRA parameter-efficient adaptation**, and a novel **KNOTS progressive merging strategy** — achieving state-of-the-art 87.58% average accuracy on 5-task CIFAR-100 (100 classes total).

类增量学习（CIL）面临**稳定性-可塑性困境**：模型必须在学习新类的同时保留旧类知识。本项目通过 **Vision Transformer (ViT) 骨干网络**、**LoRA 参数高效适配** 和创新的 **KNOTS 渐进式融合策略** 的强大组合，在 CIFAR-100 五任务（共 100 类）上达到 87.58% 的平均准确率。

| Metric | Value |
|--------|-------|
| **Dataset** | CIFAR-100 (5 tasks × 20 classes) |
| **Backbone** | ViT-Base/16 (google/vit-base-patch16-224) |
| **Best Accuracy** | **87.58%** (50 epochs enhanced) |
| **Baseline Accuracy** | 87.56% (40 epochs) |
| **Forgetting Rate (T1→T2)** | -5.82% |
| **Forgetting Rate (T4→T5)** | -1.66% |
| **Training Time** | ~8 hours (single GPU) |

---

## 🏗️ Architecture | 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input Image (224×224×3)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              ViT-Base/16 Backbone (Frozen + LoRA)                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Patch Embedding + Position Embedding                        │  │
│  │  12× Transformer Encoder Blocks                              │  │
│  │  ┌───────────────────────────────────────────────────────┐  │  │
│  │  │  Multi-Head Self-Attention (LoRA: r=32, α=64)       │  │  │
│  │  │  → Query, Value projections adapted via low-rank       │  │  │
│  │  └───────────────────────────────────────────────────────┘  │  │
│  │  MLP + LayerNorm + Residual                                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Prototype Classifier (Cosine Similarity)             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Class Prototypes (updated via momentum refinement)          │  │
│  │  Temperature-scaled cosine similarity (τ=0.05)              │  │
│  │  Hungarian matching for task-invariant class ordering        │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              KNOTS Progressive LoRA Merging                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Old LoRA weights (0.6) + New LoRA weights (0.4)           │  │
│  │  Random orthogonal initialization (CL-LoRA inspired)        │  │
│  │  Minimum merged rank constraint (r_min=32)                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Innovations | 核心创新

### 1. KNOTS Progressive LoRA Merging | KNOTS 渐进式 LoRA 融合

Instead of keeping separate LoRA adapters per task (which grows linearly), KNOTS merges old and new LoRA weights with a weighted combination:

```python
merged_lora = 0.6 * old_lora + 0.4 * new_lora
```

This maintains a **single compact adapter** while preserving knowledge from previous tasks.

### 2. Prototype-Based Classification | 基于原型的分类

Uses cosine similarity to learnable class prototypes instead of a linear classifier head. Prototypes are updated via **momentum refinement** (12 iterations, momentum=0.85), providing stable decision boundaries across tasks.

### 3. Multi-Loss Training Strategy | 多损失训练策略

| Loss Component | Weight | Purpose |
|----------------|--------|---------|
| Cross-Entropy | 1.0 | Task classification |
| Knowledge Distillation | 2.5 | Preserve old task logits |
| Contrastive Learning | 0.8 | Inter-class separation |
| Prototype Refinement | — | Stable prototype updates |

### 4. Hungarian Matching | 匈牙利匹配

Uses `scipy.optimize.linear_sum_assignment` to align class prototypes across tasks, ensuring task-invariant class ordering.

---

## 📊 Results | 实验结果

### Cumulative Accuracy | 累积准确率

| Task | 50 Epochs (Best) | 40 Epochs (Baseline) | OPCM (Failed) |
|------|-------------------|----------------------|---------------|
| Task 1 | 96.85% | 96.85% | 96.80% |
| Task 1-2 | **91.03%** | 90.88% | 90.97% |
| Task 1-3 | **89.88%** | 89.83% | 90.00% |
| Task 1-4 | **89.24%** | 89.22% | 88.64% |
| **Task 1-5** | **87.58%** | 87.56% | 86.02% |

### Forgetting Analysis | 遗忘分析

| Transition | 50 Epochs | 40 Epochs |
|------------|-----------|-----------|
| Task 1→2 | -5.82% | -5.97% |
| Task 2→3 | -1.15% | -1.05% |
| Task 3→4 | -0.64% | -0.61% |
| Task 4→5 | -1.66% | -1.66% |

**Key insight**: Forgetting is concentrated in the first task transition (T1→T2), then stabilizes to <2% per task.

---

## 📁 Project Structure | 项目结构

```
ViT-LoRA-Incremental-Learning/
├── cl2.py                          # Main training script (76KB, complete pipeline)
├── requirements.txt                # Python dependencies
│
├── src/                            # Modular source code
│   ├── main.py                     # Entry point with argument parsing
│   ├── models/
│   │   └── lora.py                 # LoRA layer implementation
│   ├── classifiers/
│   │   └── prototype_classifier.py # Prototype-based classifier
│   ├── losses/
│   │   ├── distillation.py         # Knowledge distillation loss
│   │   └── prototype_losses.py     # Prototype refinement losses
│   ├── merging/
│   │   └── merging.py              # KNOTS progressive merging
│   ├── diagnostics/
│   │   └── diagnostics.py          # Training diagnostics & logging
│   └── utils/
│       └── alignment.py            # Hungarian matching & alignment
│
├── configs/
│   └── config.py                   # Hyperparameter configuration
│
└── docs/
    ├── BEST_VERSION_TECHNICAL_REPORT.md  # Full technical report
    ├── 实验结果对比表.md                  # Complete results comparison
    ├── 使用说明_README.md                 # Usage instructions
    └── 最佳版本技术说明_中文摘要.md        # Chinese technical summary
```

---

## 🚀 Quick Start | 快速开始

### 1. Install Dependencies | 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Configure | 配置

Edit `configs/config.py` or use the default configuration:

```python
class Config:
    NUM_TASKS = 5
    CLASSES_PER_TASK = 20
    MODEL_NAME = "google/vit-base-patch16-224"
    LORA_RANK = 32
    LORA_ALPHA = 64
    BATCH_SIZE = 96
    EPOCHS_PER_TASK = 50
    LR = 2e-4
    MERGE_STRATEGY = "knots"
    MERGE_WEIGHT_OLD = 0.6
    MERGE_WEIGHT_NEW = 0.4
```

### 3. Train | 训练

```bash
python cl2.py
```

### 4. Monitor | 监控

Training logs include per-task accuracy, forgetting rates, and prototype statistics.

---

## 🔧 Hyperparameter Guide | 超参数指南

### Best Performance (87.58%) | 最佳性能配置

```python
EPOCHS_PER_TASK = 50
PROTOTYPE_REFINEMENT_ITERATIONS = 12
REFINEMENT_MOMENTUM = 0.85
CONTRASTIVE_LAMBDA = 0.8
MERGE_STRATEGY = "knots"
MERGE_WEIGHT_OLD = 0.6
MERGE_WEIGHT_NEW = 0.4
```

### Best Trade-off (87.56%, 33% faster) | 性价比配置

```python
EPOCHS_PER_TASK = 40
PROTOTYPE_REFINEMENT_ITERATIONS = 8
REFINEMENT_MOMENTUM = 0.8
CONTRASTIVE_LAMBDA = 0.5
MERGE_STRATEGY = "knots"
```

### Not Recommended | 不推荐配置

❌ **OPCM strategy** (86.02%) — Severe degradation in later tasks (T4-T5)
❌ **Round 2/3 configs** (85.18%, 84.57%) — Suboptimal hyperparameters

---

## 📈 Training Curve | 训练曲线

```
Accuracy (%)
100 ┤
 95 ┤ ● Task 1 (96.85%)
    │
 90 ┤   ● Task 1-2 (91.03%)
    │     ● Task 1-3 (89.88%)
 85 ┤       ● Task 1-4 (89.24%)
    │         ● Task 1-5 (87.58%)
 80 ┤
    └──────────────────────────
      T1   T2   T3   T4   T5
```

---

## 🛠️ Tech Stack | 技术栈

| Component | Technology |
|-----------|-----------|
| Backbone | HuggingFace ViT-Base/16 |
| Adaptation | Custom LoRA (rank=32, alpha=64) |
| Framework | PyTorch + Transformers |
| Dataset | CIFAR-100 (torchvision) |
| Optimization | AdamW + Cosine Annealing + Warmup |
| Matching | SciPy Hungarian Algorithm |
| Logging | tqdm + custom diagnostics |

---

## 📝 Citation | 引用

```bibtex
@misc{vit_lora_incremental2025,
  title={ViT-LoRA Incremental Learning: Progressive Merging for Class-Incremental Learning},
  author={Windyhhh},
  year={2025},
  howpublished={\url{https://github.com/Windyhhh/ViT-LoRA-Incremental-Learning}}
}
```

---

## 📄 License | 许可证

MIT License — free to use, modify, and distribute.

---

<div align="center">

**Built with 🧠 for continual learning research**

[Report Bug](https://github.com/Windyhhh/ViT-LoRA-Incremental-Learning/issues) · [Request Feature](https://github.com/Windyhhh/ViT-LoRA-Incremental-Learning/issues)

</div>
