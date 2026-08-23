# CIFAR-100 类增量学习项目 - 使用说明

## 📋 项目概述

本项目实现了基于LoRA的CIFAR-100类增量学习(Class-Incremental Learning),通过渐进式LoRA融合实现O(1)空间复杂度,在不使用数据回放和任务ID推理的条件下,达到了**87.58%**的最佳准确率。

**任务设置**: 5个任务,每个任务20个类别,共100个类别

---

## 🏆 最佳结果

**版本**: 50 Epochs Enhanced  
**最终准确率**: **87.58%**

**完整结果**:
```
Task 1:   96.85%
Task 1-2: 91.03% (-5.82%)
Task 1-3: 89.88% (-1.15%)
Task 1-4: 89.24% (-0.64%)
Task 1-5: 87.58% (-1.66%)
```

---

## 📁 项目结构

```
.
├── cl2.py                                          # 主训练脚本 (最新版本,包含所有策略)
├── cl2_BEST_50epochs_with_all_strategies.py       # 最佳版本备份
├── cl2_v1_50epochs_enhanced_BEST_RESULT_87.58.py  # 50 epochs增强版本
│
├── training_BEST_50epochs_87.58_percent.log        # 最佳结果训练日志 ⭐
├── training_BASELINE_40epochs_87.56_percent.log    # Baseline训练日志
├── training_OPCM_FAILED_86.02_percent.log          # OPCM失败案例
│
├── 文档索引_README.md                              # 技术文档导航 ⭐
├── 最佳版本技术说明_中文摘要.md                    # 中文技术摘要 ⭐
├── BEST_VERSION_TECHNICAL_REPORT.md                # 完整英文技术报告
├── 实验结果对比表.md                               # 所有版本对比
├── 使用说明_README.md                              # 本文档 ⭐
│
├── data/                                           # CIFAR-100数据集
├── archive_backup/                                 # 备份文件夹
│   ├── old_scripts/                                # 旧版本脚本
│   ├── old_logs/                                   # 旧训练日志
│   ├── old_docs/                                   # 旧说明文档
│   └── test_scripts/                               # 测试脚本
│
└── requirements.txt                                # Python依赖
```

---

## 🚀 快速开始

### 1. 环境配置

**Python版本**: 3.8+

**安装依赖**:
```bash
pip install -r requirements.txt
```

**主要依赖**:
- torch >= 2.0.0
- transformers >= 4.30.0
- torchvision
- numpy
- tqdm
- scikit-learn

---

### 2. 运行最佳版本

**使用最佳配置训练**:
```bash
python cl2.py
```

**配置说明**:
- 默认使用50 epochs配置
- 自动下载CIFAR-100数据集
- 训练日志保存到 `training_opcm_test.log`
- 模型checkpoint保存到 `checkpoints_*/`

**预计训练时间**: 约6-8小时 (单GPU, V100/A100)

---

### 3. 修改配置

编辑 `cl2.py` 中的配置参数:

```python
# 训练超参数
BATCH_SIZE = 96
EPOCHS_PER_TASK = 50              # 修改训练周期
LR = 2e-4
WEIGHT_DECAY = 0.015

# 原型细化
PROTOTYPE_REFINEMENT_ITERATIONS = 12    # 修改迭代次数
REFINEMENT_MOMENTUM = 0.85              # 修改momentum
CONTRASTIVE_LAMBDA = 0.8                # 修改对比学习权重

# LoRA融合策略
MERGE_STRATEGY = "knots"                # 可选: "knots", "orthogonal_projection", "sd_lora"
```

---

## 📊 版本说明

### 主要版本

| 版本 | 文件名 | 准确率 | 说明 |
|------|--------|--------|------|
| **最佳版本** | `cl2.py` | **87.58%** | 50 epochs + 增强原型细化 ⭐ |
| Baseline | `cl2.py` (40 epochs配置) | 87.56% | 40 epochs baseline |
| OPCM | `cl2.py` (OPCM策略) | 86.02% | 正交投影策略 (失败) |

### 训练日志

| 日志文件 | 准确率 | 说明 |
|---------|--------|------|
| `training_BEST_50epochs_87.58_percent.log` | **87.58%** | 最佳结果 ⭐ |
| `training_BASELINE_40epochs_87.56_percent.log` | 87.56% | Baseline |
| `training_OPCM_FAILED_86.02_percent.log` | 86.02% | OPCM失败案例 |

---

## 🔧 核心技术

### 1. 渐进式LoRA融合
- **目标**: O(1)空间复杂度
- **方法**: 递归融合 `merged = 0.6 × merged_old + 0.4 × lora_new`
- **效果**: 节省80%存储空间

### 2. 多层知识蒸馏
- **蒸馏层**: [3, 6, 9, 11]
- **权重**: λ_distill = 2.5
- **效果**: 特征漂移 < 0.15

### 3. 增强原型细化
- **迭代次数**: 12次 (vs 8次 baseline)
- **Momentum**: 0.85 (vs 0.8 baseline)
- **对比权重**: 0.8 (vs 0.5 baseline)
- **效果**: 原型对齐度 > 0.73

### 4. 扩展训练周期
- **Epochs**: 50 (vs 40 baseline)
- **效果**: Task 2改进 +0.15%

### 5. 多原型策略
- **子原型数**: 7个/类
- **方法**: K-means聚类
- **效果**: 捕获类内多样性

---

## 📖 文档导航

### 快速了解 (30分钟)
1. **使用说明_README.md** (本文档) - 5分钟
2. **最佳版本技术说明_中文摘要.md** - 15分钟
3. **实验结果对比表.md** - 10分钟

### 深入研究 (2-3小时)
1. **文档索引_README.md** - 完整导航
2. **BEST_VERSION_TECHNICAL_REPORT.md** - 完整技术报告
3. **实验结果对比表.md** - 所有版本对比

---

## 💡 使用建议

### 推荐配置

**最佳性能** (87.58%):
```python
EPOCHS_PER_TASK = 50
PROTOTYPE_REFINEMENT_ITERATIONS = 12
REFINEMENT_MOMENTUM = 0.85
CONTRASTIVE_LAMBDA = 0.8
MERGE_STRATEGY = "knots"
```

**性价比** (87.56%, 训练时间-25%):
```python
EPOCHS_PER_TASK = 40
PROTOTYPE_REFINEMENT_ITERATIONS = 8
REFINEMENT_MOMENTUM = 0.8
CONTRASTIVE_LAMBDA = 0.5
MERGE_STRATEGY = "knots"
```

### 不推荐

❌ **OPCM策略** (86.02%)
- 后期任务严重退化
- 不稳定

---

## 🔍 常见问题

### Q1: 如何查看训练进度?

**方法1**: 实时监控日志
```bash
tail -f training_opcm_test.log | grep "Overall CIL Accuracy"
```

**方法2**: 使用监控脚本
```bash
./wait_for_results.sh
```

### Q2: 训练中断如何恢复?

训练会自动保存checkpoint到 `checkpoints_*/` 目录,但当前版本不支持自动恢复。建议:
1. 检查日志确认完成的任务
2. 修改代码跳过已完成的任务
3. 重新运行

### Q3: 如何修改任务数量?

修改 `cl2.py` 中的 `NUM_TASKS` 参数:
```python
NUM_TASKS = 5  # 修改为你需要的任务数
CLASSES_PER_TASK = 100 // NUM_TASKS
```

### Q4: 如何使用其他数据集?

需要修改数据加载部分:
1. 修改 `load_cifar100_incremental()` 函数
2. 调整类别数量和任务划分
3. 修改数据预处理

### Q5: GPU内存不足怎么办?

减小batch size:
```python
BATCH_SIZE = 64  # 或更小
```

---

## 📈 性能对比

### vs Baseline (40 epochs)

| 任务 | 50 Epochs | 40 Epochs | 改进 |
|------|-----------|-----------|------|
| Task 1 | 96.85% | 96.85% | 0.00% |
| Task 1-2 | **91.03%** | 90.88% | **+0.15%** ✅ |
| Task 1-3 | **89.88%** | 89.83% | **+0.05%** ✅ |
| Task 1-4 | **89.24%** | 89.22% | **+0.02%** ✅ |
| Task 1-5 | **87.58%** | 87.56% | **+0.02%** ✅ |

### vs OPCM

| 任务 | 50 Epochs | OPCM | 差异 |
|------|-----------|------|------|
| Task 1-5 | **87.58%** | 86.02% | **+1.56%** ✅ |

---

## 🛠️ 高级用法

### 1. 测试不同融合策略

```python
# 在cl2.py中修改
MERGE_STRATEGY = "orthogonal_projection"  # OPCM策略
# 或
MERGE_STRATEGY = "sd_lora"  # SD-LoRA策略
```

### 2. 调整原型细化参数

```python
# 更激进的细化
PROTOTYPE_REFINEMENT_ITERATIONS = 15
REFINEMENT_MOMENTUM = 0.9
CONTRASTIVE_LAMBDA = 1.0

# 更保守的细化
PROTOTYPE_REFINEMENT_ITERATIONS = 8
REFINEMENT_MOMENTUM = 0.8
CONTRASTIVE_LAMBDA = 0.5
```

### 3. 修改蒸馏层

```python
# 只蒸馏最后一层
DISTILL_LAYERS = [11]

# 蒸馏所有层
DISTILL_LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

---

## 📝 引用

如果使用本项目,请引用:

```bibtex
@misc{cifar100_lora_cil_2025,
  title={CIFAR-100 Class-Incremental Learning with Progressive LoRA Merging},
  author={Your Name},
  year={2025},
  note={Best accuracy: 87.58\%}
}
```

---

## 📧 联系方式

- **项目**: CIFAR-100 Class-Incremental Learning
- **最佳准确率**: 87.58%
- **更新日期**: 2025-11-26

---

## 🔗 相关资源

### 技术文档
- `文档索引_README.md` - 完整文档导航
- `最佳版本技术说明_中文摘要.md` - 中文技术摘要
- `BEST_VERSION_TECHNICAL_REPORT.md` - 英文技术报告
- `实验结果对比表.md` - 版本对比

### 参考论文
1. LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
2. Prototypical Networks (NeurIPS 2017)
3. Knowledge Distillation (2015)

---

**祝使用愉快! 🎉**

