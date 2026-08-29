<div align="center">

# ViT+LoRA 增量学习 | ViT-LoRA-Incremental-Learning

### ViT + LoRA incremental learning.

Progressive fusion for class-incremental learning — 3 tasks on CIFAR-100 reaching 87.58%.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**ViT-LoRA-Incremental-Learning** implements **class-incremental learning** with **ViT + LoRA** using progressive fusion — reaching **87.58%** across 3 tasks on **CIFAR-100**.

> [!NOTE]
> 中文项目：ViT + LoRA 增量学习——渐进式融合实现类增量学习，3 任务 CIFAR-100 达 87.58%。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/ViT-LoRA-Incremental-Learning.git
cd ViT-LoRA-Incremental-Learning

pip install -r requirements.txt

# run the incremental pipeline
python src/main.py
# or the standalone entry
python cl2.py
```

---

## Features

- **ViT + LoRA** — parameter-efficient incremental learning.
- **Progressive fusion** — merge/distill to fight forgetting.
- **Strong accuracy** — 87.58% on 3-task CIFAR-100.

---

## Project Structure

```
ViT-LoRA-Incremental-Learning/
├── src/
│   ├── main.py, cl2.py
│   ├── losses/          # distillation, prototype_losses
│   ├── classifiers/     # prototype_classifier
│   ├── merging/         # merging
│   └── diagnostics/
├── configs/config.py
└── docs/                # technical reports
```

---

## 技术实现细节

### 架构概览

项目采用模块化设计，核心目录包括：**configs, docs, src**。

### 核心类与模块

- **TaskDataset**
- **Config**
- **LoRALayer**
- **LoRAViT**
- **PrototypeClassifier**

### 关键函数

- `set_seed`, `get_task_datasets`, `get_full_test_dataset`, `train`, `forward`, `make_lora_forward`, `extract_prototypes`, `refine_prototypes_in_merged_space`, `predict`

### 技术栈与依赖

**核心框架/库**：NumPy, PyTorch

**主要 import**：
```python
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
```

### 实现要点

- 以 `TaskDataset` 为核心类，封装主要业务逻辑
- 通过 `set_seed` 等函数实现核心流程编排
- 基于 NumPy, PyTorch 构建，技术栈成熟稳定
- 代码结构清晰，模块间低耦合，便于扩展和维护

---
## License

MIT — free to use, modify and distribute.
