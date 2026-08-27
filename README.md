<div align="center">

# 🖼️ ViT-LoRA-Incremental-Learning

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

## License

MIT — free to use, modify and distribute.
