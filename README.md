# 🎯 ViT+LoRA 增量学习 | ViT with LoRA for Incremental Learning

> **用 LoRA 低秩适配让 Vision Transformer 学会增量学习——新类别来了只更新少量参数，旧知识不遗忘。**
>
> *Enable Vision Transformer for incremental learning with LoRA low-rank adaptation — learn new categories with minimal parameter updates, no catastrophic forgetting.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🧩 **LoRA 低秩适配** | LoRA Adaptation | 只训练 0.1% 参数，效果媲美全量微调 |
| 🧠 **增量学习** | Incremental Learning | 分阶段学习新类别，缓解灾难性遗忘 |
| 👁️ **ViT 骨干** | Vision Transformer | 基于 Transformer 的图像分类骨干 |
| 💾 **参数高效** | Parameter-Efficient | 每个任务只需保存少量 LoRA 权重 |
| 📊 **完整实验** | Full Experiments | CIFAR-100 上多阶段增量学习实验 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-orange?logo=numpy)

---

## 📊 方法对比 | Method Comparison

| 方法 | 参数量 | 旧类别保留 | 新类别学习 | 存储开销 |
|------|--------|-----------|-----------|---------|
| 全量微调 | 🔴 100% | ❌ 遗忘 | ✅ 好 | 🔴 每任务全量 |
| 冻结特征 | 🟢 0% | ✅ 保留 | 🟡 一般 | 🟢 仅分类头 |
| **LoRA 增量 (本项目)** | 🟢 ~0.1% | ✅ 好 | ✅ 好 | 🟢 仅 LoRA 权重 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/ViT-LoRA-Incremental-Learning.git
cd ViT-LoRA-Incremental-Learning
pip install -r requirements.txt
python train.py --dataset cifar100 --phases 5 --lora_rank 8
```

---

## 📂 项目结构 | Project Structure

```
ViT-LoRA-Incremental-Learning/
├── train.py                   # 训练入口
├── requirements.txt           # 依赖
├── models/
│   ├── vit.py                 # ViT 模型
│   └── lora.py                # LoRA 适配层
├── data/
│   └── cifar100.py            # CIFAR-100 数据加载
├── incremental/
│   ├── strategy.py            # 增量学习策略
│   └── rehearsal.py           # 样本回放
└── results/                   # 实验结果
```

---

## 🔬 核心原理 | Core Idea

### LoRA 低秩适配 | Low-Rank Adaptation

LoRA 在 Transformer 的注意力权重矩阵上添加低秩分解矩阵：

```
原始权重: W ∈ R^{d×d}
LoRA:    W + BA, 其中 B ∈ R^{d×r}, A ∈ R^{r×d}, r << d

训练时: W 冻结, 只更新 A 和 B (参数量 = 2dr ≈ 0.1% W)
推理时: 权重合并为 W + BA, 无额外推理开销
```

### 增量学习策略 | Incremental Strategy

1. **阶段划分**：CIFAR-100 分为 5 个阶段，每阶段 20 类
2. **LoRA 适配**：每个新阶段添加新的 LoRA 适配器
3. **知识蒸馏**：用旧模型指导新模型，保留旧类别知识
4. **样本回放**：保留少量旧类别样本，防止遗忘

---

## 🎯 应用场景 | Use Cases

- 📱 **移动端部署**：参数高效，适合资源受限设备
- 🏭 **工业质检**：新产品类别来了只需快速适配
- 🩺 **医学影像**：新病种数据逐步积累，增量学习
- 🚗 **自动驾驶**：新场景、新天气条件的持续学习

---

## 📚 参考文献 | References

- Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
- Rebuffi, S. A., et al. "iCaRL: Incremental Classifier and Representation Learning." CVPR 2017.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **参数高效 + 增量学习的 ViT 方案，Star ⭐ 支持开源！**
