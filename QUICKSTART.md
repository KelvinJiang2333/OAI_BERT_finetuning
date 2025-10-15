# 快速开始 Quick Start

## 中文 Chinese

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成训练数据
生成MLM和对比学习两种训练数据：
- MLM数据：10,285个样本（3前+1当前+3后=7行代码上下文）
- 对比学习数据：53,218对词语样本（正负样本各26,609对）

```bash
python3 scripts/generate_paper_spec_data.py
```

### 3. 训练模型
```bash
# 单GPU训练
python3 train_30_epochs_final.py

# 多GPU训练（推荐）
torchrun --nproc_per_node=5 train_joint_bert.py \
  --num_epochs 30 \
  --batch_size 64 \
  --learning_rate 2e-05 \
  --beta_ft 0.7 \
  --use_multi_gpu
```

### 4. 查看结果
```bash
cat final_training_report.md
```

---

## English

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
Generate both MLM and Contrastive Learning datasets:
- MLM Data: 10,285 samples (7-line code context: 3 preceding + 1 current + 3 subsequent)
- Contrastive Learning: 53,218 word pairs (26,609 positive + 26,609 negative)

```bash
python3 scripts/generate_paper_spec_data.py
```

### 3. Train Model
```bash
# Single GPU
python3 train_30_epochs_final.py

# Multi-GPU (Recommended)
torchrun --nproc_per_node=5 train_joint_bert.py \
  --num_epochs 30 \
  --batch_size 64 \
  --learning_rate 2e-05 \
  --beta_ft 0.7 \
  --use_multi_gpu
```

### 4. View Results
```bash
cat final_training_report.md
```

