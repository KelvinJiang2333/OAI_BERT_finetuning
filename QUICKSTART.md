# 快速开始 Quick Start

## 中文 Chinese

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 生成数据
```bash
python3 generate_paper_spec_data.py
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

### 2. Generate Data
```bash
python3 generate_paper_spec_data.py
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

