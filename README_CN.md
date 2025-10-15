# 通信领域BERT嵌入模型训练项目

<div align="center">

[English](README.md) | [简体中文](README_CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 🎯 项目概述

专为通信领域设计的BERT嵌入模型训练系统，采用创新的**词语级别对比学习**方法，支持多GPU分布式训练。

### 核心特性

- 🔧 **分布式训练**: 支持多GPU并行训练，优化资源利用
- 📊 **性能监控**: 实时GPU性能监控和详细的训练指标统计
- 🎯 **领域优化**: 针对通信领域的专用词汇表和增强tokenizer
- 🆕 **词语级别对比学习**: 创新的同行代码词语配对策略
- 📈 **详细报告**: 完整的训练过程记录和结果分析

## 🚀 快速开始

### 环境要求

```bash
# Python 3.7 或更高版本
# CUDA 11.0 或更高版本（GPU支持）
# 推荐使用多GPU环境（已在NVIDIA A800上测试）
```

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/KelvinJiang2333/OAI_BERT_finetuning.git
cd OAI_BERT_finetuning

# 安装依赖
pip install -r requirements.txt
```

### 数据生成

生成完整的训练数据集（包含MLM和对比学习两种类型）：

```bash
# 生成MLM训练数据 + 词语级别对比学习数据集
python3 scripts/generate_paper_spec_data.py
```

**生成的数据包括：**
- **MLM数据**：10,285个样本，采用7行代码上下文窗口（3前+1当前+3后）
- **对比学习数据**：53,218对词语样本（正样本26,609对 + 负样本26,609对）

### 模型训练

```bash
# 单GPU训练
python3 scripts/train_30_epochs_final.py

# 多GPU分布式训练（推荐）
torchrun --nproc_per_node=5 --nnodes=1 scripts/train_joint_bert.py \
  --num_epochs 30 \
  --batch_size 64 \
  --learning_rate 2e-05 \
  --beta_ft 0.7 \
  --use_multi_gpu \
  --use_enhanced_tokenizer
```

### 查看结果

```bash
# 训练报告
cat final_training_report.md

# 性能数据  
cat final_30_epoch_output/training_performance_report.json
```

## 📊 数据集特点

- **MLM训练数据**: 10,285个样本，采用7行代码上下文窗口
- **对比学习数据**: 53,218对词语样本（正样本26,609对 + 负样本26,609对）
- **词语级别创新**: 同一代码行内的词语配对作为正样本
- **高质量负样本**: 使用9,474个常用英文词汇作为负样本
- **智能过滤**: 自动过滤单字母变量和标点符号

## 🛠️ 技术架构

### 模型架构

- **基础模型**: BERT-base-uncased
- **Transformer层数**: 12
- **注意力头数**: 12
- **隐藏维度**: 768
- **前馈维度**: 3072

### 训练策略

- **联合损失**: 结合MLM和对比学习损失，权重比例为7:3
- **优化器**: AdamW，学习率2e-5
- **批次大小**: 每GPU 64（5个GPU总批次大小：320）
- **训练轮数**: 30 epochs
- **MLM掩码比例**: 15%

### 损失函数

```
L_total = β_ft × L_MLM + (1 - β_ft) × L_InfoNCE
其中 β_ft = 0.7
```

## 📁 项目结构

```
OAI_BERT_finetuning/
├── scripts/                           # 🎯 核心脚本目录
│   ├── train_30_epochs_final.py       # 主训练脚本
│   ├── train_joint_bert.py            # 联合训练实现
│   ├── train_with_monitoring.py       # 训练监控
│   ├── generate_paper_spec_data.py    # MLM和对比学习数据生成
│   └── create_enhanced_tokenizer.py   # 增强tokenizer创建
├── 📊 模型和工具
│   ├── joint_bert_model.py            # 联合训练模型定义
│   └── tokenizer_utils.py             # tokenizer工具
├── 🗃️ 数据和资源
│   ├── paper_spec_training_data/      # 训练数据
│   ├── enhanced_communication_tokenizer/ # 增强tokenizer
│   ├── knowledge_base/                # 代码知识库
│   ├── pretrained_models/             # 预训练模型目录
│   └── google-10000-english-no-swears.txt # 英文词典
├── 📋 输出和报告
│   ├── final_30_epoch_output/         # 训练输出
│   └── final_training_report.md       # 训练报告
└── 📄 配置文档
    ├── README.md                      # 英文文档
    ├── README_CN.md                   # 中文文档
    ├── QUICKSTART.md                  # 快速开始指南
    ├── LICENSE                        # MIT许可证
    └── requirements.txt               # 依赖配置
```

## 🔬 词语级别对比学习创新

### 核心创新

传统对比学习在句子级别进行，本项目创新性地引入**词语级别对比学习**：

- **正样本配对**: 同一代码行内的词语（语义关联性更强）
- **负样本配对**: 代码词语 vs 常用英文词汇（对比度明显）
- **质量控制**: 过滤无意义的token和符号
- **本地化部署**: 使用本地英文词典，无外部依赖

### 优势

1. **更精确的词汇表示**: 直接学习代码词汇间的语义关系
2. **更好的泛化能力**: 通过高质量负样本提升判别能力
3. **更强的代码理解**: 针对编程领域优化的对比学习策略

## 🖥️ 硬件要求

### 最低配置

- **GPU**: 支持CUDA的GPU，显存8GB以上
- **内存**: 16GB系统内存
- **存储**: 10GB可用磁盘空间

### 推荐配置

- **GPU**: 多GPU环境（5× NVIDIA A800或类似）
- **内存**: 32GB以上系统内存
- **存储**: 50GB以上SSD存储

## 🤝 贡献指南

欢迎贡献！请随时提交Pull Request。对于重大更改，请先开issue讨论您想要改变的内容。

### 如何贡献

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

## 📝 引用

如果您在研究中使用了本项目代码，请引用：

```bibtex
@software{bert_telecom_finetuning,
  title={通信领域BERT嵌入模型训练：词语级别对比学习},
  author={Haihang Jiang},
  year={2025},
  url={https://github.com/KelvinJiang2333/OAI_BERT_finetuning}
}
```

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [Hugging Face Transformers](https://github.com/huggingface/transformers) 提供BERT实现
- [PyTorch](https://pytorch.org/) 深度学习框架
- 开源社区提供的各种工具和库

## 📧 联系方式

如有问题或反馈，请在GitHub上提交issue。

## 🔍 详细文档

### 数据生成模块

**MLM数据生成：**
- **上下文窗口**: 为每个代码行提取7行上下文（3前+1当前+3后）
- **掩码策略**: 随机掩码15%的token，80%用[MASK]，10%随机替换，10%保持不变
- **样本数量**: 10,285个高质量MLM训练样本

**对比学习数据生成：**
- **词语提取**: 使用正则表达式提取代码中的有效词语
- **正样本生成**: 同一代码行内的词语两两配对（26,609对）
- **负样本生成**: 代码词语与常用英文词汇配对（26,609对）
- **质量控制**: 自动过滤无意义的token和符号

### 模型训练模块

- **联合损失**: 结合MLM和对比学习损失，权重比例为7:3
- **分布式训练**: 支持多GPU并行，优化资源利用
- **实时监控**: GPU性能、能耗统计和训练指标记录

## 🚀 技术亮点

### 词语级别对比学习

- **创新方法**: 突破传统句子级别，采用词语级别对比学习
- **精确配对**: 同一代码行内词语语义相关性更强
- **高效训练**: 避免句子内多概念混杂的噪声问题

### 分布式训练优化

- **多GPU支持**: 优化的分布式数据并行训练
- **内存管理**: 高效的GPU内存使用和数据加载
- **性能监控**: 实时GPU利用率、显存、功耗监控

### 智能数据处理

- **自动过滤**: 智能识别和过滤无意义的token
- **本地词典**: 使用本地常用英文词典，无外部依赖
- **质量保证**: 多层次的数据质量检查和验证

## 📈 预期效果

- **精确词汇表示**: 针对代码词汇优化的语义表示
- **强泛化能力**: 通过高质量负样本提升模型鲁棒性
- **领域适应性**: 专门针对通信领域代码的理解能力

## 🔧 高级用法

### 自定义训练参数

```bash
python3 scripts/train_30_epochs_final.py \
  --num_epochs 50 \
  --batch_size 128 \
  --learning_rate 1e-05 \
  --beta_ft 0.8
```

### 使用自定义数据集

1. 准备您的代码数据文件
2. 修改 `scripts/generate_paper_spec_data.py` 中的数据路径
3. 运行数据生成脚本
4. 开始训练

### 模型评估

```python
from transformers import AutoModel, AutoTokenizer

# 加载训练好的模型
model = AutoModel.from_pretrained('./final_30_epoch_output/final_model')
tokenizer = AutoTokenizer.from_pretrained('./enhanced_communication_tokenizer')

# 使用模型
inputs = tokenizer("Your code snippet here", return_tensors="pt")
outputs = model(**inputs)
```

## 🐛 故障排除

### 常见问题

**Q: CUDA out of memory 错误**
```bash
# 减小批次大小
python3 scripts/train_30_epochs_final.py --batch_size 32
```

**Q: 多GPU训练报错**
```bash
# 检查NCCL环境
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

**Q: tokenizer加载失败**
```bash
# 重新生成tokenizer
python3 scripts/create_enhanced_tokenizer.py
```

---

<div align="center">
<b>专为通信领域优化的代码语义理解模型训练系统</b>
</div>

