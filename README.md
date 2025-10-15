# BERT Fine-tuning for Telecommunication Domain

<div align="center">

[English](README.md) | [简体中文](README_CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## 🎯 Overview

A BERT-based embedding model training system specifically designed for the telecommunication domain, featuring **word-level contrastive learning** and multi-GPU distributed training support.

### Key Features

- 🔧 **Distributed Training**: Multi-GPU parallel training with optimized resource utilization
- 📊 **Performance Monitoring**: Real-time GPU monitoring and detailed training metrics
- 🎯 **Domain-Optimized**: Specialized vocabulary and enhanced tokenizer for telecom
- 🆕 **Word-Level Contrastive Learning**: Novel same-line code word pairing strategy
- 📈 **Comprehensive Reports**: Complete training logs and result analysis

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.7 or higher
# CUDA 11.0 or higher (for GPU support)
# Multiple GPUs recommended (tested on NVIDIA A800)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/KelvinJiang2333/OAI_BERT_finetuning.git
cd OAI_BERT_finetuning

# Install dependencies
pip install -r requirements.txt
```

### Data Generation

Generate complete training datasets (both MLM and Contrastive Learning):

```bash
# Generate MLM training data + Word-level contrastive learning dataset
python3 scripts/generate_paper_spec_data.py
```

**Generated datasets include:**
- **MLM Data**: 10,285 samples with 7-line code context window (3 preceding + 1 current + 3 subsequent)
- **Contrastive Learning Data**: 53,218 word pairs (26,609 positive + 26,609 negative)

### Training

```bash
# Single GPU training
python3 train_30_epochs_final.py

# Multi-GPU distributed training (recommended)
torchrun --nproc_per_node=5 --nnodes=1 train_joint_bert.py \
  --num_epochs 30 \
  --batch_size 64 \
  --learning_rate 2e-05 \
  --beta_ft 0.7 \
  --use_multi_gpu \
  --use_enhanced_tokenizer
```

### View Results

```bash
# Training report
cat final_training_report.md

# Performance metrics
cat final_30_epoch_output/training_performance_report.json
```

## 📊 Dataset Characteristics

- **MLM Training Data**: 10,285 samples with 7-line code context window
- **Contrastive Learning Data**: 53,218 word pairs (26,609 positive + 26,609 negative)
- **Word-Level Innovation**: Word pairs from the same code line as positive samples
- **High-Quality Negatives**: 9,474 common English words as negative samples
- **Smart Filtering**: Automatic filtering of single-letter variables and punctuation

## 🛠️ Technical Architecture

### Model Architecture

- **Base Model**: BERT-base-uncased
- **Transformer Layers**: 12
- **Attention Heads**: 12
- **Hidden Dimension**: 768
- **Feedforward Dimension**: 3072

### Training Strategy

- **Joint Loss**: Combines MLM and contrastive learning (70:30 ratio)
- **Optimizer**: AdamW with learning rate 2e-5
- **Batch Size**: 64 per GPU (effective batch size: 320 on 5 GPUs)
- **Training Epochs**: 30
- **MLM Masking Ratio**: 15%

### Loss Function

```
L_total = β_ft × L_MLM + (1 - β_ft) × L_InfoNCE
where β_ft = 0.7
```

## 📁 Project Structure

```
OAI_BERT_finetuning/
├── 🎯 Core Training Scripts
│   ├── train_30_epochs_final.py       # Main training script
│   ├── train_joint_bert.py            # Joint training implementation
│   └── train_with_monitoring.py       # Training with monitoring
├── 📊 Data Processing
│   ├── generate_paper_spec_data.py    # MLM and Contrastive Learning data generation
│   ├── joint_bert_model.py            # Joint training model definition
│   ├── tokenizer_utils.py             # Tokenizer utilities
│   └── create_enhanced_tokenizer.py   # Enhanced tokenizer creation
├── 🗃️ Data and Resources
│   ├── paper_spec_training_data/      # Training data
│   ├── enhanced_communication_tokenizer/ # Enhanced tokenizer
│   ├── knowledge_base/                # Code knowledge base
│   └── google-10000-english-no-swears.txt # English dictionary
├── 📋 Outputs and Reports
│   ├── final_30_epoch_output/         # Training outputs
│   └── final_training_report.md       # Training report
└── 📄 Documentation
    ├── README.md                      # English documentation
    ├── README_CN.md                   # Chinese documentation
    ├── LICENSE                        # MIT License
    └── requirements.txt               # Dependencies
```

## 🔬 Word-Level Contrastive Learning Innovation

### Key Innovation

Traditional contrastive learning operates at the sentence level, but this project introduces **word-level contrastive learning**:

- **Positive Pairs**: Words from the same code line (stronger semantic correlation)
- **Negative Pairs**: Code words vs. common English vocabulary (clear contrast)
- **Quality Control**: Filters out meaningless tokens and symbols
- **Local Deployment**: Uses local English dictionary, no external dependencies

### Benefits

1. **More Precise Representations**: Directly learns semantic relationships between code words
2. **Better Generalization**: High-quality negative samples improve discriminative ability
3. **Stronger Code Understanding**: Optimized contrastive learning strategy for programming domain

## 🖥️ Hardware Requirements

### Minimum Requirements

- **GPU**: CUDA-capable GPU with 8GB+ VRAM
- **RAM**: 16GB system memory
- **Storage**: 10GB available disk space

### Recommended Configuration

- **GPU**: Multi-GPU setup (5× NVIDIA A800 or similar)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ SSD storage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{bert_telecom_finetuning,
  title={BERT Fine-tuning for Telecommunication Domain with Word-Level Contrastive Learning},
  author={Haihang Jiang},
  year={2025},
  url={https://github.com/KelvinJiang2333/OAI_BERT_finetuning}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the BERT implementation
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for various tools and libraries

## 🔍 Detailed Documentation

### Data Generation Module

**MLM Data Generation:**
- **Context Window**: Extract 7-line context for each code line (3 preceding + 1 current + 3 subsequent)
- **Masking Strategy**: Randomly mask 15% tokens (80% with [MASK], 10% random replacement, 10% unchanged)
- **Sample Count**: 10,285 high-quality MLM training samples

**Contrastive Learning Data Generation:**
- **Word Extraction**: Use regex to extract valid words from code
- **Positive Pairs**: Word-to-word pairing within the same code line (26,609 pairs)
- **Negative Pairs**: Code words paired with common English vocabulary (26,609 pairs)
- **Quality Control**: Automatically filter meaningless tokens and symbols

### Model Training Module

- **Joint Loss**: Combines MLM and contrastive learning losses with 70:30 ratio
- **Distributed Training**: Multi-GPU parallel training with optimized resource utilization
- **Real-time Monitoring**: GPU performance, power consumption, and training metrics tracking

## 📧 Contact

For questions and feedback, please open an issue on GitHub.

---

<div align="center">
<b>A Code Semantic Understanding Model Training System Optimized for the Telecommunication Domain</b>
</div>
