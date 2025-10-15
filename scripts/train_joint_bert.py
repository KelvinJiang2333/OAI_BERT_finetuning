#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合MLM+对比学习BERT模型训练脚本

实现论文中的联合优化框架，用于无线通信领域的嵌入模型微调
"""

import os
import json
import argparse
import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import copy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    set_seed
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from tqdm import tqdm

from joint_bert_model import JointBertModel, JointBertTrainer
from tokenizer_utils import get_enhanced_tokenizer, get_extended_embeddings

# 设置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        """
        初始化早停机制
        
        Args:
            patience: 容忍没有改善的epoch数
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        """
        检查是否应该早停
        
        Args:
            val_loss: 当前验证损失
            model: 当前模型
            
        Returns:
            bool: 是否应该早停
        """
        # 检查是否有改善
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logger.info(f"🛑 早停触发！验证损失在 {self.patience} 个epoch内没有改善")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("🔄 已恢复最佳权重")
        
        return self.early_stop

class GradientNormTracker:
    """梯度范数追踪器"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.gradient_norms = []
        
    def update(self, model):
        """更新梯度范数"""
        total_norm = 0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
            
            # 保持窗口大小
            if len(self.gradient_norms) > self.window_size:
                self.gradient_norms.pop(0)
    
    def get_stats(self):
        """获取梯度统计信息"""
        if not self.gradient_norms:
            return {}
        
        return {
            'mean_grad_norm': np.mean(self.gradient_norms),
            'max_grad_norm': np.max(self.gradient_norms),
            'min_grad_norm': np.min(self.gradient_norms),
            'std_grad_norm': np.std(self.gradient_norms)
        }

class MLMDataset(Dataset):
    """MLM数据集"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        """
        初始化MLM数据集
        
        Args:
            data_path: MLM数据CSV文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        df = pd.read_csv(data_path)
        self.samples = []
        
        for _, row in df.iterrows():
            original_text = str(row['original_text'])
            masked_text = str(row['masked_text'])
            
            # 解析标签
            try:
                mask_labels = eval(row['mask_labels']) if isinstance(row['mask_labels'], str) else row['mask_labels']
            except:
                mask_labels = []
            
            self.samples.append({
                'original_text': original_text,
                'masked_text': masked_text,
                'mask_labels': mask_labels,
                'source_file': row.get('source_file', 'unknown')
            })
        
        logger.info(f"加载了 {len(self.samples)} 个MLM样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 编码掩码文本
        encoding = self.tokenizer(
            sample['masked_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 创建MLM标签
        input_ids = encoding['input_ids'].squeeze()
        labels = input_ids.clone()
        
        # 找到[MASK]位置并设置标签
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
        
        # 将非掩码位置的标签设为-100（忽略）
        labels[:] = -100
        
        # 为掩码位置设置真实标签（这里简化处理，实际应该根据原始文本确定）
        # 由于我们的数据生成比较复杂，这里使用一个简化的处理方式
        for pos in mask_positions:
            # 随机选择一个通信领域词汇作为标签（简化处理）
            # 在实际应用中，应该根据原始文本的对应位置确定真实标签
            if sample['mask_labels']:
                # 如果有标签信息，尝试使用
                try:
                    original_word = random.choice([label['original_word'] for label in sample['mask_labels']])
                    token_id = self.tokenizer.convert_tokens_to_ids(original_word)
                    if token_id != self.tokenizer.unk_token_id:
                        labels[pos] = token_id
                    else:
                        labels[pos] = input_ids[pos]  # 保持原样
                except:
                    labels[pos] = input_ids[pos]
            else:
                labels[pos] = input_ids[pos]
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

class ContrastiveDataset(Dataset):
    """对比学习数据集"""
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128):
        """
        初始化对比学习数据集
        
        Args:
            data_path: 对比学习数据CSV文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        df = pd.read_csv(data_path)
        self.samples = []
        
        for _, row in df.iterrows():
            self.samples.append({
                'text1': str(row['text1']),
                'text2': str(row['text2']),
                'label': int(row['label']),
                'context1': str(row.get('context1', row['text1'])),
                'context2': str(row.get('context2', row['text2'])),
                'pair_type': str(row.get('pair_type', 'unknown'))
            })
        
        logger.info(f"加载了 {len(self.samples)} 个对比学习样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 编码第一个文本（使用context以获得更丰富的信息）
        encoding1 = self.tokenizer(
            sample['context1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 编码第二个文本
        encoding2 = self.tokenizer(
            sample['context2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].squeeze(),
            'attention_mask_1': encoding1['attention_mask'].squeeze(),
            'input_ids_2': encoding2['input_ids'].squeeze(),
            'attention_mask_2': encoding2['attention_mask'].squeeze(),
            'labels': torch.tensor(sample['label'], dtype=torch.long)
        }

def collate_mlm_batch(batch):
    """MLM批次整理函数"""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def collate_contrastive_batch(batch):
    """对比学习批次整理函数"""
    return {
        'input_ids_1': torch.stack([item['input_ids_1'] for item in batch]),
        'attention_mask_1': torch.stack([item['attention_mask_1'] for item in batch]),
        'input_ids_2': torch.stack([item['input_ids_2'] for item in batch]),
        'attention_mask_2': torch.stack([item['attention_mask_2'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

class JointTrainingManager:
    """联合训练管理器"""
    
    def __init__(self, 
                 model: JointBertModel,
                 mlm_dataloader: DataLoader,
                 contrastive_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 device: str,
                 beta_ft: float = 0.5,
                 use_distributed: bool = False,
                 mlm_sampler: Optional[DistributedSampler] = None,
                 contrastive_sampler: Optional[DistributedSampler] = None,
                 val_mlm_dataloader: Optional[DataLoader] = None,
                 val_contrastive_dataloader: Optional[DataLoader] = None,
                 early_stopping: Optional[EarlyStopping] = None,
                 use_dynamic_lr: bool = False,
                 use_dynamic_beta: bool = False):
        """
        初始化训练管理器
        
        Args:
            model: 联合BERT模型
            mlm_dataloader: MLM数据加载器
            contrastive_dataloader: 对比学习数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            beta_ft: 联合损失权重参数
            use_distributed: 是否使用分布式训练
            mlm_sampler: MLM分布式采样器
            contrastive_sampler: 对比学习分布式采样器
            val_mlm_dataloader: 验证MLM数据加载器
            val_contrastive_dataloader: 验证对比学习数据加载器
            early_stopping: 早停机制
            use_dynamic_lr: 是否使用动态学习率
            use_dynamic_beta: 是否使用动态beta_ft
        """
        self.model = model
        self.mlm_dataloader = mlm_dataloader
        self.contrastive_dataloader = contrastive_dataloader
        self.val_mlm_dataloader = val_mlm_dataloader
        self.val_contrastive_dataloader = val_contrastive_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.beta_ft = beta_ft
        self.initial_beta_ft = beta_ft  # 保存初始值
        self.use_distributed = use_distributed
        self.mlm_sampler = mlm_sampler
        self.contrastive_sampler = contrastive_sampler
        self.early_stopping = early_stopping
        self.use_dynamic_lr = use_dynamic_lr
        self.use_dynamic_beta = use_dynamic_beta
        
        # 创建训练器
        self.trainer = JointBertTrainer(model, beta_ft, device)
        
        # 创建动态学习率调度器
        if self.use_dynamic_lr:
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3,
                verbose=True,
                min_lr=1e-7
            )
        
        # 梯度范数追踪器
        self.grad_tracker = GradientNormTracker()
        
        # 训练统计
        self.training_stats = {
            'epoch_losses': [],
            'mlm_losses': [],
            'contrastive_losses': [],
            'joint_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'beta_ft_history': [],
            'gradient_stats': []
        }
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_mlm_dataloader is None or self.val_contrastive_dataloader is None:
            return {}
        
        self.model.eval()
        val_mlm_loss = 0.0
        val_contrastive_loss = 0.0
        val_joint_loss = 0.0
        total_steps = 0
        
        with torch.no_grad():
            # 创建验证数据迭代器
            val_mlm_iter = iter(self.val_mlm_dataloader)
            val_contrastive_iter = iter(self.val_contrastive_dataloader)
            
            # 计算步数
            steps = min(len(self.val_mlm_dataloader), len(self.val_contrastive_dataloader))
            
            for step in range(steps):
                try:
                    # 获取验证批次
                    mlm_batch = next(val_mlm_iter)
                    mlm_batch = {k: v.to(self.device) for k, v in mlm_batch.items()}
                    
                    contrastive_batch = next(val_contrastive_iter)
                    contrastive_batch = {k: v.to(self.device) for k, v in contrastive_batch.items()}
                    
                    # 前向传播
                    results = self.trainer.forward_batch(mlm_batch, contrastive_batch)
                    
                    val_mlm_loss += results.get('mlm_loss', 0.0).item()
                    val_contrastive_loss += results.get('contrastive_loss', 0.0).item()
                    val_joint_loss += results['joint_loss'].item()
                    total_steps += 1
                    
                except StopIteration:
                    break
                except Exception as e:
                    logger.warning(f"验证步骤 {step} 出错: {e}")
                    continue
        
        if total_steps > 0:
            return {
                'val_mlm_loss': val_mlm_loss / total_steps,
                'val_contrastive_loss': val_contrastive_loss / total_steps,
                'val_joint_loss': val_joint_loss / total_steps
            }
        else:
            return {}
    
    def update_beta_ft(self, epoch: int, total_epochs: int):
        """动态更新beta_ft"""
        if not self.use_dynamic_beta:
            return
        
        # 使用余弦退火调整beta_ft
        # 训练初期偏重MLM，后期偏重对比学习
        progress = epoch / total_epochs
        self.beta_ft = self.initial_beta_ft * (1 + np.cos(np.pi * progress)) / 2
        
        # 更新训练器的beta_ft
        self.trainer.beta_ft = self.beta_ft
        
        logger.info(f"🔄 动态beta_ft更新: {self.beta_ft:.3f} (epoch {epoch}/{total_epochs})")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        try:
            self.model.train()
            
            # 如果使用分布式训练，设置epoch用于shuffling
            if self.use_distributed:
                if self.mlm_sampler is not None:
                    self.mlm_sampler.set_epoch(epoch)
                if self.contrastive_sampler is not None:
                    self.contrastive_sampler.set_epoch(epoch)
            
            # 创建数据迭代器
            logger.info(f"创建数据迭代器...")
            mlm_iter = iter(self.mlm_dataloader)
            contrastive_iter = iter(self.contrastive_dataloader)
            logger.info(f"数据迭代器创建成功")
            
            # 计算总步数（取两个数据集的最小值）
            total_steps = min(len(self.mlm_dataloader), len(self.contrastive_dataloader))
            
            epoch_mlm_loss = 0.0
            epoch_contrastive_loss = 0.0
            epoch_joint_loss = 0.0
            
            logger.info(f"开始训练循环，总步数: {total_steps}")
            progress_bar = tqdm(range(total_steps), desc=f"Epoch {epoch}")
            
            for step in progress_bar:
                try:
                    self.optimizer.zero_grad()
                    
                    # 获取MLM批次
                    try:
                        mlm_batch = next(mlm_iter)
                        mlm_batch = {k: v.to(self.device) for k, v in mlm_batch.items()}
                    except StopIteration:
                        mlm_iter = iter(self.mlm_dataloader)
                        mlm_batch = next(mlm_iter)
                        mlm_batch = {k: v.to(self.device) for k, v in mlm_batch.items()}
                    
                    # 获取对比学习批次
                    try:
                        contrastive_batch = next(contrastive_iter)
                        contrastive_batch = {k: v.to(self.device) for k, v in contrastive_batch.items()}
                    except StopIteration:
                        contrastive_iter = iter(self.contrastive_dataloader)
                        contrastive_batch = next(contrastive_iter)
                        contrastive_batch = {k: v.to(self.device) for k, v in contrastive_batch.items()}
                    
                    # 前向传播
                    results = self.trainer.forward_batch(mlm_batch, contrastive_batch)
                    
                    # 反向传播
                    joint_loss = results['joint_loss']
                    joint_loss.backward()
                    
                    # 追踪梯度范数
                    self.grad_tracker.update(self.model)
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 优化步骤
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # 累计损失
                    epoch_mlm_loss += results.get('mlm_loss', 0.0).item()
                    epoch_contrastive_loss += results.get('contrastive_loss', 0.0).item()
                    epoch_joint_loss += joint_loss.item()
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'MLM_Loss': f"{results.get('mlm_loss', 0.0).item():.4f}",
                        'Cont_Loss': f"{results.get('contrastive_loss', 0.0).item():.4f}",
                        'Joint_Loss': f"{joint_loss.item():.4f}",
                        'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                    })
                    
                except Exception as e:
                    logger.error(f"训练步骤 {step} 出错: {e}")
                    if self.use_distributed:
                        # 在分布式训练中，需要清理进程组
                        import torch.distributed as dist
                        if dist.is_initialized():
                            dist.destroy_process_group()
                    raise e
            
            # 计算平均损失
            avg_mlm_loss = epoch_mlm_loss / total_steps if total_steps > 0 else 0.0
            avg_contrastive_loss = epoch_contrastive_loss / total_steps if total_steps > 0 else 0.0
            avg_joint_loss = epoch_joint_loss / total_steps if total_steps > 0 else 0.0
            
            return {
                'mlm_loss': avg_mlm_loss,
                'contrastive_loss': avg_contrastive_loss,
                'joint_loss': avg_joint_loss
            }
            
        except Exception as e:
            logger.error(f"训练epoch {epoch} 失败: {e}")
            if self.use_distributed:
                # 在分布式训练中，需要清理进程组
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()
            raise e
    
    def train(self, num_epochs: int, save_dir: str):
        """完整训练过程"""
        logger.info(f"开始联合训练 - {num_epochs} epochs")
        logger.info(f"β_ft = {self.beta_ft} (MLM权重: {self.beta_ft:.2f}, 对比学习权重: {1-self.beta_ft:.2f})")
        
        if self.use_dynamic_beta:
            logger.info("🔄 启用动态beta_ft调整")
        if self.use_dynamic_lr:
            logger.info("📈 启用动态学习率调整")
        if self.early_stopping:
            logger.info(f"🛑 启用早停机制 (patience={self.early_stopping.patience})")
        
        os.makedirs(save_dir, exist_ok=True)
        
        best_joint_loss = float('inf')
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n=== Epoch {epoch}/{num_epochs} ===")
            
            # 动态更新beta_ft
            self.update_beta_ft(epoch, num_epochs)
            
            # 训练一个epoch
            epoch_results = self.train_epoch(epoch)
            
            # 验证
            val_results = self.validate()
            
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 获取梯度统计
            grad_stats = self.grad_tracker.get_stats()
            
            # 记录统计信息
            combined_results = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'beta_ft': self.beta_ft,
                **epoch_results,
                **val_results,
                **grad_stats
            }
            
            self.training_stats['epoch_losses'].append(combined_results)
            self.training_stats['mlm_losses'].append(epoch_results['mlm_loss'])
            self.training_stats['contrastive_losses'].append(epoch_results['contrastive_loss'])
            self.training_stats['joint_losses'].append(epoch_results['joint_loss'])
            self.training_stats['learning_rates'].append(current_lr)
            self.training_stats['beta_ft_history'].append(self.beta_ft)
            
            if val_results:
                self.training_stats['val_losses'].append(val_results['val_joint_loss'])
            if grad_stats:
                self.training_stats['gradient_stats'].append(grad_stats)
            
            # 打印结果
            logger.info(f"Epoch {epoch} Results:")
            logger.info(f"  训练 - MLM: {epoch_results['mlm_loss']:.4f}, "
                       f"对比学习: {epoch_results['contrastive_loss']:.4f}, "
                       f"联合: {epoch_results['joint_loss']:.4f}")
            
            if val_results:
                logger.info(f"  验证 - MLM: {val_results['val_mlm_loss']:.4f}, "
                           f"对比学习: {val_results['val_contrastive_loss']:.4f}, "
                           f"联合: {val_results['val_joint_loss']:.4f}")
            
            logger.info(f"  学习率: {current_lr:.6f}, β_ft: {self.beta_ft:.3f}")
            
            if grad_stats:
                logger.info(f"  梯度范数: 平均={grad_stats['mean_grad_norm']:.4f}, "
                           f"最大={grad_stats['max_grad_norm']:.4f}")
            
            # 动态学习率调整
            if self.use_dynamic_lr and val_results:
                self.lr_scheduler.step(val_results['val_joint_loss'])
            
            # 保存最佳模型（基于验证损失，如果有的话）
            current_loss = val_results.get('val_joint_loss', epoch_results['joint_loss'])
            
            if current_loss < best_joint_loss:
                best_joint_loss = current_loss
                self.save_model(save_dir, epoch, is_best=True)
                logger.info(f"  🎉 新的最佳模型! (损失: {best_joint_loss:.4f})")
            
            # 早停检查
            if self.early_stopping and val_results:
                val_loss = val_results['val_joint_loss']
                if self.early_stopping(val_loss, self.model):
                    logger.info(f"🛑 训练提前停止在epoch {epoch}")
                    break
            
            # 定期保存检查点
            if epoch % 5 == 0:
                self.save_model(save_dir, epoch, is_best=False)
        
        # 保存最终模型和统计信息
        self.save_model(save_dir, epoch, is_best=False, is_final=True)
        self.save_training_stats(save_dir)
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        logger.info(f"\n🎉 训练完成！")
        logger.info(f"训练时间: {training_time/3600:.2f} 小时")
        logger.info(f"最佳损失: {best_joint_loss:.4f}")
        logger.info(f"最终学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"模型保存在: {save_dir}")
    
    def save_model(self, save_dir: str, epoch: int, is_best: bool = False, is_final: bool = False):
        """保存模型"""
        if is_best:
            model_path = os.path.join(save_dir, 'best_model')
        elif is_final:
            model_path = os.path.join(save_dir, 'final_model')
        else:
            model_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}')
        
        os.makedirs(model_path, exist_ok=True)
        
        # 保存模型权重 - 同时保存pytorch_model.bin和safetensors格式
        torch.save(self.model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
        
        # 尝试保存safetensors格式
        try:
            from safetensors.torch import save_file
            save_file(self.model.state_dict(), os.path.join(model_path, 'model.safetensors'))
            logger.info(f"✅ 已保存safetensors格式: {model_path}/model.safetensors")
        except ImportError:
            logger.warning("⚠️ safetensors库未安装，跳过safetensors格式保存")
        except Exception as e:
            logger.warning(f"⚠️ safetensors保存失败: {e}")
        
        # 创建完整的模型配置（基于BERT配置 + 我们的扩展）
        # 处理DistributedDataParallel包装的情况
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        model_config = model_to_use.bert_config.to_dict()
        
        # 添加我们的自定义配置
        model_config.update({
            "architectures": ["JointBertModel"],
            "model_type": "joint_bert",
            "custom_config": {
                "contrastive_dim": model_to_use.contrastive_dim,
                "temperature": model_to_use.temperature,
                "beta_ft": self.beta_ft,
                "use_enhanced_tokenizer": hasattr(model_to_use, 'enhanced_tokenizer_used'),
                "vocab_size": model_to_use.bert_config.vocab_size
            }
        })
        
        # 保存完整的模型配置
        with open(os.path.join(model_path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False)
        
        # 保存训练特定配置（向后兼容）
        training_config = {
            'epoch': epoch,
            'beta_ft': self.beta_ft,
            'contrastive_dim': model_to_use.contrastive_dim,
            'temperature': model_to_use.temperature,
            'model_class': 'JointBertModel'
        }
        
        with open(os.path.join(model_path, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        # 如果使用了增强tokenizer，复制所有tokenizer文件
        if hasattr(model_to_use, 'enhanced_tokenizer_used') and model_to_use.enhanced_tokenizer_used:
            # 保存tokenizer配置
            tokenizer_config = {
                "model_max_length": 512,
                "tokenizer_class": "BertTokenizer",
                "enhanced_tokenizer": True,
                "enhanced_tokenizer_path": "./enhanced_communication_tokenizer"
            }
            with open(os.path.join(model_path, 'tokenizer_config.json'), 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            
            # 复制tokenizer核心文件
            import shutil
            enhanced_tokenizer_path = "./enhanced_communication_tokenizer"
            tokenizer_files = [
                'tokenizer.json',
                'vocab.txt', 
                'decomposition_mapping.json'
            ]
            
            for file_name in tokenizer_files:
                src_file = os.path.join(enhanced_tokenizer_path, file_name)
                dst_file = os.path.join(model_path, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    logger.info(f"✅ 已复制tokenizer文件: {file_name}")
                else:
                    logger.warning(f"⚠️ tokenizer文件不存在: {src_file}")
            
            # 保存词汇表信息
            vocab_info = {
                "original_vocab_size": 30522,
                "extended_vocab_size": 30680,
                "added_tokens": 158,
                "source": "enhanced_communication_tokenizer"
            }
            with open(os.path.join(model_path, 'vocab_info.json'), 'w', encoding='utf-8') as f:
                json.dump(vocab_info, f, indent=2, ensure_ascii=False)
    
    def save_training_stats(self, save_dir: str):
        """保存训练统计信息"""
        stats_path = os.path.join(save_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # 保存CSV格式的损失数据
        df = pd.DataFrame(self.training_stats['epoch_losses'])
        df.to_csv(os.path.join(save_dir, 'training_losses.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description="联合MLM+对比学习BERT模型训练")
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, 
                       default="./paper_spec_training_data",
                       help="训练数据目录")
    parser.add_argument("--mlm_data_file", type=str, 
                       default="paper_spec_mlm_data.csv",
                       help="MLM训练数据文件名")
    parser.add_argument("--contrastive_data_file", type=str,
                       default="paper_spec_contrastive_data.csv",
                       help="对比学习训练数据文件名")
    
    # 模型参数
    parser.add_argument("--bert_model_path", type=str,
                       default="./bert-base-uncased",
                       help="BERT模型路径")
    parser.add_argument("--use_enhanced_tokenizer", action="store_true",
                       help="使用增强通信tokenizer")
    parser.add_argument("--no_enhanced_tokenizer", action="store_true",
                       help="禁用增强tokenizer，强制使用标准BERT tokenizer（默认使用增强tokenizer）")
    parser.add_argument("--enhanced_tokenizer_path", type=str,
                       default="./enhanced_communication_tokenizer",
                       help="增强tokenizer路径")
    parser.add_argument("--contrastive_dim", type=int, default=128,
                       help="对比学习嵌入维度")
    parser.add_argument("--temperature", type=float, default=0.07,
                       help="对比学习温度参数")
    parser.add_argument("--beta_ft", type=float, default=0.7,
                       help="联合损失权重参数 β_ft (论文规格: 0.7)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64,
                       help="批次大小（每个GPU的批次大小）")
    parser.add_argument("--num_epochs", type=int, default=30,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="预热步数比例")
    parser.add_argument("--max_length", type=int, default=128,
                       help="最大序列长度")
    
    # 优化参数
    parser.add_argument("--use_validation", action="store_true",
                       help="使用验证集评估")
    parser.add_argument("--validation_split", type=float, default=0.2,
                       help="验证集比例")
    parser.add_argument("--early_stopping", action="store_true",
                       help="启用早停机制")
    parser.add_argument("--patience", type=int, default=7,
                       help="早停耐心值")
    parser.add_argument("--min_delta", type=float, default=0.001,
                       help="早停最小改善阈值")
    parser.add_argument("--use_dynamic_lr", action="store_true",
                       help="使用动态学习率调整")
    parser.add_argument("--use_dynamic_beta", action="store_true",
                       help="使用动态beta_ft调整")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str,
                       default="./final_30_epoch_output",
                       help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    # GPU和分布式训练参数
    parser.add_argument("--use_multi_gpu", action="store_true",
                       help="使用多GPU训练")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="指定GPU ID，格式: 0,1,2 (None表示使用所有可用GPU)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="数据加载器工作线程数")
    parser.add_argument("--no_pin_memory", action="store_true",
                       help="禁用内存固定（默认启用内存固定以加速数据传输）")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 处理pin_memory参数 - 默认启用，除非指定--no_pin_memory
    pin_memory = not args.no_pin_memory if hasattr(args, 'no_pin_memory') else True
    
    # 初始化分布式训练
    if args.use_multi_gpu and torch.cuda.is_available():
        # 检查是否在分布式环境中
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            # 分布式训练
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # 初始化进程组
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            
            logger.info(f"🚀 分布式训练 - Rank: {rank}/{world_size}, Local Rank: {local_rank}")
        else:
            # 多GPU但非分布式训练（DataParallel）
            device = torch.device("cuda")
            logger.info(f"🚀 多GPU训练 (DataParallel) - {torch.cuda.device_count()} GPUs")
    else:
        # 单GPU或CPU训练
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 单设备训练: {device}")
    
    # 只在主进程打印配置信息
    if not args.use_multi_gpu or not ('RANK' in os.environ) or int(os.environ.get("RANK", 0)) == 0:
        logger.info("🚀 开始联合BERT模型训练...")
        logger.info(f"📋 训练配置:")
        logger.info(f"  - 数据目录: {args.data_dir}")
        logger.info(f"  - BERT模型: {args.bert_model_path}")
        logger.info(f"  - 批次大小: {args.batch_size}")
        logger.info(f"  - 训练轮数: {args.num_epochs}")
        logger.info(f"  - 学习率: {args.learning_rate}")
        logger.info(f"  - β_ft: {args.beta_ft}")
        logger.info(f"  - 对比学习维度: {args.contrastive_dim}")
        logger.info(f"  - 温度参数: {args.temperature}")
        logger.info(f"  - 多GPU训练: {args.use_multi_gpu}")
        logger.info(f"  - 数据加载工作线程: {args.dataloader_num_workers}")
        logger.info(f"  - 内存固定: {pin_memory}")
    
    # 加载分词器 - 默认使用增强tokenizer，除非明确指定--no_enhanced_tokenizer
    if hasattr(args, 'no_enhanced_tokenizer') and args.no_enhanced_tokenizer:
        logger.info(f"📚 用户指定使用标准tokenizer: {args.bert_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_path)
        use_enhanced_tokenizer = False
    else:
        logger.info("🚀 使用增强通信tokenizer（支持自动创建）")
        tokenizer = get_enhanced_tokenizer(auto_create=True)
        use_enhanced_tokenizer = True
    
    # 创建数据集
    mlm_data_path = os.path.join(args.data_dir, args.mlm_data_file)
    contrastive_data_path = os.path.join(args.data_dir, args.contrastive_data_file)
    
    logger.info("📊 加载数据集...")
    full_mlm_dataset = MLMDataset(mlm_data_path, tokenizer, args.max_length)
    full_contrastive_dataset = ContrastiveDataset(contrastive_data_path, tokenizer, args.max_length)
    
    # 分割数据集为训练集和验证集
    val_mlm_dataloader = None
    val_contrastive_dataloader = None
    
    if args.use_validation:
        # 计算分割大小
        mlm_val_size = int(len(full_mlm_dataset) * args.validation_split)
        mlm_train_size = len(full_mlm_dataset) - mlm_val_size
        
        contrastive_val_size = int(len(full_contrastive_dataset) * args.validation_split)
        contrastive_train_size = len(full_contrastive_dataset) - contrastive_val_size
        
        # 分割MLM数据集
        mlm_dataset, val_mlm_dataset = random_split(
            full_mlm_dataset, 
            [mlm_train_size, mlm_val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        # 分割对比学习数据集
        contrastive_dataset, val_contrastive_dataset = random_split(
            full_contrastive_dataset, 
            [contrastive_train_size, contrastive_val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        logger.info(f"📊 数据集分割:")
        logger.info(f"  MLM - 训练: {len(mlm_dataset)}, 验证: {len(val_mlm_dataset)}")
        logger.info(f"  对比学习 - 训练: {len(contrastive_dataset)}, 验证: {len(val_contrastive_dataset)}")
        
        # 创建验证数据加载器
        val_mlm_dataloader = DataLoader(
            val_mlm_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_mlm_batch
        )
        
        val_contrastive_dataloader = DataLoader(
            val_contrastive_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_contrastive_batch
        )
    else:
        mlm_dataset = full_mlm_dataset
        contrastive_dataset = full_contrastive_dataset
        logger.info("📊 使用全量数据集训练（无验证集）")
    
    # 创建数据加载器
    # 检查是否使用分布式采样
    use_distributed_sampler = args.use_multi_gpu and 'RANK' in os.environ
    
    if use_distributed_sampler:
        mlm_sampler = DistributedSampler(mlm_dataset)
        contrastive_sampler = DistributedSampler(contrastive_dataset)
        shuffle_mlm = False
        shuffle_contrastive = False
    else:
        mlm_sampler = None
        contrastive_sampler = None
        shuffle_mlm = True
        shuffle_contrastive = True
    
    mlm_dataloader = DataLoader(
        mlm_dataset, 
        batch_size=args.batch_size, 
        shuffle=shuffle_mlm,
        sampler=mlm_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_mlm_batch
    )
    
    contrastive_dataloader = DataLoader(
        contrastive_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_contrastive,
        sampler=contrastive_sampler,
        num_workers=args.dataloader_num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_contrastive_batch
    )
    
    logger.info(f"  - MLM样本: {len(mlm_dataset)}")
    logger.info(f"  - 对比学习样本: {len(contrastive_dataset)}")
    logger.info(f"  - MLM批次数: {len(mlm_dataloader)}")
    logger.info(f"  - 对比学习批次数: {len(contrastive_dataloader)}")
    
    # 创建模型
    logger.info("🤖 创建联合BERT模型...")
    model = JointBertModel(
        bert_model_path=args.bert_model_path,
        contrastive_dim=args.contrastive_dim,
        temperature=args.temperature,
        use_enhanced_tokenizer=use_enhanced_tokenizer
    )
    # 标记是否使用了增强tokenizer
    model.enhanced_tokenizer_used = use_enhanced_tokenizer
    model.to(device)
    
    # 设置多GPU训练
    if args.use_multi_gpu and torch.cuda.is_available():
        if 'RANK' in os.environ:
            # 分布式训练
            model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None,
                       find_unused_parameters=True)
            logger.info("🚀 模型已包装为DistributedDataParallel")
        elif torch.cuda.device_count() > 1:
            # DataParallel训练
            model = nn.DataParallel(model)
            logger.info(f"🚀 模型已包装为DataParallel，使用 {torch.cuda.device_count()} 个GPU")
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # 创建学习率调度器
    total_steps = min(len(mlm_dataloader), len(contrastive_dataloader)) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"📈 优化器配置:")
    logger.info(f"  - 总训练步数: {total_steps}")
    logger.info(f"  - 预热步数: {warmup_steps}")
    
    # 创建早停机制
    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=args.min_delta,
            restore_best_weights=True
        )
        logger.info(f"🛑 早停配置: patience={args.patience}, min_delta={args.min_delta}")
    
    # 创建训练管理器
    training_manager = JointTrainingManager(
        model=model,
        mlm_dataloader=mlm_dataloader,
        contrastive_dataloader=contrastive_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        beta_ft=args.beta_ft,
        use_distributed=use_distributed_sampler,
        mlm_sampler=mlm_sampler,
        contrastive_sampler=contrastive_sampler,
        val_mlm_dataloader=val_mlm_dataloader,
        val_contrastive_dataloader=val_contrastive_dataloader,
        early_stopping=early_stopping,
        use_dynamic_lr=args.use_dynamic_lr,
        use_dynamic_beta=args.use_dynamic_beta
    )
    
    # 开始训练
    training_manager.train(args.num_epochs, args.output_dir)
    
    logger.info("🎉 训练完成！")


if __name__ == "__main__":
    main()
