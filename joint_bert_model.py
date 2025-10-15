#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联合MLM+对比学习的BERT模型架构

实现论文中的联合优化框架：
- MLM (Masked Language Modeling) 任务
- 对比学习 (Contrastive Learning) 任务
- 联合损失函数：L_joint = β_ft * L_MLM + (1 - β_ft) * L_InfoNCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from typing import Dict, Optional, Tuple
import logging
from tokenizer_utils import get_enhanced_tokenizer, get_extended_embeddings

logger = logging.getLogger(__name__)

class JointBertModel(nn.Module):
    """
    联合MLM+对比学习的BERT模型
    
    架构：
    1. 共享的BERT编码器
    2. MLM头：用于掩码语言建模
    3. 对比学习投影头：用于生成对比学习嵌入
    """
    
    def __init__(self, 
                 bert_model_path: str,
                 contrastive_dim: int = 128,
                 dropout_rate: float = 0.1,
                 temperature: float = 0.07,
                 use_enhanced_tokenizer: bool = False):
        """
        初始化联合模型
        
        Args:
            bert_model_path: BERT模型路径
            contrastive_dim: 对比学习嵌入维度
            dropout_rate: Dropout率
            temperature: 对比学习温度参数
            use_enhanced_tokenizer: 是否使用增强tokenizer
        """
        super().__init__()
        
        # 加载BERT配置和模型
        self.bert_config = BertConfig.from_pretrained(bert_model_path)
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        # 获取BERT隐藏层维度
        self.hidden_size = self.bert_config.hidden_size
        self.vocab_size = self.bert_config.vocab_size
        self.contrastive_dim = contrastive_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        
        # 如果使用增强tokenizer，扩展词汇表
        if use_enhanced_tokenizer:
            self._setup_enhanced_tokenizer()
        
        # 初始化模型头部
        self._initialize_heads()
        
    def _setup_enhanced_tokenizer(self):
        """设置增强tokenizer和扩展嵌入"""
        try:
            logger.info("🔧 设置增强tokenizer...")
            extended_embeddings = get_extended_embeddings()
            
            if extended_embeddings is not None:
                current_vocab_size = self.bert.config.vocab_size
                new_vocab_size = extended_embeddings.shape[0]
                
                if new_vocab_size > current_vocab_size:
                    logger.info(f"📈 扩展词汇表: {current_vocab_size} → {new_vocab_size}")
                    
                    # 创建新的嵌入层
                    new_embeddings = torch.nn.Embedding(
                        new_vocab_size, 
                        extended_embeddings.shape[1]
                    )
                    new_embeddings.weight.data = extended_embeddings
                    
                    # 替换原有嵌入层
                    self.bert.embeddings.word_embeddings = new_embeddings
                    
                    # 更新配置
                    self.bert.config.vocab_size = new_vocab_size
                    self.vocab_size = new_vocab_size  # 更新实例变量
                    
                    logger.info("✅ 增强tokenizer设置完成")
                else:
                    logger.warning("⚠️ 扩展嵌入大小不匹配")
            else:
                logger.warning("⚠️ 无法加载扩展嵌入")
                
        except Exception as e:
            logger.error(f"❌ 设置增强tokenizer失败: {e}")
    
    def _initialize_heads(self):
        """初始化模型头部"""
        # MLM头：用于预测掩码token (在tokenizer设置后创建)
        self.mlm_head = nn.Linear(self.hidden_size, self.vocab_size)
        
        # 对比学习投影头：将BERT嵌入映射到对比学习空间
        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.contrastive_dim)
        )
        
        # 参数已在初始化时设置
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"初始化JointBertModel - 隐藏维度: {self.hidden_size}, 对比学习维度: {self.contrastive_dim}")
    
    def _init_weights(self):
        """初始化新增层的权重"""
        # MLM头权重初始化
        nn.init.normal_(self.mlm_head.weight, mean=0.0, std=self.bert_config.initializer_range)
        nn.init.zeros_(self.mlm_head.bias)
        
        # 对比学习投影头权重初始化
        for module in self.contrastive_projection:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.bert_config.initializer_range)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                mlm_labels: Optional[torch.Tensor] = None,
                contrastive_labels: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: token类型IDs [batch_size, seq_len]
            mlm_labels: MLM标签 [batch_size, seq_len]，-100表示非掩码位置
            contrastive_labels: 对比学习标签 [batch_size]，1表示正样本对，0表示负样本对
            return_embeddings: 是否返回嵌入向量
            
        Returns:
            Dict包含：
            - mlm_logits: MLM预测logits [batch_size, seq_len, vocab_size]
            - contrastive_embeddings: 对比学习嵌入 [batch_size, contrastive_dim]
            - mlm_loss: MLM损失 (如果提供mlm_labels)
            - contrastive_loss: 对比学习损失 (如果提供contrastive_labels)
            - joint_loss: 联合损失
        """
        outputs = {}
        
        # 通过共享BERT编码器
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取序列输出和池化输出
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output  # [batch_size, hidden_size]
        
        # MLM任务
        mlm_logits = self.mlm_head(sequence_output)  # [batch_size, seq_len, vocab_size]
        outputs['mlm_logits'] = mlm_logits
        
        # 对比学习任务
        # 使用[CLS]token的表示进行对比学习
        cls_embeddings = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        contrastive_embeddings = self.contrastive_projection(cls_embeddings)  # [batch_size, contrastive_dim]
        # L2归一化
        contrastive_embeddings = F.normalize(contrastive_embeddings, p=2, dim=1)
        outputs['contrastive_embeddings'] = contrastive_embeddings
        
        # 如果需要返回原始嵌入
        if return_embeddings:
            outputs['cls_embeddings'] = cls_embeddings
            outputs['sequence_embeddings'] = sequence_output
        
        # 计算损失
        total_loss = 0.0
        
        # MLM损失
        if mlm_labels is not None:
            mlm_loss = self._compute_mlm_loss(mlm_logits, mlm_labels)
            outputs['mlm_loss'] = mlm_loss
            total_loss = total_loss + mlm_loss  # 避免inplace操作
        
        # 对比学习损失
        if contrastive_labels is not None:
            contrastive_loss = self._compute_contrastive_loss(contrastive_embeddings, contrastive_labels)
            outputs['contrastive_loss'] = contrastive_loss
            total_loss = total_loss + contrastive_loss  # 避免inplace操作
        
        if mlm_labels is not None or contrastive_labels is not None:
            outputs['joint_loss'] = total_loss
        
        return outputs
    
    def _compute_mlm_loss(self, mlm_logits: torch.Tensor, mlm_labels: torch.Tensor) -> torch.Tensor:
        """
        计算MLM损失
        
        Args:
            mlm_logits: MLM预测logits [batch_size, seq_len, vocab_size]
            mlm_labels: MLM标签 [batch_size, seq_len]，-100表示非掩码位置
            
        Returns:
            MLM损失
        """
        # 重塑张量
        mlm_logits = mlm_logits.view(-1, self.vocab_size)
        mlm_labels = mlm_labels.view(-1)
        
        # 计算交叉熵损失，忽略-100标签
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        mlm_loss = loss_fct(mlm_logits, mlm_labels)
        
        return mlm_loss
    
    def _compute_contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算改进的InfoNCE对比学习损失
        
        Args:
            embeddings: 归一化的嵌入向量 [batch_size, contrastive_dim]
            labels: 对比学习标签 [batch_size]，1表示正样本，0表示负样本
            
        Returns:
            对比学习损失
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        # 1. 确保embeddings已归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 2. 动态调整温度参数（如果损失过高）
        # 这是一个简单的自适应策略
        effective_temperature = max(self.temperature, 0.1)  # 避免温度过低
        
        # 3. 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / effective_temperature
        
        # 4. 创建掩码，排除自身对角线
        eye_mask = torch.eye(batch_size, device=device).bool()
        
        # 5. 改进的正样本对识别 - 支持不同的标签策略
        if labels.dtype == torch.bool or labels.max() <= 1:
            # 二分类标签：1表示正样本类，0表示负样本类
            positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (labels.unsqueeze(1) == 1)
        else:
            # 多分类标签：相同类别为正样本对
            positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        positive_mask = positive_mask & (~eye_mask)  # 排除自身
        
        # 6. 计算每个样本的损失
        losses = []
        
        # 检查是否有足够的正样本对
        total_positive_pairs = positive_mask.sum().item()
        if total_positive_pairs == 0:
            # 没有正样本对时，使用修改的策略
            # 鼓励同类样本相似，不同类样本不相似
            same_class_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            same_class_mask = same_class_mask & (~eye_mask)
            
            if same_class_mask.sum() > 0:
                # 有同类样本
                pos_similarities = similarity_matrix[same_class_mask]
                neg_similarities = similarity_matrix[~same_class_mask & ~eye_mask]
                
                # 简化的对比损失：最大化同类相似度，最小化不同类相似度
                if len(pos_similarities) > 0 and len(neg_similarities) > 0:
                    pos_loss = -torch.log(torch.sigmoid(pos_similarities)).mean()
                    neg_loss = -torch.log(torch.sigmoid(-neg_similarities)).mean()
                    contrastive_loss = (pos_loss + neg_loss) / 2
                else:
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # 标准InfoNCE损失计算
            for i in range(batch_size):
                positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]
                
                if len(positive_indices) > 0:
                    # 获取当前样本的相似度分数
                    sim_i = similarity_matrix[i]  # [batch_size]
                    
                    # 为了数值稳定性，减去最大值
                    sim_i = sim_i - sim_i.max().detach()
                    
                    # 对每个正样本计算损失
                    for pos_idx in positive_indices:
                        pos_sim = sim_i[pos_idx]
                        
                        # 所有样本（除自己）作为分母
                        all_sims = torch.cat([sim_i[:i], sim_i[i+1:]])  # 排除自身
                        
                        # InfoNCE损失
                        log_prob = pos_sim - torch.logsumexp(all_sims, dim=0)
                        losses.append(-log_prob)
            
            if losses:
                contrastive_loss = torch.stack(losses).mean()
            else:
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 7. 添加损失统计信息（可选，用于调试）
        if hasattr(self, 'training') and self.training:
            # 记录一些统计信息
            with torch.no_grad():
                self._contrastive_stats = {
                    'avg_similarity': similarity_matrix[~eye_mask].mean().item(),
                    'positive_pairs': total_positive_pairs,
                    'effective_temperature': effective_temperature,
                    'loss_value': contrastive_loss.item()
                }
        
        return contrastive_loss
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        获取文本的嵌入表示（用于推理）
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            归一化的嵌入向量 [batch_size, contrastive_dim]
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=True
            )
            return outputs['contrastive_embeddings']


class JointBertTrainer:
    """
    联合BERT模型训练器
    实现论文中的联合损失函数：L_joint = β_ft * L_MLM + (1 - β_ft) * L_InfoNCE
    """
    
    def __init__(self, 
                 model: JointBertModel,
                 beta_ft: float = 0.5,
                 device: str = 'cuda'):
        """
        初始化训练器
        
        Args:
            model: 联合BERT模型
            beta_ft: MLM和对比学习的权重平衡参数 β_ft ∈ [0, 1]
            device: 设备
        """
        self.model = model
        self.beta_ft = beta_ft
        self.device = device
        # 获取模型的temperature参数
        model_to_use = model.module if hasattr(model, 'module') else model
        self.temperature = model_to_use.temperature
        
        logger.info(f"初始化JointBertTrainer - β_ft: {beta_ft}, temperature: {self.temperature}")
    
    def _compute_pairwise_contrastive_loss(self, 
                                          emb1: torch.Tensor, 
                                          emb2: torch.Tensor, 
                                          labels: torch.Tensor) -> torch.Tensor:
        """
        计算pair-wise对比学习损失
        
        对于每一对(emb1[i], emb2[i]):
        - 如果labels[i]=1（正样本对）：使它们相似
        - 如果labels[i]=0（负样本对）：使它们不相似
        
        Args:
            emb1: 第一组嵌入 [batch_size, dim]
            emb2: 第二组嵌入 [batch_size, dim]
            labels: 标签 [batch_size]，1=正样本对，0=负样本对
            
        Returns:
            对比学习损失
        """
        # 计算每对之间的余弦相似度
        # emb1和emb2已经在调用前归一化，所以点积=余弦相似度
        similarities = torch.sum(emb1 * emb2, dim=1)  # [batch_size]
        
        # 缩放相似度（应用温度参数）
        similarities = similarities / self.temperature
        
        # 计算损失
        # 正样本对：最大化相似度 -> -log(sigmoid(sim))
        # 负样本对：最小化相似度 -> -log(sigmoid(-sim)) = -log(1-sigmoid(sim))
        
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        loss = 0.0
        n_pos = positive_mask.sum().item()
        n_neg = negative_mask.sum().item()
        
        if n_pos > 0:
            # 正样本对损失：希望相似度高
            pos_similarities = similarities[positive_mask]
            pos_loss = -torch.log(torch.sigmoid(pos_similarities) + 1e-10).mean()
            loss = loss + pos_loss
        
        if n_neg > 0:
            # 负样本对损失：希望相似度低
            neg_similarities = similarities[negative_mask]
            neg_loss = -torch.log(torch.sigmoid(-neg_similarities) + 1e-10).mean()
            loss = loss + neg_loss
        
        # 如果有正负样本，取平均
        if n_pos > 0 and n_neg > 0:
            loss = loss / 2
        
        return loss
    
    def compute_joint_loss(self, 
                          mlm_loss: torch.Tensor,
                          contrastive_loss: torch.Tensor) -> torch.Tensor:
        """
        计算联合损失：L_joint = β_ft * L_MLM + (1 - β_ft) * L_InfoNCE
        
        Args:
            mlm_loss: MLM损失
            contrastive_loss: 对比学习损失
            
        Returns:
            联合损失
        """
        joint_loss = self.beta_ft * mlm_loss + (1 - self.beta_ft) * contrastive_loss
        return joint_loss
    
    def forward_batch(self, 
                     mlm_batch: Dict[str, torch.Tensor],
                     contrastive_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        处理一个批次的联合训练
        
        Args:
            mlm_batch: MLM批次数据
            contrastive_batch: 对比学习批次数据
            
        Returns:
            包含损失和指标的字典
        """
        results = {}
        
        # MLM前向传播
        if mlm_batch:
            mlm_outputs = self.model(
                input_ids=mlm_batch['input_ids'],
                attention_mask=mlm_batch['attention_mask'],
                mlm_labels=mlm_batch['labels']
            )
            results['mlm_loss'] = mlm_outputs['mlm_loss']
            results['mlm_logits'] = mlm_outputs['mlm_logits']
        
        # 对比学习前向传播
        if contrastive_batch:
            # 处理成对的输入
            batch_size = contrastive_batch['input_ids_1'].shape[0]
            
            # 处理DistributedDataParallel包装的情况
            model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
            
            # 第一个文本
            outputs_1 = model_to_use(
                input_ids=contrastive_batch['input_ids_1'],
                attention_mask=contrastive_batch['attention_mask_1']
            )
            
            # 第二个文本
            outputs_2 = model_to_use(
                input_ids=contrastive_batch['input_ids_2'],
                attention_mask=contrastive_batch['attention_mask_2']
            )
            
            # 获取嵌入
            emb1 = outputs_1['contrastive_embeddings']  # [batch_size, dim]
            emb2 = outputs_2['contrastive_embeddings']  # [batch_size, dim]
            
            # 归一化
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            
            # 使用pair-wise对比学习损失
            labels = contrastive_batch['labels']  # [batch_size], 1=正样本对, 0=负样本对
            contrastive_loss = self._compute_pairwise_contrastive_loss(emb1, emb2, labels)
            results['contrastive_loss'] = contrastive_loss
        
        # 计算联合损失
        if 'mlm_loss' in results and 'contrastive_loss' in results:
            joint_loss = self.compute_joint_loss(results['mlm_loss'], results['contrastive_loss'])
            results['joint_loss'] = joint_loss
        elif 'mlm_loss' in results:
            results['joint_loss'] = results['mlm_loss']
        elif 'contrastive_loss' in results:
            results['joint_loss'] = results['contrastive_loss']
        
        return results


def create_joint_bert_model(bert_model_path: str, 
                           contrastive_dim: int = 128,
                           temperature: float = 0.07) -> JointBertModel:
    """
    创建联合BERT模型的便捷函数
    
    Args:
        bert_model_path: BERT模型路径
        contrastive_dim: 对比学习嵌入维度
        temperature: 温度参数
        
    Returns:
        JointBertModel实例
    """
    model = JointBertModel(
        bert_model_path=bert_model_path,
        contrastive_dim=contrastive_dim,
        temperature=temperature
    )
    
    return model


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模型
    model_path = "./bert-base-uncased"
    model = create_joint_bert_model(model_path)
    
    # 创建训练器
    trainer = JointBertTrainer(model, beta_ft=0.5)
    
    print("✅ 联合BERT模型创建成功！")
    print(f"📊 模型参数:")
    print(f"  - BERT隐藏维度: {model.hidden_size}")
    print(f"  - 对比学习维度: {model.contrastive_dim}")
    print(f"  - 温度参数: {model.temperature}")
    print(f"  - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
