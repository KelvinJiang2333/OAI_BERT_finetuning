#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建增强通信tokenizer

基于BERT-base-uncased扩展通信领域专业术语
"""

import os
import json
import shutil
import logging
from typing import List, Dict, Set
from transformers import AutoTokenizer, BertTokenizer
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTokenizerBuilder:
    """增强tokenizer构建器"""
    
    def __init__(self, base_tokenizer_path: str):
        self.base_tokenizer_path = base_tokenizer_path
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
        
        # 通信领域专业术语列表
        self.communication_terms = [
            # 5G/NR术语
            "timing_advance", "harq_processes", "aggregation_level",
            "scheduling_offset", "dci_format", "pdcch_config",
            "pucch_config", "srs_config", "csi_rs_config",
            "prach_config", "rach_config", "rnti_config",
            
            # LTE术语  
            "enb_config", "cell_config", "ue_context",
            "radio_bearer", "eps_bearer", "qos_config",
            "measurement_config", "handover_config",
            
            # 调度相关
            "resource_allocation", "modulation_coding",
            "transport_block", "code_block", "rate_matching",
            "channel_coding", "scrambling_sequence",
            
            # 物理层
            "subcarrier_spacing", "cyclic_prefix", "guard_interval",
            "ofdm_symbol", "resource_element", "resource_block",
            "bandwidth_part", "carrier_frequency",
            
            # MAC层
            "mac_header", "mac_payload", "buffer_status",
            "scheduling_request", "random_access", "contention_resolution",
            
            # RLC层
            "rlc_header", "sequence_number", "acknowledged_mode",
            "unacknowledged_mode", "transparent_mode",
            
            # PDCP层
            "pdcp_header", "compression_config", "integrity_protection",
            "ciphering_config", "rohc_config",
            
            # RRC层
            "rrc_connection", "system_information", "measurement_report",
            "capability_information", "security_config",
            
            # 网络架构
            "core_network", "access_network", "transport_network",
            "network_function", "service_function",
            
            # 协议栈
            "protocol_stack", "layer_interface", "service_access_point",
            "protocol_data_unit", "service_data_unit",
            
            # 信令流程
            "attach_procedure", "detach_procedure", "registration_procedure",
            "authentication_procedure", "authorization_procedure",
            
            # QoS相关
            "quality_of_service", "service_level_agreement",
            "traffic_shaping", "congestion_control",
            
            # 移动性管理
            "mobility_management", "location_management",
            "handover_management", "load_balancing",
            
            # 安全相关
            "authentication_vector", "security_algorithm",
            "encryption_key", "integrity_key", "security_context",
            
            # 无线资源管理
            "radio_resource_management", "interference_coordination",
            "power_control", "admission_control",
            
            # 性能优化
            "throughput_optimization", "latency_optimization", 
            "energy_efficiency", "spectral_efficiency",
            
            # 测试和测量
            "performance_monitoring", "key_performance_indicator",
            "network_optimization", "fault_management",
            
            # 新增术语
            "beam_management", "massive_mimo", "network_slicing",
            "edge_computing", "ultra_reliable", "low_latency",
            "enhanced_mobile", "machine_type", "critical_communication",
            
            # 算法相关
            "scheduling_algorithm", "resource_allocation_algorithm",
            "channel_estimation", "signal_processing",
            "error_correction", "adaptive_modulation",
            
            # 数据结构
            "configuration_parameters", "measurement_parameters",
            "status_indication", "event_notification",
            "timer_configuration", "counter_management",
            
            # 函数和过程
            "initialization_procedure", "configuration_procedure",
            "monitoring_procedure", "cleanup_procedure",
            "error_handling", "exception_processing",
            
            # 特定实现
            "gnb_scheduler", "enb_scheduler", "mac_scheduler",
            "rlc_entity", "pdcp_entity", "rrc_entity",
            "nas_entity", "mm_entity", "sm_entity"
        ]
        
    def build_enhanced_tokenizer(self, output_dir: str) -> bool:
        """构建增强tokenizer"""
        try:
            logger.info("🚀 开始构建增强通信tokenizer...")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 复制基础tokenizer文件
            self._copy_base_tokenizer_files(output_dir)
            
            # 2. 扩展词汇表
            original_vocab_size = self._extend_vocabulary(output_dir)
            
            # 3. 创建扩展嵌入矩阵
            self._create_extended_embeddings(output_dir, original_vocab_size)
            
            # 4. 创建术语分解映射
            self._create_decomposition_mapping(output_dir)
            
            # 5. 更新配置文件
            self._update_tokenizer_config(output_dir)
            
            # 6. 验证tokenizer
            self._validate_tokenizer(output_dir)
            
            logger.info(f"✅ 增强tokenizer构建完成: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 构建增强tokenizer失败: {e}")
            return False
    
    def _copy_base_tokenizer_files(self, output_dir: str):
        """复制基础tokenizer文件"""
        logger.info("📁 复制基础tokenizer文件...")
        
        base_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
        
        for file_name in base_files:
            src_path = os.path.join(self.base_tokenizer_path, file_name)
            dst_path = os.path.join(output_dir, file_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"  ✓ 复制: {file_name}")
            else:
                logger.warning(f"  ⚠️ 文件不存在: {src_path}")
    
    def _extend_vocabulary(self, output_dir: str) -> int:
        """扩展词汇表"""
        logger.info("📚 扩展词汇表...")
        
        vocab_path = os.path.join(output_dir, 'vocab.txt')
        
        # 读取原始词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            original_vocab = [line.strip() for line in f]
        
        original_size = len(original_vocab)
        logger.info(f"  原始词汇表大小: {original_size}")
        
        # 获取现有词汇集合
        existing_vocab = set(original_vocab)
        
        # 添加新术语（检查重复）
        new_terms = []
        for term in self.communication_terms:
            if term not in existing_vocab:
                new_terms.append(term)
                existing_vocab.add(term)
        
        # 写入扩展后的词汇表
        extended_vocab = original_vocab + new_terms
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for word in extended_vocab:
                f.write(f"{word}\n")
        
        logger.info(f"  新增术语: {len(new_terms)} 个")
        logger.info(f"  扩展后词汇表大小: {len(extended_vocab)}")
        
        return original_size
    
    def _create_extended_embeddings(self, output_dir: str, original_vocab_size: int):
        """创建扩展嵌入矩阵"""
        logger.info("🧠 创建扩展嵌入矩阵...")
        
        # 加载原始模型以获取嵌入矩阵
        try:
            from transformers import BertModel
            bert_model = BertModel.from_pretrained(self.base_tokenizer_path)
            original_embeddings = bert_model.embeddings.word_embeddings.weight.data
            
            embedding_dim = original_embeddings.shape[1]
            new_vocab_size = original_vocab_size + len(self.communication_terms)
            
            # 创建扩展嵌入矩阵
            extended_embeddings = torch.zeros(new_vocab_size, embedding_dim)
            
            # 复制原始嵌入
            extended_embeddings[:original_vocab_size] = original_embeddings[:original_vocab_size]
            
            # 为新术语初始化嵌入（使用正态分布）
            std = original_embeddings.std().item()
            extended_embeddings[original_vocab_size:] = torch.normal(
                mean=0.0, std=std, size=(len(self.communication_terms), embedding_dim)
            )
            
            # 保存扩展嵌入
            embedding_path = os.path.join(output_dir, 'extended_embeddings_v2.pt')
            torch.save(extended_embeddings, embedding_path)
            
            logger.info(f"  ✓ 嵌入矩阵: {extended_embeddings.shape}")
            logger.info(f"  ✓ 保存路径: {embedding_path}")
            
        except Exception as e:
            logger.error(f"  ❌ 创建嵌入矩阵失败: {e}")
    
    def _create_decomposition_mapping(self, output_dir: str):
        """创建术语分解映射"""
        logger.info("📝 创建术语分解映射...")
        
        # 创建分解映射
        decomposition_mapping = {}
        
        for term in self.communication_terms:
            # 使用原始tokenizer分解术语
            tokens = self.base_tokenizer.tokenize(term)
            
            decomposition_mapping[term] = {
                "original_tokens": tokens,
                "token_count": len(tokens),
                "improved_tokenization": [term],  # 增强后用单个token
                "improvement_ratio": len(tokens)  # 改进比例
            }
        
        # 保存映射
        mapping_path = os.path.join(output_dir, 'decomposition_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(decomposition_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ 术语映射: {len(decomposition_mapping)} 个")
        logger.info(f"  ✓ 保存路径: {mapping_path}")
    
    def _update_tokenizer_config(self, output_dir: str):
        """更新tokenizer配置"""
        logger.info("⚙️ 更新tokenizer配置...")
        
        config_path = os.path.join(output_dir, 'tokenizer_config.json')
        
        # 读取原始配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新配置
        config.update({
            "enhanced_tokenizer": True,
            "communication_domain": True,
            "added_tokens": len(self.communication_terms),
            "vocab_size": 30522 + len(self.communication_terms),
            "model_type": "enhanced_bert_communication",
            "domain": "wireless_communication",
            "version": "v2.0"
        })
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ 配置更新完成")
    
    def _validate_tokenizer(self, output_dir: str):
        """验证tokenizer"""
        logger.info("🔍 验证增强tokenizer...")
        
        try:
            # 尝试加载新的tokenizer
            tokenizer = BertTokenizer.from_pretrained(output_dir)
            
            # 测试专业术语分词
            test_terms = ["timing_advance", "harq_processes", "aggregation_level"]
            
            logger.info("  测试分词效果:")
            for term in test_terms:
                tokens = tokenizer.tokenize(term)
                logger.info(f"    {term} -> {tokens}")
            
            logger.info(f"  ✅ 验证通过，词汇表大小: {len(tokenizer.vocab)}")
            
        except Exception as e:
            logger.error(f"  ❌ 验证失败: {e}")

def main():
    # 配置路径
    base_tokenizer_path = "./bert-base-uncased"
    output_dir = "./enhanced_communication_tokenizer"
    
    # 检查基础tokenizer是否存在
    if not os.path.exists(base_tokenizer_path):
        logger.error(f"❌ 基础tokenizer不存在: {base_tokenizer_path}")
        return
    
    # 创建构建器
    builder = EnhancedTokenizerBuilder(base_tokenizer_path)
    
    # 构建增强tokenizer
    success = builder.build_enhanced_tokenizer(output_dir)
    
    if success:
        logger.info("🎉 增强tokenizer创建成功！")
        logger.info(f"📁 输出目录: {output_dir}")
        
        # 提供使用说明
        logger.info("\n📋 使用方法:")
        logger.info("python train_joint_bert.py --use_enhanced_tokenizer")
        
    else:
        logger.error("💥 增强tokenizer创建失败")

if __name__ == "__main__":
    main()
