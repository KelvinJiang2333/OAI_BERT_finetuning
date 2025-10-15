#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强tokenizer工具函数
提供统一的tokenizer加载和配置接口，支持自动创建增强分词器
"""

import os
import json
import shutil
import torch
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, BertTokenizer, BertModel
import logging

logger = logging.getLogger(__name__)

class EnhancedTokenizerLoader:
    """增强tokenizer加载器"""
    
    # 通信领域专业术语列表
    # ⚠️ 修改此列表后，请调用 sync_enhanced_tokenizer() 来同步分词器模型
    COMMUNICATION_TERMS = [
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
    
    @staticmethod
    def load_enhanced_tokenizer(
        enhanced_tokenizer_path: str = "./enhanced_communication_tokenizer",
        fallback_path: str = "bert-base-uncased",
        auto_create: bool = True
    ) -> AutoTokenizer:
        """
        加载增强的通信领域tokenizer，支持自动创建
        
        Args:
            enhanced_tokenizer_path: 增强tokenizer路径
            fallback_path: 回退tokenizer路径
            auto_create: 是否自动创建增强tokenizer
            
        Returns:
            AutoTokenizer: 加载的tokenizer
        """
        try:
            # 检查增强tokenizer是否存在
            if os.path.exists(enhanced_tokenizer_path):
                vocab_file = os.path.join(enhanced_tokenizer_path, "vocab.txt")
                config_file = os.path.join(enhanced_tokenizer_path, "tokenizer_config.json")
                
                if os.path.exists(vocab_file) and os.path.exists(config_file):
                    logger.info(f"📚 加载现有增强tokenizer: {enhanced_tokenizer_path}")
                    
                    # 手动创建tokenizer
                    tokenizer = BertTokenizer(
                        vocab_file=vocab_file,
                        do_lower_case=True,
                        strip_accents=None,
                        unk_token="[UNK]",
                        sep_token="[SEP]",
                        pad_token="[PAD]",
                        cls_token="[CLS]",
                        mask_token="[MASK]"
                    )
                    
                    # 读取配置以获取词汇表大小
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    logger.info(f"✅ 增强tokenizer加载成功")
                    logger.info(f"📊 词汇表大小: {config.get('vocab_size', len(tokenizer.vocab))}")
                    
                    return tokenizer
                else:
                    logger.warning(f"⚠️ 增强tokenizer文件不完整")
                    if auto_create:
                        logger.info("🔧 尝试重新创建增强tokenizer...")
                        return EnhancedTokenizerLoader._auto_create_enhanced_tokenizer(
                            enhanced_tokenizer_path, fallback_path
                        )
            else:
                logger.warning(f"⚠️ 增强tokenizer路径不存在: {enhanced_tokenizer_path}")
                if auto_create:
                    logger.info("🚀 自动创建增强tokenizer...")
                    return EnhancedTokenizerLoader._auto_create_enhanced_tokenizer(
                        enhanced_tokenizer_path, fallback_path
                    )
                
        except Exception as e:
            logger.error(f"❌ 加载增强tokenizer失败: {e}")
            if auto_create:
                logger.info("🔧 尝试自动创建...")
                return EnhancedTokenizerLoader._auto_create_enhanced_tokenizer(
                    enhanced_tokenizer_path, fallback_path
                )
        
        # 回退到标准tokenizer
        logger.info(f"📚 使用标准tokenizer: {fallback_path}")
        return AutoTokenizer.from_pretrained(fallback_path)
    
    @staticmethod
    def load_extended_embeddings(
        embedding_path: str = "./enhanced_communication_tokenizer/extended_embeddings_v2.pt",
        device: str = "cpu"
    ) -> Optional[torch.Tensor]:
        """
        加载扩展的嵌入矩阵
        
        Args:
            embedding_path: 嵌入文件路径
            device: 目标设备
            
        Returns:
            torch.Tensor: 扩展的嵌入矩阵，如果加载失败返回None
        """
        try:
            if os.path.exists(embedding_path):
                logger.info(f"🧠 加载扩展嵌入: {embedding_path}")
                embeddings = torch.load(embedding_path, map_location=device, weights_only=True)
                logger.info(f"✅ 嵌入矩阵加载成功: {embeddings.shape}")
                return embeddings
            else:
                logger.warning(f"⚠️ 嵌入文件不存在: {embedding_path}")
                
        except Exception as e:
            logger.error(f"❌ 加载嵌入矩阵失败: {e}")
        
        return None
    
    @staticmethod
    def get_decomposition_mapping(
        mapping_path: str = "./enhanced_communication_tokenizer/decomposition_mapping.json"
    ) -> Optional[Dict[str, Any]]:
        """
        获取术语分解映射
        
        Args:
            mapping_path: 映射文件路径
            
        Returns:
            Dict: 分解映射字典，如果加载失败返回None
        """
        try:
            if os.path.exists(mapping_path):
                logger.info(f"📝 加载分解映射: {mapping_path}")
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"✅ 分解映射加载成功: {len(mapping)} 个术语")
                return mapping
            else:
                logger.warning(f"⚠️ 映射文件不存在: {mapping_path}")
                
        except Exception as e:
            logger.error(f"❌ 加载分解映射失败: {e}")
        
        return None
    
    @staticmethod
    def get_tokenizer_info(tokenizer: AutoTokenizer) -> Dict[str, Any]:
        """
        获取tokenizer信息
        
        Args:
            tokenizer: tokenizer对象
            
        Returns:
            Dict: tokenizer信息
        """
        info = {
            "vocab_size": len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "special_tokens": {
                "pad_token": tokenizer.pad_token,
                "unk_token": tokenizer.unk_token,
                "cls_token": tokenizer.cls_token,
                "sep_token": tokenizer.sep_token,
                "mask_token": tokenizer.mask_token,
            }
        }
        return info

    @staticmethod
    def _auto_create_enhanced_tokenizer(
        enhanced_tokenizer_path: str,
        base_tokenizer_path: str
    ) -> AutoTokenizer:
        """
        自动创建增强tokenizer
        
        Args:
            enhanced_tokenizer_path: 增强tokenizer输出路径
            base_tokenizer_path: 基础tokenizer路径
            
        Returns:
            AutoTokenizer: 创建的增强tokenizer
        """
        try:
            logger.info("🛠️ 开始创建增强通信tokenizer...")
            
            # 检查基础tokenizer路径
            if not os.path.exists(base_tokenizer_path):
                # 尝试从预训练模型下载
                logger.info(f"📥 从HuggingFace下载基础tokenizer: {base_tokenizer_path}")
                base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
                # 这里我们直接使用下载的tokenizer，不保存到本地
            else:
                base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
            
            # 创建输出目录
            os.makedirs(enhanced_tokenizer_path, exist_ok=True)
            
            # 1. 复制基础tokenizer文件（如果本地存在）
            if os.path.exists(base_tokenizer_path):
                EnhancedTokenizerLoader._copy_base_tokenizer_files(
                    base_tokenizer_path, enhanced_tokenizer_path
                )
            else:
                # 保存下载的tokenizer文件
                base_tokenizer.save_pretrained(enhanced_tokenizer_path)
            
            # 2. 扩展词汇表
            original_vocab_size = EnhancedTokenizerLoader._extend_vocabulary(
                enhanced_tokenizer_path, base_tokenizer
            )
            
            # 3. 创建扩展嵌入矩阵
            EnhancedTokenizerLoader._create_extended_embeddings(
                enhanced_tokenizer_path, base_tokenizer_path, original_vocab_size
            )
            
            # 4. 创建术语分解映射
            EnhancedTokenizerLoader._create_decomposition_mapping(
                enhanced_tokenizer_path, base_tokenizer
            )
            
            # 5. 更新配置文件
            EnhancedTokenizerLoader._update_tokenizer_config(enhanced_tokenizer_path)
            
            # 6. 同步tokenizer.json文件
            EnhancedTokenizerLoader._sync_tokenizer_json(enhanced_tokenizer_path)
            
            # 7. 加载新创建的tokenizer
            tokenizer = BertTokenizer.from_pretrained(enhanced_tokenizer_path)
            
            logger.info("✅ 增强tokenizer创建成功！")
            logger.info(f"📊 最终词汇表大小: {len(tokenizer.vocab)}")
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"❌ 自动创建增强tokenizer失败: {e}")
            logger.info(f"🔄 回退到标准tokenizer: {base_tokenizer_path}")
            return AutoTokenizer.from_pretrained(base_tokenizer_path)
    
    @staticmethod
    def _copy_base_tokenizer_files(base_path: str, output_path: str):
        """复制基础tokenizer文件"""
        logger.info("📁 复制基础tokenizer文件...")
        
        base_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
        
        for file_name in base_files:
            src_path = os.path.join(base_path, file_name)
            dst_path = os.path.join(output_path, file_name)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"  ✓ 复制: {file_name}")
    
    @staticmethod
    def _extend_vocabulary(output_path: str, base_tokenizer: AutoTokenizer) -> int:
        """扩展词汇表"""
        logger.info("📚 扩展词汇表...")
        
        vocab_path = os.path.join(output_path, 'vocab.txt')
        
        # 读取原始词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            original_vocab = [line.strip() for line in f]
        
        original_size = len(original_vocab)
        logger.info(f"  原始词汇表大小: {original_size}")
        
        # 获取现有词汇集合
        existing_vocab = set(original_vocab)
        
        # 添加新术语（检查重复）
        new_terms = []
        for term in EnhancedTokenizerLoader.COMMUNICATION_TERMS:
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
    
    @staticmethod
    def _create_extended_embeddings(output_path: str, base_path: str, original_vocab_size: int):
        """创建扩展嵌入矩阵"""
        logger.info("🧠 创建扩展嵌入矩阵...")
        
        try:
            # 尝试加载原始模型以获取嵌入矩阵
            if os.path.exists(base_path):
                bert_model = BertModel.from_pretrained(base_path)
            else:
                bert_model = BertModel.from_pretrained("bert-base-uncased")
                
            original_embeddings = bert_model.embeddings.word_embeddings.weight.data
            
            embedding_dim = original_embeddings.shape[1]
            new_vocab_size = original_vocab_size + len(EnhancedTokenizerLoader.COMMUNICATION_TERMS)
            
            # 创建扩展嵌入矩阵
            extended_embeddings = torch.zeros(new_vocab_size, embedding_dim)
            
            # 复制原始嵌入
            extended_embeddings[:original_vocab_size] = original_embeddings[:original_vocab_size]
            
            # 为新术语初始化嵌入（使用正态分布）
            std = original_embeddings.std().item()
            extended_embeddings[original_vocab_size:] = torch.normal(
                mean=0.0, std=std, 
                size=(len(EnhancedTokenizerLoader.COMMUNICATION_TERMS), embedding_dim)
            )
            
            # 保存扩展嵌入
            embedding_path = os.path.join(output_path, 'extended_embeddings_v2.pt')
            torch.save(extended_embeddings, embedding_path)
            
            logger.info(f"  ✓ 嵌入矩阵: {extended_embeddings.shape}")
            
        except Exception as e:
            logger.warning(f"  ⚠️ 创建嵌入矩阵失败: {e}")
    
    @staticmethod
    def _create_decomposition_mapping(output_path: str, base_tokenizer: AutoTokenizer):
        """创建术语分解映射"""
        logger.info("📝 创建术语分解映射...")
        
        decomposition_mapping = {}
        
        for term in EnhancedTokenizerLoader.COMMUNICATION_TERMS:
            tokens = base_tokenizer.tokenize(term)
            
            decomposition_mapping[term] = {
                "original_tokens": tokens,
                "token_count": len(tokens),
                "improved_tokenization": [term],
                "improvement_ratio": len(tokens)
            }
        
        mapping_path = os.path.join(output_path, 'decomposition_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(decomposition_mapping, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✓ 术语映射: {len(decomposition_mapping)} 个")
    
    @staticmethod
    def _update_tokenizer_config(output_path: str):
        """更新tokenizer配置"""
        logger.info("⚙️ 更新tokenizer配置...")
        
        config_path = os.path.join(output_path, 'tokenizer_config.json')
        
        # 读取原始配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 更新配置
        config.update({
            "enhanced_tokenizer": True,
            "communication_domain": True,
            "added_tokens": len(EnhancedTokenizerLoader.COMMUNICATION_TERMS),
            "vocab_size": 30522 + len(EnhancedTokenizerLoader.COMMUNICATION_TERMS),
            "model_type": "enhanced_bert_communication",
            "domain": "wireless_communication",
            "version": "v2.0",
            "auto_created": True
        })
        
        # 保存更新后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def _sync_tokenizer_json(output_path: str):
        """同步更新 tokenizer.json 以匹配扩展的词汇表"""
        logger.info("🔄 同步tokenizer.json...")
        
        vocab_path = os.path.join(output_path, 'vocab.txt')
        tokenizer_json_path = os.path.join(output_path, 'tokenizer.json')
        
        # 读取当前词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f if line.strip()]
        
        # 加载现有的 tokenizer.json
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # 创建新的词汇映射
        from collections import OrderedDict
        new_vocab = OrderedDict()
        for idx, token in enumerate(vocab_list):
            new_vocab[token] = idx
        
        # 更新 tokenizer.json 中的词汇表
        if 'model' not in tokenizer_data:
            tokenizer_data['model'] = {}
        
        tokenizer_data['model']['vocab'] = new_vocab
        
        # 更新 added_tokens（如果存在）
        if 'added_tokens' in tokenizer_data:
            # 确保所有新增的通信术语都在 added_tokens 中
            existing_added = {token['content'] for token in tokenizer_data.get('added_tokens', [])}
            
            # 从词汇表末尾提取新增的术语
            original_vocab_size = 30522
            new_tokens = vocab_list[original_vocab_size:]
            
            for token in new_tokens:
                if token not in existing_added:
                    tokenizer_data['added_tokens'].append({
                        "id": vocab_list.index(token),
                        "content": token,
                        "single_word": True,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": False
                    })
        
        # 确保模型类型正确
        if 'model' in tokenizer_data:
            if 'type' not in tokenizer_data['model']:
                tokenizer_data['model']['type'] = 'WordPiece'
            
            # 添加其他必要的模型配置
            if 'unk_token' not in tokenizer_data['model']:
                tokenizer_data['model']['unk_token'] = '[UNK]'
            if 'sep_token' not in tokenizer_data['model']:
                tokenizer_data['model']['sep_token'] = '[SEP]'
            if 'cls_token' not in tokenizer_data['model']:
                tokenizer_data['model']['cls_token'] = '[CLS]'
        
        # 保存更新后的 tokenizer.json
        with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, separators=(',', ':'))
        
        logger.info(f"  ✓ tokenizer.json已同步")
        logger.info(f"  ✓ 词汇表大小: {len(new_vocab)}")
        
        return len(new_vocab)

def get_enhanced_tokenizer(enhanced_path: str = None, auto_create: bool = True) -> AutoTokenizer:
    """
    便捷函数：获取增强tokenizer，支持自动创建
    
    Args:
        enhanced_path: 增强tokenizer路径，默认使用标准路径
        auto_create: 是否自动创建增强tokenizer
        
    Returns:
        AutoTokenizer: 增强tokenizer
    """
    if enhanced_path is None:
        enhanced_path = "./enhanced_communication_tokenizer"
    
    return EnhancedTokenizerLoader.load_enhanced_tokenizer(
        enhanced_path, auto_create=auto_create
    )

def sync_enhanced_tokenizer(enhanced_path: str = None) -> bool:
    """
    手动同步增强分词器的tokenizer.json文件
    当修改了COMMUNICATION_TERMS知识库后，使用此函数确保分词器模型同步
    
    Args:
        enhanced_path: 增强tokenizer路径，默认使用标准路径
        
    Returns:
        bool: 同步是否成功
    """
    if enhanced_path is None:
        enhanced_path = "./enhanced_communication_tokenizer"
    
    try:
        logger.info("🔄 手动同步增强分词器...")
        
        # 检查增强分词器是否存在
        if not os.path.exists(enhanced_path):
            logger.error(f"❌ 增强分词器目录不存在: {enhanced_path}")
            return False
        
        # 检查必要文件是否存在
        required_files = ['vocab.txt', 'tokenizer.json']
        for file in required_files:
            file_path = os.path.join(enhanced_path, file)
            if not os.path.exists(file_path):
                logger.error(f"❌ 必要文件不存在: {file_path}")
                return False
        
        # 执行同步
        vocab_size = EnhancedTokenizerLoader._sync_tokenizer_json(enhanced_path)
        
        # 验证同步结果
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(enhanced_path)
            
            if tokenizer.vocab_size == vocab_size:
                logger.info("✅ 分词器同步成功！")
                logger.info(f"📊 词汇表大小: {tokenizer.vocab_size}")
                
                # 快速测试几个术语
                test_terms = ["aggregation_level", "timing_advance", "transport_block"]
                for term in test_terms:
                    tokens = tokenizer.tokenize(term)
                    if len(tokens) == 1 and tokens[0] == term:
                        logger.info(f"  ✓ {term} -> {tokens}")
                    else:
                        logger.warning(f"  ⚠️ {term} -> {tokens} (未优化)")
                
                return True
            else:
                logger.error(f"❌ 词汇表大小不匹配: 期望{vocab_size}, 实际{tokenizer.vocab_size}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 验证同步结果失败: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 同步失败: {e}")
        return False

def get_extended_embeddings(embedding_path: str = None, device: str = "cpu") -> Optional[torch.Tensor]:
    """
    便捷函数：获取扩展嵌入
    
    Args:
        embedding_path: 嵌入文件路径，默认使用标准路径
        device: 目标设备
        
    Returns:
        torch.Tensor: 扩展嵌入矩阵
    """
    if embedding_path is None:
        embedding_path = "./enhanced_communication_tokenizer/extended_embeddings_v2.pt"
    
    return EnhancedTokenizerLoader.load_extended_embeddings(embedding_path, device)
