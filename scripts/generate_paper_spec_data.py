#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文规格数据生成器
根据论文要求生成精确的训练数据:
- MLM: 10,285个样本，每个包含3前+1当前+3后=7行代码
- CL: 26,609对样本，N与N+1,N+2,N+3配对
"""

import json
import re
import random
import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from tokenizer_utils import get_enhanced_tokenizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperSpecDataGenerator:
    def __init__(self, tokenizer_path: str = None, seed: int = 42):
        """初始化论文规格数据生成器"""
        if tokenizer_path and tokenizer_path != "./enhanced_communication_tokenizer":
            logger.info(f"🔄 使用指定tokenizer路径: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.info(f"🚀 使用增强通信tokenizer")
            self.tokenizer = get_enhanced_tokenizer()
        
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_mlm_samples(self, file_paths: List[str], target_samples: int = 10285, 
                           context_lines: int = 3) -> List[Dict]:
        """
        生成MLM样本 (论文规格)
        每个样本包含3前+1当前+3后=7行代码，掩码15%的token
        """
        logger.info(f"📝 生成{target_samples}个MLM样本 (论文规格: {context_lines}前+1当前+{context_lines}后)")
        
        all_lines = []
        for file_path in file_paths:
            logger.info(f"处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_lines.extend([(line, file_path) for line in lines])
        
        logger.info(f"总有效行数: {len(all_lines)}")
        
        mlm_samples = []
        
        # 为每一行生成上下文样本
        for i in tqdm(range(len(all_lines)), desc="生成MLM样本"):
            if len(mlm_samples) >= target_samples:
                break
                
            current_line, file_path = all_lines[i]
            
            # 收集上下文行 (3前+1当前+3后)
            context_window = []
            
            # 添加前面的行
            for offset in range(-context_lines, 0):
                if i + offset >= 0:
                    prev_line, prev_file = all_lines[i + offset]
                    if prev_file == file_path:  # 同一文件
                        context_window.append(prev_line)
                    else:
                        context_window.append("")  # 占位符
                else:
                    context_window.append("")  # 占位符
            
            # 添加当前行
            context_window.append(current_line)
            
            # 添加后面的行
            for offset in range(1, context_lines + 1):
                if i + offset < len(all_lines):
                    next_line, next_file = all_lines[i + offset]
                    if next_file == file_path:  # 同一文件
                        context_window.append(next_line)
                    else:
                        context_window.append("")  # 占位符
                else:
                    context_window.append("")  # 占位符
            
            # 过滤空行并组合文本
            valid_lines = [line for line in context_window if line.strip()]
            if len(valid_lines) >= 3:  # 至少需要3行有效内容
                context_text = ' '.join(valid_lines)
                
                # 应用MLM掩码
                masked_sample = self.apply_mlm_masking_single(context_text, mask_ratio=0.15)
                if masked_sample:
                    mlm_samples.append(masked_sample)
        
        logger.info(f"✅ 生成{len(mlm_samples)}个MLM训练样本")
        return mlm_samples[:target_samples]
    
    def generate_contrastive_pairs(self, file_paths: List[str], target_pairs: int = 26609) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        生成词语级别对比学习样本对 (改进版)
        正样本: 同一行代码中的有效词语配对
        负样本: 代码词语与其他领域词语配对
        """
        logger.info(f"📝 生成{target_pairs}对词语级别对比学习样本")
        
        # 收集所有代码行
        all_lines = []
        for file_path in file_paths:
            logger.info(f"处理文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                all_lines.extend(lines)
        
        logger.info(f"总有效行数: {len(all_lines)}")
        
        # 生成正样本对: 同一行内的有效词语配对
        positive_pairs = []
        
        for line in tqdm(all_lines, desc="生成词语正样本对"):
            if len(positive_pairs) >= target_pairs:
                break
                
            # 提取有效词语
            valid_tokens = self._extract_valid_words(line)
            
            if len(valid_tokens) >= 2:
                # 在同一行内进行词语配对
                for i in range(len(valid_tokens)):
                    for j in range(i + 1, len(valid_tokens)):
                        if len(positive_pairs) >= target_pairs:
                            break
                        positive_pairs.append((valid_tokens[i], valid_tokens[j]))
            if len(positive_pairs) >= target_pairs:
                break
        
        # 生成负样本对: 代码词语与其他领域词语
        negative_pairs = []
        general_words = self._generate_general_domain_words(target_pairs)
        code_words = [pair[0] for pair in positive_pairs[:target_pairs]]
        
        for i, code_word in enumerate(code_words):
            if i < len(general_words):
                negative_pairs.append((code_word, general_words[i]))
        
        logger.info(f"✅ 生成{len(positive_pairs)}个词语正样本对，{len(negative_pairs)}个词语负样本对")
        return positive_pairs[:target_pairs], negative_pairs[:target_pairs]
    
    def _extract_valid_words(self, code_line: str) -> List[str]:
        """提取代码行中的完整有效词语，只过滤单字母变量"""
        import re
        
        # 1. 使用正则表达式提取所有可能的标识符和关键词
        # 匹配：字母开头，包含字母、数字、下划线的完整词语
        word_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        potential_words = re.findall(word_pattern, code_line)
        
        # 2. 只过滤单字母变量，保留所有多字母词语
        single_letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}
        
        valid_words = []
        for word in potential_words:
            word_lower = word.lower()
            
            # 简单过滤条件：只排除单字母
            if len(word) >= 2 or word_lower not in single_letters:
                valid_words.append(word_lower)
        
        # 3. 去重但保持顺序
        seen = set()
        unique_words = []
        for word in valid_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        
        return unique_words
    
    def _generate_general_domain_words(self, count: int) -> List[str]:
        """从常用英文词典生成负样本词汇"""
        try:
            # 使用常用词典文件
            common_words_path = os.path.join(os.path.dirname(__file__), 'google-10000-english-no-swears.txt')
            
            if os.path.exists(common_words_path):
                with open(common_words_path, 'r', encoding='utf-8') as f:
                    common_words = [line.strip().lower() for line in f if line.strip()]
                
                # 过滤词汇：长度>=3，只包含字母
                filtered_words = [
                    word for word in common_words 
                    if len(word) >= 3 and word.isalpha()
                ]
                
                logger.info(f"📖 使用常用词典: {len(filtered_words)} 个单词")
                
            else:
                # 备用词库
                filtered_words = [
                    "weather", "sunny", "rain", "cloud", "wind", "temperature", "season",
                    "food", "restaurant", "cooking", "meal", "breakfast", "lunch", "dinner",
                    "book", "library", "reading", "study", "learn", "education", "school",
                    "movie", "music", "entertainment", "game", "sport", "exercise", "health",
                    "travel", "vacation", "country", "city", "park", "building", "street",
                    "family", "friend", "people", "person", "child", "adult", "student",
                    "work", "job", "business", "office", "meeting", "project", "task",
                    "money", "price", "shopping", "market", "store", "product", "service",
                    "car", "transport", "traffic", "road", "bridge", "airport", "station",
                    "phone", "computer", "internet", "website", "email", "message", "news",
                    "animal", "dog", "cat", "bird", "fish", "tree", "flower", "garden",
                    "mountain", "river", "ocean", "forest", "field", "sky", "sun", "moon",
                    "color", "red", "blue", "green", "yellow", "black", "white", "purple",
                    "big", "small", "good", "bad", "new", "old", "fast", "slow", "easy",
                    "happy", "sad", "angry", "excited", "tired", "hungry", "thirsty",
                    "time", "day", "night", "morning", "afternoon", "evening", "week",
                    "month", "year", "history", "future", "past", "present", "moment",
                    "idea", "thought", "feeling", "emotion", "dream", "goal", "plan",
                    "problem", "solution", "question", "answer", "reason", "cause", "effect",
                    "important", "necessary", "possible", "difficult", "simple", "complex"
                ]
                logger.info(f"⚠️ 使用备用词库: {len(filtered_words)} 个单词")
        
        except Exception as e:
            logger.error(f"❌ 词典加载失败: {e}")
            return []
        
        # 随机选择指定数量的词汇
        if len(filtered_words) >= count:
            return random.sample(filtered_words, count)
        else:
            # 如果词汇不够，重复使用
            multiplier = (count // len(filtered_words)) + 1
            extended_words = filtered_words * multiplier
            return random.sample(extended_words, count)
    
    def apply_mlm_masking_single(self, text: str, mask_ratio: float = 0.15) -> Dict:
        """对单个文本应用MLM掩码"""
        tokens = self.tokenizer.tokenize(text)
        
        if len(tokens) < 3:  # 太短的跳过
            return None
        
        # 确定掩码数量
        num_masks = max(1, int(len(tokens) * mask_ratio))
        
        # 随机选择掩码位置
        mask_positions = random.sample(range(len(tokens)), min(num_masks, len(tokens)))
        
        original_tokens = tokens.copy()
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)
        
        for pos in mask_positions:
            labels[pos] = self.tokenizer.convert_tokens_to_ids([original_tokens[pos]])[0]
            
            rand_val = random.random()
            if rand_val < 0.8:
                masked_tokens[pos] = '[MASK]'
            elif rand_val < 0.9:
                # 替换为随机token
                vocab_tokens = list(self.tokenizer.get_vocab().keys())
                masked_tokens[pos] = random.choice(vocab_tokens)
            # 10%保持原样
        
        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
        
        return {
            'original_text': text,
            'masked_text': masked_text,
            'mask_positions': mask_positions,
            'labels': labels
        }
    
    def save_paper_spec_datasets(self, mlm_samples: List[Dict], 
                                positive_pairs: List[Tuple[str, str]],
                                negative_pairs: List[Tuple[str, str]],
                                output_dir: str):
        """保存论文规格数据集"""
        logger.info(f"💾 保存论文规格数据集到: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存MLM数据
        mlm_df = pd.DataFrame(mlm_samples)
        mlm_file = output_path / "paper_spec_mlm_data.csv"
        mlm_df.to_csv(mlm_file, index=False)
        logger.info(f"✅ MLM数据: {len(mlm_samples)}个样本 -> {mlm_file}")
        
        # 保存对比学习数据
        contrastive_data = []
        
        # 添加正样本
        for text1, text2 in positive_pairs:
            contrastive_data.append({
                'text1': text1,
                'text2': text2,
                'label': 1
            })
        
        # 添加负样本
        for text1, text2 in negative_pairs:
            contrastive_data.append({
                'text1': text1,
                'text2': text2,
                'label': 0
            })
        
        # 随机打乱
        random.shuffle(contrastive_data)
        
        contrastive_df = pd.DataFrame(contrastive_data)
        contrastive_file = output_path / "paper_spec_contrastive_data.csv"
        contrastive_df.to_csv(contrastive_file, index=False)
        logger.info(f"✅ 对比学习数据: {len(contrastive_data)}个样本 -> {contrastive_file}")
        
        # 保存配置信息
        config = {
            "paper_specification": {
                "mlm_samples": len(mlm_samples),
                "positive_pairs": len(positive_pairs),
                "negative_pairs": len(negative_pairs),
                "total_contrastive_pairs": len(positive_pairs) + len(negative_pairs)
            },
            "mlm_config": {
                "context_window": "3_preceding + 1_current + 3_subsequent",
                "mask_ratio": 0.15,
                "total_context_lines": 7
            },
            "contrastive_config": {
                "positive_strategy": "word_pairs_within_same_line",
                "negative_strategy": "code_words_with_general_domain_words",
                "token_filtering": "remove_punctuation_and_symbols",
                "min_token_length": 2
            },
            "joint_training": {
                "beta_ft": 0.7,
                "loss_function": "L_joint = β_ft * L_MLM + (1 - β_ft) * L_InfoNCE"
            }
        }
        
        config_file = output_path / "paper_spec_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 配置信息: {config_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="论文规格数据生成器")
    parser.add_argument("--knowledge_base_dir", type=str, default="./knowledge_base",
                       help="知识库目录")
    parser.add_argument("--tokenizer_path", type=str, default="./enhanced_communication_tokenizer",
                       help="Tokenizer路径")
    parser.add_argument("--output_dir", type=str, default="paper_spec_training_data",
                       help="输出目录")
    parser.add_argument("--mlm_samples", type=int, default=10285,
                       help="MLM样本数量")
    parser.add_argument("--contrastive_pairs", type=int, default=26609,
                       help="对比学习样本对数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    logger.info("🚀 启动论文规格数据生成器")
    logger.info(f"📋 配置参数:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # 创建数据生成器
    generator = PaperSpecDataGenerator(args.tokenizer_path, args.seed)
    
    # 获取知识库文件
    kb_dir = Path(args.knowledge_base_dir)
    c_files = list(kb_dir.glob("*.c")) + list(kb_dir.glob("*.h"))
    
    if not c_files:
        logger.error(f"在{kb_dir}中未找到C代码文件")
        return
    
    logger.info(f"📁 找到{len(c_files)}个代码文件")
    
    # 1. 生成MLM样本
    mlm_samples = generator.generate_mlm_samples(
        [str(f) for f in c_files],
        args.mlm_samples
    )
    
    # 2. 生成对比学习样本对
    positive_pairs, negative_pairs = generator.generate_contrastive_pairs(
        [str(f) for f in c_files],
        args.contrastive_pairs
    )
    
    # 3. 保存数据集
    generator.save_paper_spec_datasets(
        mlm_samples, positive_pairs, negative_pairs, args.output_dir
    )
    
    # 4. 打印统计信息
    logger.info("📊 论文规格数据集统计:")
    logger.info(f"  MLM样本: {len(mlm_samples)}")
    logger.info(f"  正样本对: {len(positive_pairs)}")
    logger.info(f"  负样本对: {len(negative_pairs)}")
    logger.info(f"  总对比样本: {len(positive_pairs) + len(negative_pairs)}")
    
    logger.info("🎉 论文规格数据生成完成!")

if __name__ == "__main__":
    main()
