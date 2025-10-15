#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带性能和能源监控的嵌入模型训练脚本
记录详细的训练时间、GPU使用率、内存消耗等指标
"""

import subprocess
import time
import json
import threading
import psutil
import logging
from pathlib import Path
from datetime import datetime
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.system_stats = []
        self.gpu_stats = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = datetime.now()
        self.monitoring = True
        self.system_stats = []
        self.gpu_stats = []
        
        # 启动系统监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("🔍 开始性能监控...")
        
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        self.end_time = datetime.now()
        logger.info("⏹️ 停止性能监控")
        
    def _monitor_system(self):
        """监控系统资源"""
        while self.monitoring:
            try:
                # CPU和内存使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU信息 (如果可用)
                gpu_info = self._get_gpu_info()
                
                timestamp = datetime.now()
                
                system_stat = {
                    'timestamp': timestamp.isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3)
                }
                
                self.system_stats.append(system_stat)
                
                if gpu_info:
                    gpu_stat = {
                        'timestamp': timestamp.isoformat(),
                        'gpu_info': gpu_info
                    }
                    self.gpu_stats.append(gpu_stat)
                    
            except Exception as e:
                logger.warning(f"监控过程中出现错误: {e}")
                
            time.sleep(5)  # 每5秒采集一次
            
    def _get_gpu_info(self):
        """获取GPU信息"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_data = []
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpu_data.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_used_mb': int(parts[2]),
                                'memory_total_mb': int(parts[3]),
                                'utilization_percent': int(parts[4]),
                                'power_draw_w': float(parts[5]) if parts[5] != '[N/A]' else 0,
                                'temperature_c': int(parts[6]) if parts[6] != '[N/A]' else 0
                            })
                return gpu_data
        except Exception as e:
            logger.debug(f"无法获取GPU信息: {e}")
        return None
        
    def get_training_duration(self):
        """获取训练时长"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
        
    def generate_report(self, output_dir: str):
        """生成详细报告"""
        report_dir = Path(output_dir)
        report_dir.mkdir(exist_ok=True)
        
        duration = self.get_training_duration()
        
        # 计算统计指标
        cpu_avg = sum(s['cpu_percent'] for s in self.system_stats) / len(self.system_stats) if self.system_stats else 0
        memory_avg = sum(s['memory_percent'] for s in self.system_stats) / len(self.system_stats) if self.system_stats else 0
        memory_peak = max(s['memory_used_gb'] for s in self.system_stats) if self.system_stats else 0
        
        # GPU统计
        gpu_summary = {}
        if self.gpu_stats:
            for gpu_stat in self.gpu_stats:
                for i, gpu in enumerate(gpu_stat['gpu_info']):
                    if i not in gpu_summary:
                        gpu_summary[i] = {
                            'name': gpu['name'],
                            'utilizations': [],
                            'memory_used': [],
                            'power_draws': [],
                            'temperatures': []
                        }
                    gpu_summary[i]['utilizations'].append(gpu['utilization_percent'])
                    gpu_summary[i]['memory_used'].append(gpu['memory_used_mb'])
                    gpu_summary[i]['power_draws'].append(gpu['power_draw_w'])
                    gpu_summary[i]['temperatures'].append(gpu['temperature_c'])
        
        # 计算GPU平均值
        for gpu_id in gpu_summary:
            gpu = gpu_summary[gpu_id]
            gpu['avg_utilization'] = sum(gpu['utilizations']) / len(gpu['utilizations'])
            gpu['avg_memory_used_gb'] = sum(gpu['memory_used']) / len(gpu['memory_used']) / 1024
            gpu['peak_memory_used_gb'] = max(gpu['memory_used']) / 1024
            gpu['avg_power_draw'] = sum(gpu['power_draws']) / len(gpu['power_draws'])
            gpu['peak_temperature'] = max(gpu['temperatures'])
            
            # 估算能源消耗 (kWh)
            gpu['estimated_energy_kwh'] = (gpu['avg_power_draw'] * duration / 3600) / 1000
        
        # 生成综合报告
        report = {
            'training_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': duration,
                'duration_minutes': duration / 60,
                'duration_hours': duration / 3600
            },
            'system_performance': {
                'cpu_avg_percent': round(cpu_avg, 2),
                'memory_avg_percent': round(memory_avg, 2),
                'memory_peak_gb': round(memory_peak, 2),
                'total_samples': len(self.system_stats)
            },
            'gpu_performance': gpu_summary,
            'energy_estimation': {
                'total_gpu_energy_kwh': sum(gpu['estimated_energy_kwh'] for gpu in gpu_summary.values()),
                'avg_power_consumption_w': sum(gpu['avg_power_draw'] for gpu in gpu_summary.values())
            }
        }
        
        # 保存详细报告
        report_file = report_dir / "training_performance_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 保存原始数据
        raw_data_file = report_dir / "raw_monitoring_data.json"
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump({
                'system_stats': self.system_stats,
                'gpu_stats': self.gpu_stats
            }, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📊 性能报告已保存: {report_file}")
        return report

def main():
    """带监控的模型训练主函数"""
    
    logger.info("🚀 启动带监控的嵌入模型训练")
    logger.info("=" * 70)
    
    # 创建监控器
    monitor = TrainingMonitor()
    
    # 多GPU训练命令 (使用torchrun分布式训练)
    training_cmd = [
        "torchrun", 
        "--nproc_per_node=5",      # 使用5个GPU
        "--nnodes=1",              # 单节点
        "--node_rank=0",           # 节点排名
        "--master_addr=localhost", # 主节点地址
        "--master_port=12355",     # 主节点端口
        "train_joint_bert.py",
        "--data_dir", "paper_spec_training_data",
        "--mlm_data_file", "paper_spec_mlm_data.csv", 
        "--contrastive_data_file", "paper_spec_contrastive_data.csv",
        "--output_dir", "multi_gpu_training_output",
        "--bert_model_path", "./bert-base-uncased",
        
        # 训练参数
        "--num_epochs", "5",                     # 5个epoch
        "--batch_size", "32",                    # 降低batch size以适应监控
        "--learning_rate", "2e-05",              
        "--beta_ft", "0.7",                      
        
        # 模型设置
        "--contrastive_dim", "768",              
        "--temperature", "0.07",                 
        
        # 多GPU设置
        "--use_multi_gpu",
        "--dataloader_num_workers", "4",
        "--pin_memory", "True",
        
        # 增强tokenizer
        "--use_enhanced_tokenizer",
        "--enhanced_tokenizer_path", "./enhanced_communication_tokenizer"
    ]
    
    logger.info("📋 多GPU训练配置:")
    logger.info(f"  - GPU数量: 5个 (A800-SXM4-80GB)")
    logger.info(f"  - 训练轮数: 5 epochs")
    logger.info(f"  - 批次大小: 32 (每GPU)")
    logger.info(f"  - 有效批次大小: 160 (32×5)")
    logger.info(f"  - 学习率: 2e-05")
    logger.info(f"  - β_ft: 0.7")
    logger.info(f"  - MLM样本: 10,285个")
    logger.info(f"  - 对比学习: 53,218对")
    
    try:
        # 开始监控
        monitor.start_monitoring()
        
        logger.info("🎯 开始训练...")
        
        # 执行训练
        result = subprocess.run(training_cmd, check=True)
        
        logger.info("✅ 训练完成!")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 训练失败: {e}")
        raise
    except KeyboardInterrupt:
        logger.info("🛑 训练被用户中断")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        
        # 生成报告
        logger.info("📊 正在生成性能报告...")
        report = monitor.generate_report("multi_gpu_training_output")
        
        # 打印摘要
        print_training_summary(report)

def print_training_summary(report):
    """打印训练摘要"""
    print("\n" + "="*70)
    print("📊 嵌入模型训练性能报告")
    print("="*70)
    
    # 训练时间
    duration = report['training_summary']['duration_seconds']
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    print(f"⏱️  训练时长: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📅 开始时间: {report['training_summary']['start_time']}")
    print(f"📅 结束时间: {report['training_summary']['end_time']}")
    
    # 系统性能
    sys_perf = report['system_performance']
    print(f"\n💻 系统性能:")
    print(f"  - CPU平均使用率: {sys_perf['cpu_avg_percent']:.1f}%")
    print(f"  - 内存平均使用率: {sys_perf['memory_avg_percent']:.1f}%")
    print(f"  - 内存峰值使用: {sys_perf['memory_peak_gb']:.1f} GB")
    
    # GPU性能
    if report['gpu_performance']:
        print(f"\n🎮 GPU性能:")
        for gpu_id, gpu_info in report['gpu_performance'].items():
            print(f"  GPU {gpu_id} ({gpu_info['name']}):")
            print(f"    - 平均利用率: {gpu_info['avg_utilization']:.1f}%")
            print(f"    - 平均显存使用: {gpu_info['avg_memory_used_gb']:.1f} GB")
            print(f"    - 峰值显存使用: {gpu_info['peak_memory_used_gb']:.1f} GB")
            print(f"    - 平均功耗: {gpu_info['avg_power_draw']:.1f} W")
            print(f"    - 峰值温度: {gpu_info['peak_temperature']}°C")
    
    # 能源消耗
    energy = report['energy_estimation']
    print(f"\n⚡ 能源消耗估算:")
    print(f"  - 总能耗: {energy['total_gpu_energy_kwh']:.4f} kWh")
    print(f"  - 平均功耗: {energy['avg_power_consumption_w']:.1f} W")
    print(f"  - 能源成本估算: ¥{energy['total_gpu_energy_kwh'] * 0.6:.2f} (按0.6元/kWh)")
    
    print("\n✅ 详细报告已保存在 monitored_training_output/ 目录")
    print("="*70)

if __name__ == "__main__":
    main()
