import os
import csv
import datetime
import json
from typing import Dict, Any, List, Optional

class MetricsLogger:
    """
    记录模型运行指标到CSV文件的工具类
    """
    def __init__(self, output_dir: str, run_id: Optional[str] = None):
        """
        初始化指标记录器
        
        Args:
            output_dir: 输出目录
            run_id: 运行ID，如不提供则使用时间戳
        """
        self.output_dir = output_dir
        
        # 如果run_id未提供，使用当前时间戳
        if run_id is None:
            self.run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.run_id = run_id
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV文件路径
        self.csv_path = os.path.join(output_dir, f"metrics_{self.run_id}.csv")
        
        # 记录是否已初始化CSV文件
        self.csv_initialized = False
        
        # 存储指标历史
        self.metrics_history = []

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        记录一组指标
        
        Args:
            metrics: 指标字典
            step: 当前步骤编号（可选）
        """
        # 添加步骤信息
        metrics_with_step = metrics.copy()
        if step is not None:
            metrics_with_step['step'] = step
            
        # 添加运行ID
        metrics_with_step['run_id'] = self.run_id
        
        # 添加时间戳
        metrics_with_step['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 存储到历史记录
        self.metrics_history.append(metrics_with_step)
        
        # 写入CSV
        self._write_to_csv(metrics_with_step)
        
    def _write_to_csv(self, metrics: Dict[str, Any]):
        """
        将指标写入CSV文件
        
        Args:
            metrics: 指标字典
        """
        # 检查CSV是否已初始化
        if not self.csv_initialized:
            # 首次写入，创建表头
            fieldnames = list(metrics.keys())
            
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(metrics)
                
            self.csv_initialized = True
        else:
            # 追加写入
            # 检查是否有新字段
            existing_fieldnames = []
            if os.path.exists(self.csv_path):
                with open(self.csv_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    existing_fieldnames = next(reader, [])
            
            fieldnames = list(set(existing_fieldnames) | set(metrics.keys()))
            
            # 读取现有数据
            existing_data = []
            if os.path.exists(self.csv_path):
                with open(self.csv_path, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_data = list(reader)
            
            # 写入所有数据
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入现有数据
                for row in existing_data:
                    writer.writerow(row)
                
                # 写入新数据
                writer.writerow(metrics)
                
    def save_summary(self):
        """
        保存指标汇总和统计信息
        """
        if not self.metrics_history:
            return
            
        # 计算每个指标的均值和标准差
        # 略过非数值字段
        numeric_metrics = {}
        
        # 提取所有指标名称
        all_metric_names = set()
        for metrics in self.metrics_history:
            all_metric_names.update(metrics.keys())
            
        # 筛选数值型指标
        for metric_name in all_metric_names:
            values = []
            for metrics in self.metrics_history:
                if metric_name in metrics:
                    try:
                        value = float(metrics[metric_name])
                        values.append(value)
                    except (ValueError, TypeError):
                        # 非数值型跳过
                        pass
            
            if values:
                # 计算均值和标准差
                import numpy as np
                mean = np.mean(values)
                std = np.std(values)
                numeric_metrics[metric_name] = {
                    'mean': mean,
                    'std': std,
                    'min': min(values),
                    'max': max(values)
                }
                
        # 保存汇总
        summary_path = os.path.join(self.output_dir, f"summary_{self.run_id}.json")
        with open(summary_path, 'w') as jsonfile:
            json.dump({
                'run_id': self.run_id,
                'metrics_count': len(self.metrics_history),
                'metrics_stats': numeric_metrics
            }, jsonfile, indent=2)
            