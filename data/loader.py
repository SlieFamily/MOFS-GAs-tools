"""
数据加载模块
负责从CSV文件加载数据集
"""
import numpy as np
import pandas as pd
from typing import Tuple


class DataLoader:

    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            filepath: CSV文件路径

        Returns:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)

        Note:
            自动检测是否有表头，最后一列为标签，前面列为特征
        """
        first_row = pd.read_csv(filepath, nrows=1, header=None)
        first_value = first_row.iloc[0, 0]

        # 判断第一个值是否为数值
        has_header = False
        try:
            float(first_value)
        except (ValueError, TypeError):
            has_header = True

        # 根据是否有表头读取数据 
        if has_header:
            data = pd.read_csv(filepath, low_memory=False)
        else:
            data = pd.read_csv(filepath, header=None, low_memory=False)

        # 转换为numpy数组
        data_array = data.values

        # 分离特征和标签（最后一列为标签）
        X = data_array[:, :-1].astype(np.float64)
        y = data_array[:, -1].astype(np.float64)

        return X, y

    @staticmethod
    def get_dataset_info(filepath: str) -> dict:
        """
        获取数据集基本信息

        Args:
            filepath: CSV文件路径

        Returns:
            包含数据集信息的字典
        """
        X, y = DataLoader.load(filepath)
        unique_labels = np.unique(y)

        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "n_classes": len(unique_labels),
            "class_labels": unique_labels.tolist(),
            "class_distribution": {
                label: int(np.sum(y == label)) for label in unique_labels
            }
        }
