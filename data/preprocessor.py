"""
数据预处理模块
负责缺失值处理、标准化和数据集划分
"""
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        """
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.test_size = test_size
        self.random_state = random_state
        self.imputer = None
        self.scaler = None

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        预处理数据：缺失值填充、标准化、划分数据集

        Args:
            X: 特征矩阵
            y: 标签向量

        Returns:
            X_train: 训练集特征
            X_test: 测试集特征
            y_train: 训练集标签
            y_test: 测试集标签
        """
        # 1. 分层划分数据集（先划分，避免数据泄露）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # 2. 缺失值处理：中位数填充
        self.imputer = SimpleImputer(strategy='median')
        X_train = self.imputer.fit_transform(X_train)
        X_test = self.imputer.transform(X_test)

        # 3. 标准化：仅在训练集上fit，然后transform两者
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用已拟合的参数转换新数据

        Args:
            X: 特征矩阵

        Returns:
            转换后的特征矩阵
        """
        if self.imputer is None or self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X
