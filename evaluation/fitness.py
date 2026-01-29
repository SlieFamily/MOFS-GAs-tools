"""
适应度评估模块
使用KNN进行特征子集评估
"""
import numpy as np
from typing import Tuple, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


class FitnessEvaluator:
    """适应度评估器"""

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        k_neighbors: int = 5,
        cv_folds: int = 5,
        seed: int = 42
    ):
        """
        初始化适应度评估器

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            k_neighbors: KNN的k值
            cv_folds: 交叉验证折数
            seed: 随机种子
        """
        self.X_train = X_train
        self.y_train = y_train
        self.k_neighbors = k_neighbors
        self.cv_folds = cv_folds
        self.seed = seed
        self.n_features = X_train.shape[1]

    def evaluate(self, individual: np.ndarray) -> Tuple[float, float]:
        """
        评估单个个体的适应度

        Args:
            individual: 二进制编码的个体

        Returns:
            (error_rate, feature_ratio): 分类错误率和特征选择率
        """
        # 获取选中的特征索引
        selected_indices = np.where(individual == 1)[0]

        # 空特征集处理
        if len(selected_indices) == 0:
            return (1.0, 0.0)

        # 提取选中的特征
        X_selected = self.X_train[:, selected_indices]

        # 5折分层交叉验证
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.seed
        )

        errors = []
        for train_idx, val_idx in skf.split(X_selected, self.y_train):
            X_train_fold = X_selected[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = X_selected[val_idx]
            y_val_fold = self.y_train[val_idx]

            # KNN分类
            knn = KNeighborsClassifier(
                n_neighbors=self.k_neighbors,
                metric='euclidean'
            )
            knn.fit(X_train_fold, y_train_fold)
            y_pred = knn.predict(X_val_fold)

            # 计算错误率
            error = 1.0 - np.mean(y_pred == y_val_fold)
            errors.append(error)

        # 平均错误率
        avg_error = np.mean(errors)

        # 特征选择率
        feature_ratio = len(selected_indices) / self.n_features

        return (avg_error, feature_ratio)

    def evaluate_population(
        self, population: List[np.ndarray]
    ) -> np.ndarray:
        """
        评估整个种群

        Args:
            population: 种群（个体列表）

        Returns:
            fitness: 适应度矩阵 (n_individuals, 2)
        """
        fitness = np.array([self.evaluate(ind) for ind in population])
        return fitness

    def evaluate_on_test(
        self,
        individual: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[float, float]:
        """
        在测试集上评估个体

        Args:
            individual: 二进制编码的个体
            X_test: 测试集特征
            y_test: 测试集标签

        Returns:
            (test_error, train_error): 测试集错误率和训练集错误率
        """
        selected_indices = np.where(individual == 1)[0]

        if len(selected_indices) == 0:
            return (1.0, 1.0)

        X_train_selected = self.X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]

        knn = KNeighborsClassifier(
            n_neighbors=self.k_neighbors,
            metric='euclidean'
        )
        knn.fit(X_train_selected, self.y_train)

        # 测试集错误率
        y_pred_test = knn.predict(X_test_selected)
        test_error = 1.0 - np.mean(y_pred_test == y_test)

        # 训练集错误率
        y_pred_train = knn.predict(X_train_selected)
        train_error = 1.0 - np.mean(y_pred_train == self.y_train)

        return (test_error, train_error)
