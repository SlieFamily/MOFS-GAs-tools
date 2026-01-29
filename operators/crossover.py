"""
交叉算子模块
实现二进制编码的交叉操作
"""
import numpy as np
from typing import Tuple


class Crossover:
    """交叉算子"""

    @staticmethod
    def single_point(
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        单点交叉

        Args:
            parent1: 父代1的二进制编码
            parent2: 父代2的二进制编码
            rng: 随机数生成器

        Returns:
            child1, child2: 两个子代
        """
        n = len(parent1)
        point = rng.integers(1, n)

        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1.copy(), child2.copy()

    @staticmethod
    def two_point(
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        两点交叉

        Args:
            parent1: 父代1的二进制编码
            parent2: 父代2的二进制编码
            rng: 随机数生成器

        Returns:
            child1, child2: 两个子代
        """
        n = len(parent1)
        points = sorted(rng.choice(n, size=2, replace=False))
        p1, p2 = points[0], points[1]

        child1 = np.concatenate([parent1[:p1], parent2[p1:p2], parent1[p2:]])
        child2 = np.concatenate([parent2[:p1], parent1[p1:p2], parent2[p2:]])

        return child1.copy(), child2.copy()

    @staticmethod
    def uniform(
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        均匀交叉

        Args:
            parent1: 父代1的二进制编码
            parent2: 父代2的二进制编码
            rng: 随机数生成器
            prob: 每个位置交换的概率

        Returns:
            child1, child2: 两个子代
        """
        mask = rng.random(len(parent1)) < prob

        child1 = np.where(mask, parent2, parent1)
        child2 = np.where(mask, parent1, parent2)

        return child1.copy(), child2.copy()
