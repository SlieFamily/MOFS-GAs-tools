"""
选择算子模块
实现锦标赛选择等操作
"""
import numpy as np
from typing import List, Tuple


class Selection:
    """选择算子"""

    @staticmethod
    def binary_tournament(
        population: List[np.ndarray],
        fitness: np.ndarray,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        基于非支配排序和拥挤度的二元锦标赛选择

        Args:
            population: 种群列表
            fitness: 适应度矩阵 (n_individuals, n_objectives)
            ranks: 非支配等级
            crowding_distances: 拥挤度距离
            rng: 随机数生成器

        Returns:
            选中个体的索引
        """
        pop_size = len(population)
        i, j = rng.choice(pop_size, size=2, replace=False)

        # 比较规则：
        # 1. 等级低（更好）的获胜
        # 2. 等级相同时，拥挤度大的获胜
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        elif crowding_distances[i] > crowding_distances[j]:
            return i
        elif crowding_distances[j] > crowding_distances[i]:
            return j
        else:
            return rng.choice([i, j])

    @staticmethod
    def select_parents(
        population: List[np.ndarray],
        fitness: np.ndarray,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        n_parents: int,
        rng: np.random.Generator
    ) -> List[int]:
        """
        选择多个父代

        Args:
            population: 种群列表
            fitness: 适应度矩阵
            ranks: 非支配等级
            crowding_distances: 拥挤度距离
            n_parents: 需要选择的父代数量
            rng: 随机数生成器

        Returns:
            选中父代的索引列表
        """
        selected = []
        for _ in range(n_parents):
            idx = Selection.binary_tournament(
                population, fitness, ranks, crowding_distances, rng
            )
            selected.append(idx)
        return selected
