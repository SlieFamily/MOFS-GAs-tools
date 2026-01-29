"""
Pareto操作模块
实现非支配排序和拥挤度距离计算
"""
import numpy as np
from typing import List, Tuple


class ParetoOperations:
    """Pareto相关操作"""

    @staticmethod
    def dominates(obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        判断obj1是否支配obj2（最小化问题）

        Args:
            obj1: 第一个目标向量
            obj2: 第二个目标向量

        Returns:
            True如果obj1支配obj2
        """
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)

    @staticmethod
    def fast_non_dominated_sort(fitness: np.ndarray) -> List[List[int]]:
        """
        快速非支配排序

        Args:
            fitness: 适应度矩阵 (n_individuals, n_objectives)

        Returns:
            fronts: 各层前沿的索引列表
        """
        n = len(fitness)
        domination_count = np.zeros(n, dtype=int)  # 被支配次数
        dominated_set = [[] for _ in range(n)]  # 支配的个体集合
        ranks = np.zeros(n, dtype=int)
        fronts = [[]]

        # 计算支配关系
        for i in range(n):
            for j in range(i + 1, n):
                if ParetoOperations.dominates(fitness[i], fitness[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif ParetoOperations.dominates(fitness[j], fitness[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        # 找出第一层前沿
        for i in range(n):
            if domination_count[i] == 0:
                ranks[i] = 0
                fronts[0].append(i)

        # 逐层构建前沿
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        ranks[j] = current_front + 1
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        # 移除最后的空前沿
        fronts.pop()

        return fronts

    @staticmethod
    def get_ranks(fitness: np.ndarray) -> np.ndarray:
        """
        获取所有个体的非支配等级

        Args:
            fitness: 适应度矩阵

        Returns:
            ranks: 等级数组
        """
        n = len(fitness)
        ranks = np.zeros(n, dtype=int)
        fronts = ParetoOperations.fast_non_dominated_sort(fitness)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank

        return ranks

    @staticmethod
    def crowding_distance(fitness: np.ndarray, front: List[int]) -> np.ndarray:
        """
        计算拥挤度距离

        Args:
            fitness: 适应度矩阵
            front: 前沿中的个体索引

        Returns:
            distances: 拥挤度距离数组（对应front中的个体）
        """
        n = len(front)
        if n == 0:
            return np.array([])

        n_objectives = fitness.shape[1]
        distances = np.zeros(n)

        for m in range(n_objectives):
            # 按第m个目标排序
            sorted_indices = np.argsort([fitness[i, m] for i in front])
            sorted_front = [front[i] for i in sorted_indices]

            # 边界个体设为无穷大
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # 计算目标范围
            f_min = fitness[sorted_front[0], m]
            f_max = fitness[sorted_front[-1], m]
            f_range = f_max - f_min

            if f_range == 0:
                continue

            # 计算中间个体的拥挤度
            for i in range(1, n - 1):
                distances[sorted_indices[i]] += (
                    fitness[sorted_front[i + 1], m] - fitness[sorted_front[i - 1], m]
                ) / f_range

        return distances

    @staticmethod
    def get_all_crowding_distances(
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> np.ndarray:
        """
        计算所有个体的拥挤度距离

        Args:
            fitness: 适应度矩阵
            fronts: 前沿列表

        Returns:
            distances: 所有个体的拥挤度距离
        """
        n = len(fitness)
        distances = np.zeros(n)

        for front in fronts:
            if len(front) > 0:
                front_distances = ParetoOperations.crowding_distance(fitness, front)
                for i, idx in enumerate(front):
                    distances[idx] = front_distances[i]

        return distances

    @staticmethod
    def extract_pareto_front(
        fitness: np.ndarray,
        population: List[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        提取Pareto第一前沿

        Args:
            fitness: 适应度矩阵
            population: 种群（可选）

        Returns:
            pareto_fitness: 前沿的适应度
            pareto_individuals: 前沿的个体（如果提供population）
        """
        fronts = ParetoOperations.fast_non_dominated_sort(fitness)
        pareto_indices = fronts[0]

        pareto_fitness = fitness[pareto_indices]

        if population is not None:
            pareto_individuals = [population[i] for i in pareto_indices]
            return pareto_fitness, pareto_individuals

        return pareto_fitness, None
