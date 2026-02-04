import numpy as np

class Mutation:
    """变异算子"""

    @staticmethod
    def bit_flip(
        individual: np.ndarray,
        mutation_prob: float,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        位翻转变异

        Args:
            individual: 个体的二进制编码
            mutation_prob: 每个位的变异概率 (通常为 1/D)
            rng: 随机数生成器

        Returns:
            变异后的个体
        """
        mutant = individual.copy()
        mask = rng.random(len(individual)) < mutation_prob
        mutant[mask] = 1 - mutant[mask]
        return mutant

    @staticmethod
    def ensure_not_empty(
        individual: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        确保个体至少选择一个特征

        Args:
            individual: 个体的二进制编码
            rng: 随机数生成器

        Returns:
            修正后的个体
        """
        if np.sum(individual) == 0:
            # 随机选择一个位置置为1
            idx = rng.integers(len(individual))
            individual[idx] = 1
        return individual
