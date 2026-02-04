import time
import numpy as np
from typing import List, Dict

from algorithms.base import BaseAlgorithm, RunResult
from evaluation.fitness import FitnessEvaluator
from evaluation.pareto import ParetoOperations
from evaluation.metrics import MetricsCalculator
from operators.crossover import Crossover
from operators.mutation import Mutation
from operators.selection import Selection


class NSGA2(BaseAlgorithm):
    def __init__(
        self,
        pop_size: int = 100,
        max_generations: int = 100,
        crossover_prob: float = 1.0,
        mutation_prob: float = None,
        seed: int = 42,
        k_neighbors: int = 5,
        cv_folds: int = 5,
        convergence_interval: int = 10,
        hv_ref_point: tuple = (1.0, 1.0)
    ):
        super().__init__(pop_size, max_generations, crossover_prob, mutation_prob, seed)
        self.k_neighbors = k_neighbors
        self.cv_folds = cv_folds
        self.convergence_interval = convergence_interval
        self.hv_ref_point = hv_ref_point
        self.evaluator = None

    def initialize_population(self) -> List[np.ndarray]:
        """
        初始化种群
        使用随机二进制编码，每个个体至少选择一个特征
        """
        population = []
        for _ in range(self.pop_size):
            # 随机生成二进制向量
            individual = self.rng.integers(0, 2, size=self.n_features)
            # 确保至少选择一个特征
            individual = Mutation.ensure_not_empty(individual, self.rng)
            population.append(individual)
        return population

    def evolve(self) -> RunResult:
        """
        执行NSGA-II进化过程
        """
        # 初始化适应度评估器
        self.evaluator = FitnessEvaluator(
            self.X_train, self.y_train,
            k_neighbors=self.k_neighbors,
            cv_folds=self.cv_folds,
            seed=self.seed
        )

        # 初始化种群
        self.population = self.initialize_population()

        # 记录开始时间（种群初始化完成后）
        start_time = time.time()

        # 评估初始种群
        self.fitness = self.evaluator.evaluate_population(self.population)

        # 收敛数据记录
        convergence_data = []

        # 主循环
        for gen in range(self.max_generations):
            self.generation = gen

            # 记录收敛数据（每隔interval代记录一次，包括第0代）
            if gen % self.convergence_interval == 0:
                conv_record = self._record_convergence(gen)
                convergence_data.append(conv_record)

            # 生成子代
            offspring = self._generate_offspring()

            # 评估子代
            offspring_fitness = self.evaluator.evaluate_population(offspring)

            # 合并父代和子代
            combined_population = self.population + offspring
            combined_fitness = np.vstack([self.fitness, offspring_fitness])

            # 环境选择
            self.population, self.fitness = self._environmental_selection(
                combined_population, combined_fitness
            )

        # 记录最后一代（第100代，即gen=99后的状态）
        conv_record = self._record_convergence(self.max_generations)
        convergence_data.append(conv_record)

        # 记录结束时间
        end_time = time.time()
        run_time = end_time - start_time

        # 在测试集上评估最终Pareto前沿
        result = self._evaluate_final_result(convergence_data, run_time)

        return result

    def _generate_offspring(self) -> List[np.ndarray]:
        """生成子代种群"""
        # 计算非支配等级和拥挤度距离
        fronts = ParetoOperations.fast_non_dominated_sort(self.fitness)
        ranks = ParetoOperations.get_ranks(self.fitness)
        crowding_distances = ParetoOperations.get_all_crowding_distances(
            self.fitness, fronts
        )

        offspring = []
        while len(offspring) < self.pop_size:
            # 锦标赛选择两个父代
            parent_indices = Selection.select_parents(
                self.population, self.fitness, ranks, crowding_distances,
                n_parents=2, rng=self.rng
            )
            parent1 = self.population[parent_indices[0]]
            parent2 = self.population[parent_indices[1]]

            # 交叉
            if self.rng.random() < self.crossover_prob:
                child1, child2 = Crossover.single_point(parent1, parent2, self.rng)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # 变异
            child1 = Mutation.bit_flip(child1, self.mutation_prob, self.rng)
            child2 = Mutation.bit_flip(child2, self.mutation_prob, self.rng)

            # 确保至少选择一个特征
            child1 = Mutation.ensure_not_empty(child1, self.rng)
            child2 = Mutation.ensure_not_empty(child2, self.rng)

            offspring.append(child1)
            if len(offspring) < self.pop_size:
                offspring.append(child2)

        return offspring

    def _environmental_selection(
        self,
        combined_population: List[np.ndarray],
        combined_fitness: np.ndarray
    ) -> tuple:
        """
        环境选择：基于非支配排序和拥挤度距离
        """
        # 非支配排序
        fronts = ParetoOperations.fast_non_dominated_sort(combined_fitness)

        new_population = []
        new_fitness = []

        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                # 整个前沿都可以加入
                for idx in front:
                    new_population.append(combined_population[idx])
                    new_fitness.append(combined_fitness[idx])
            else:
                # 需要根据拥挤度距离选择
                remaining = self.pop_size - len(new_population)
                if remaining > 0:
                    # 计算这个前沿的拥挤度距离
                    front_fitness = combined_fitness[front]
                    distances = ParetoOperations.crowding_distance(
                        combined_fitness, front
                    )
                    # 按拥挤度降序排序
                    sorted_indices = np.argsort(-distances)
                    for i in range(remaining):
                        idx = front[sorted_indices[i]]
                        new_population.append(combined_population[idx])
                        new_fitness.append(combined_fitness[idx])
                break

        return new_population, np.array(new_fitness)

    def _record_convergence(self, generation: int) -> Dict:
        """记录收敛数据"""
        # 提取当前Pareto前沿
        pareto_fitness, _ = ParetoOperations.extract_pareto_front(
            self.fitness, self.population
        )

        # 计算HV（基于训练集的目标值）
        hv_value = MetricsCalculator.hypervolume(pareto_fitness, self.hv_ref_point)

        # 最小错误率和最小特征数
        best_error = np.min(self.fitness[:, 0])
        min_feature_ratio = np.min(self.fitness[:, 1])
        min_features = int(min_feature_ratio * self.n_features)

        return {
            "generation": generation,
            "hv_value": hv_value,
            "best_error": best_error,
            "min_features": min_features,
            "feature_ratio": min_feature_ratio
        }

    def _evaluate_final_result(
        self,
        convergence_data: List[Dict],
        run_time: float
    ) -> RunResult:
        """在测试集上评估最终结果"""
        # 获取训练阶段的所有解
        all_solutions = self.population
        all_fitness = self.fitness

        # 在测试集上重新评估所有解
        test_fitness = []
        for individual in all_solutions:
            test_error, train_error = self.evaluator.evaluate_on_test(
                individual, self.X_test, self.y_test
            )
            selected_count = np.sum(individual)
            feature_ratio = selected_count / self.n_features
            test_fitness.append([test_error, feature_ratio])

        test_fitness = np.array(test_fitness)

        # 在测试集目标值上进行非支配排序，提取Pareto第一前沿
        pareto_fitness, pareto_solutions = ParetoOperations.extract_pareto_front(
            test_fitness, all_solutions
        )

        # 计算HV（基于测试集目标值）
        hv_value = MetricsCalculator.hypervolume(pareto_fitness, self.hv_ref_point)

        return RunResult(
            pareto_front=pareto_fitness,
            pareto_solutions=pareto_solutions,
            hv_value=hv_value,
            convergence_data=convergence_data,
            run_time=run_time
        )

    def get_algorithm_name(self) -> str:
        return "NSGA-II"
