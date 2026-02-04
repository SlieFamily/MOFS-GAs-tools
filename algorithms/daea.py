"""
DAEA (Duplication Analysis-based Evolutionary Algorithm) 实现
论文: A Duplication Analysis-Based Evolutionary Algorithm for Biobjective Feature Selection
      Hang Xu, Bing Xue, Mengjie Zhang (IEEE TEVC, 2021)

对 NSGA-II 的三大改进：
1. 重复解分析 — 过滤目标空间中的冗余重复解
2. 基于多样性的选择 — 替代传统拥挤度距离截断
3. 改进的繁殖过程 — 邻域配对 + k-bit交叉 + 混合变异
"""
import math
import time
import numpy as np
from typing import List, Dict, Tuple

from algorithms.base import BaseAlgorithm, RunResult
from evaluation.fitness import FitnessEvaluator
from evaluation.pareto import ParetoOperations
from evaluation.metrics import MetricsCalculator
from operators.mutation import Mutation


class DAEA(BaseAlgorithm):

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

    # ================================================================
    # 初始化 (Algorithm 1, lines 1-5)
    # ================================================================
    def initialize_population(self) -> List[np.ndarray]:
        """
        DAEA 初始化：高维时限制每个个体的选择特征数
        若 3*N < D，每个个体选择的特征数上限为 3*N
        """
        population = []
        if 3 * self.pop_size < self.n_features:
            max_selected = 3 * self.pop_size
            for _ in range(self.pop_size):
                individual = np.zeros(self.n_features, dtype=int)
                num_selected = self.rng.integers(1, max_selected + 1)
                selected_idx = self.rng.choice(self.n_features, size=num_selected, replace=False)
                individual[selected_idx] = 1
                population.append(individual)
        else:
            for _ in range(self.pop_size):
                individual = self.rng.integers(0, 2, size=self.n_features)
                individual = Mutation.ensure_not_empty(individual, self.rng)
                population.append(individual)
        return population

    # ================================================================
    # 主进化循环
    # ================================================================
    def evolve(self) -> RunResult:
        self.evaluator = FitnessEvaluator(
            self.X_train, self.y_train,
            k_neighbors=self.k_neighbors,
            cv_folds=self.cv_folds,
            seed=self.seed
        )

        self.population = self.initialize_population()
        start_time = time.time()
        self.fitness = self.evaluator.evaluate_population(self.population)

        convergence_data = []

        for gen in range(self.max_generations):
            self.generation = gen

            if gen % self.convergence_interval == 0:
                convergence_data.append(self._record_convergence(gen))

            # 改进的繁殖过程 (Algorithm 4)
            offspring = self._reproduce()

            # 评估子代
            offspring_fitness = self.evaluator.evaluate_population(offspring)

            # 合并父代和子代
            combined_pop = self.population + offspring
            combined_fit = np.vstack([self.fitness, offspring_fitness])

            # 环境选择（含重复解分析 + 多样性选择）
            self.population, self.fitness = self._environmental_selection(
                combined_pop, combined_fit
            )

        convergence_data.append(self._record_convergence(self.max_generations))
        end_time = time.time()

        return self._evaluate_final_result(convergence_data, end_time - start_time)

    # ================================================================
    # 改进的繁殖 (Algorithm 4 + Algorithm 5)
    # ================================================================
    def _reproduce(self) -> List[np.ndarray]:
        """DAEA 繁殖过程：邻域配对 + k-bit交叉 + 混合变异"""
        N = len(self.population)
        T = max(4, math.ceil(N * 0.2))  # 邻域大小

        # --- 邻域配对 (Niche Mating) ---
        # 归一化目标值到 [0,1]
        obj = self.fitness.copy()
        obj_min = obj.min(axis=0)
        obj_max = obj.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1.0
        obj_norm = (obj - obj_min) / obj_range

        # 构建邻域矩阵：每个个体的 T 个最近邻
        from scipy.spatial.distance import cdist
        dist_obj = cdist(obj_norm, obj_norm, metric='euclidean')
        neighborhoods = np.argsort(dist_obj, axis=1)[:, 1:T + 1]  # 排除自身

        # 为每个个体选择配偶
        parents_idx = np.empty(N, dtype=int)
        for i in range(N):
            if self.rng.random() < 0.8:
                parents_idx[i] = self.rng.choice(neighborhoods[i])
            else:
                parents_idx[i] = self.rng.integers(N)

        # --- k-bit 交叉 ---
        offspring_arr = []
        for i in range(N):
            pop_i = self.population[i].copy()
            par_i = self.population[parents_idx[i]]

            diff_bits = np.where(pop_i != par_i)[0]
            if len(diff_bits) > 1:
                # 随机选 k ∈ [1, |diff|-1] 个位交换
                k = self.rng.integers(1, len(diff_bits))
                selected_bits = self.rng.choice(diff_bits, size=k, replace=False)
                child = pop_i.copy()
                child[selected_bits] = par_i[selected_bits]
                offspring_arr.append(child)
            else:
                offspring_arr.append(pop_i.copy())

        # --- 混合变异 (Algorithm 5) ---
        for i in range(len(offspring_arr)):
            offspring_arr[i] = self._hybrid_mutate(offspring_arr[i])
            offspring_arr[i] = Mutation.ensure_not_empty(offspring_arr[i], self.rng)

        return offspring_arr

    def _hybrid_mutate(self, individual: np.ndarray) -> np.ndarray:
        """混合变异 (Algorithm 5)"""
        child = individual.copy()
        if self.rng.random() < 0.2:
            # 改进变异：分别处理已选和未选特征
            t1 = np.where(child == 1)[0]
            if len(t1) > 0:
                prob1 = 1.0 / (1 + len(t1))
                mask1 = self.rng.random(len(t1)) < prob1
                child[t1[mask1]] = 0

            t0 = np.where(child == 0)[0]
            if len(t0) > 0:
                prob0 = 1.0 / (1 + len(t0))
                mask0 = self.rng.random(len(t0)) < prob0
                child[t0[mask0]] = 1
        else:
            # 传统位翻转变异
            child = Mutation.bit_flip(child, self.mutation_prob, self.rng)
        return child

    # ================================================================
    # 环境选择（整合三步）
    # ================================================================
    def _environmental_selection(
        self,
        combined_pop: List[np.ndarray],
        combined_fit: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        DAEA 环境选择：
        1. 移除决策空间重复解
        2. 重复解分析（目标空间）
        3. 非支配排序 + 多样性选择
        """
        pop, fit = self._remove_decision_duplicates(combined_pop, combined_fit)
        pop, fit = self._duplication_analysis(pop, fit)

        # 非支配排序
        fronts = ParetoOperations.fast_non_dominated_sort(fit)

        # 找临界前沿
        alpha = []  # 前 k-1 层全部入选的索引
        beta = []   # 第 k 层需截断的索引

        for front in fronts:
            if len(alpha) + len(front) <= self.pop_size:
                alpha.extend(front)
            else:
                beta = front
                break

        # 如果已选满或需要截断
        if len(alpha) >= self.pop_size:
            selected = alpha[:self.pop_size]
            new_pop = [pop[i] for i in selected]
            new_fit = fit[selected]
            return new_pop, new_fit

        # 多样性选择
        return self._diversity_selection(pop, fit, alpha, beta)

    # ================================================================
    # 重复解分析 (Algorithm 2)
    # ================================================================
    def _duplication_analysis(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        目标空间重复解分析。
        对于映射到相同目标向量的解，基于决策空间曼哈顿距离
        过滤不相似度低的冗余解。
        """
        n = len(population)
        if n <= 1:
            return population, fitness

        D = self.n_features

        # 将种群转为矩阵用于距离计算
        pop_matrix = np.array([ind for ind in population])

        # 曼哈顿距离矩阵
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(pop_matrix, pop_matrix, metric='cityblock')

        # 找出所有唯一目标向量及其分组
        # 使用四舍五入避免浮点精度问题
        rounded_fit = np.round(fitness, decimals=10)
        unique_objs, inverse = np.unique(rounded_fit, axis=0, return_inverse=True)

        removal_set = set()

        for u_idx in range(len(unique_objs)):
            group = np.where(inverse == u_idx)[0]
            if len(group) <= 1:
                continue

            # t: 该组解选择的特征数（用第一个解的，同组目标值相同所以 f1 相同）
            t = int(np.sum(population[group[0]]))
            if t == 0:
                t = 1  # 防止除零

            # 计算每个解的不相似度
            diss = np.zeros(len(group))
            for k_idx in range(len(group)):
                # 到组内其他解的最小曼哈顿距离
                others = [group[j] for j in range(len(group)) if j != k_idx]
                min_dist = np.min(dist_matrix[group[k_idx], others])
                diss[k_idx] = (min_dist / 2.0) / t

            # 阈值
            threshold = 0.8 - 0.6 * (t - 1) / max(D - 1, 1)

            # 找出 clustered 解（diss < threshold）
            clustered_mask = diss < threshold
            clustered_indices = group[clustered_mask]

            if len(clustered_indices) > 1:
                # 随机保留一个，其余加入移除集
                to_remove = self.rng.choice(
                    clustered_indices, size=len(clustered_indices) - 1, replace=False
                )
                removal_set.update(to_remove.tolist())

        # 过滤
        keep = [i for i in range(n) if i not in removal_set]
        if len(keep) == 0:
            keep = [0]  # 至少保留一个

        new_pop = [population[i] for i in keep]
        new_fit = fitness[keep]
        return new_pop, new_fit

    # ================================================================
    # 基于多样性的选择 (Algorithm 3)
    # ================================================================
    def _diversity_selection(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        alpha: List[int],
        beta: List[int]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        基于多样性的选择方法。
        主标准：已选解中同特征数（f1值）的个数最少优先
        次标准：拥挤度距离大者优先
        """
        selected = list(alpha)
        remaining = list(beta)

        # 计算 beta 中各解的拥挤度距离
        if len(remaining) > 0:
            crowding = ParetoOperations.crowding_distance(fitness, remaining)
        else:
            crowding = np.array([])

        while len(selected) < self.pop_size and len(remaining) > 0:
            # 按拥挤度降序排序 remaining
            crowd_vals = np.array([
                crowding[remaining.index(idx)] if idx in remaining else 0.0
                for idx in remaining
            ])
            # 重新计算：直接用 remaining 中各元素对应的拥挤度
            remaining_crowding = []
            for r_idx in range(len(remaining)):
                remaining_crowding.append(crowding[r_idx])
            remaining_crowding = np.array(remaining_crowding)

            sorted_order = np.argsort(-remaining_crowding)
            remaining = [remaining[i] for i in sorted_order]
            crowding = crowding[sorted_order]

            # 计算多样性分数
            div_scores = np.zeros(len(remaining))
            for i, cand_idx in enumerate(remaining):
                cand_f1 = fitness[cand_idx, 0]  # 特征选择率 (第1个目标是error, 第2个是feature_ratio)
                # 论文中比较的是 f1（即特征选择率），对应我们的 fitness[:, 1]
                cand_fr = fitness[cand_idx, 1]
                count = 0
                for s_idx in selected:
                    if abs(fitness[s_idx, 1] - cand_fr) < 1e-10:
                        count += 1
                div_scores[i] = count

            # 选多样性分数最小的（已按拥挤度排序，所以平局自动取拥挤度大的）
            best_local = np.argmin(div_scores)
            best_idx = remaining[best_local]

            selected.append(best_idx)
            remaining.pop(best_local)
            crowding = np.delete(crowding, best_local)

        # 构建结果
        new_pop = [population[i] for i in selected]
        new_fit = fitness[selected]
        return new_pop, new_fit

    # ================================================================
    # 辅助方法
    # ================================================================
    @staticmethod
    def _remove_decision_duplicates(
        population: List[np.ndarray],
        fitness: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """移除决策空间完全重复的解"""
        seen = {}
        keep = []
        for i, ind in enumerate(population):
            key = ind.tobytes()
            if key not in seen:
                seen[key] = True
                keep.append(i)

        if len(keep) == 0:
            keep = [0]

        new_pop = [population[i] for i in keep]
        new_fit = fitness[keep]
        return new_pop, new_fit

    def _record_convergence(self, generation: int) -> Dict:
        """记录收敛数据"""
        pareto_fitness, _ = ParetoOperations.extract_pareto_front(
            self.fitness, self.population
        )
        hv_value = MetricsCalculator.hypervolume(pareto_fitness, self.hv_ref_point)
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
        test_fitness = []
        for individual in self.population:
            test_error, _ = self.evaluator.evaluate_on_test(
                individual, self.X_test, self.y_test
            )
            feature_ratio = np.sum(individual) / self.n_features
            test_fitness.append([test_error, feature_ratio])

        test_fitness = np.array(test_fitness)

        pareto_fitness, pareto_solutions = ParetoOperations.extract_pareto_front(
            test_fitness, self.population
        )
        hv_value = MetricsCalculator.hypervolume(pareto_fitness, self.hv_ref_point)

        return RunResult(
            pareto_front=pareto_fitness,
            pareto_solutions=pareto_solutions,
            hv_value=hv_value,
            convergence_data=convergence_data,
            run_time=run_time
        )

    def get_algorithm_name(self) -> str:
        return "DAEA"
