"""
PRDH (Problem Reformulation and Duplication Handling) 实现
论文: Solving Multiobjective Feature Selection Problems in Classification via
      Problem Reformulation and Duplication Handling
      Ruwang Jiao, Bing Xue, Mengjie Zhang (IEEE TEVC, 2024)

核心创新：
1. 问题重构 — 将多目标问题重构为约束多目标问题
2. 约束处理 — 优先选择分类性能好的解（可行解）
3. 重复解处理 — 基于目标空间重复和决策空间相似度
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Set

from algorithms.base import BaseAlgorithm, RunResult
from evaluation.fitness import FitnessEvaluator
from evaluation.pareto import ParetoOperations
from evaluation.metrics import MetricsCalculator
from operators.crossover import Crossover
from operators.mutation import Mutation
from operators.selection import Selection


class PRDH(BaseAlgorithm):
    """
    PRDH: Problem Reformulation and Duplication Handling

    主要特点：
    - 问题重构：根据非支配前沿动态设置分类性能约束
    - 约束处理：优先保留分类性能好的解
    - 重复解处理：移除目标空间重复且决策空间相似的解
    """

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
    # 初始化
    # ================================================================
    def initialize_population(self) -> List[np.ndarray]:
        """
        初始化种群
        高维数据时限制每个个体的特征选择数量
        """
        population = []
        # 如果特征数很多，限制初始选择的特征数
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
        """执行 PRDH 进化过程"""
        # 初始化适应度评估器
        self.evaluator = FitnessEvaluator(
            self.X_train, self.y_train,
            k_neighbors=self.k_neighbors,
            cv_folds=self.cv_folds,
            seed=self.seed
        )

        # 初始化种群
        self.population = self.initialize_population()

        # 记录开始时间
        start_time = time.time()

        # 评估初始种群
        self.fitness = self.evaluator.evaluate_population(self.population)

        # 收敛数据记录
        convergence_data = []

        # 主循环
        for gen in range(self.max_generations):
            self.generation = gen

            # 记录收敛数据
            if gen % self.convergence_interval == 0:
                convergence_data.append(self._record_convergence(gen))

            # 生成子代（标准交叉变异）
            offspring = self._generate_offspring()

            # 评估子代
            offspring_fitness = self.evaluator.evaluate_population(offspring)

            # 合并父代和子代
            combined_population = self.population + offspring
            combined_fitness = np.vstack([self.fitness, offspring_fitness])

            # 环境选择（PRDH 核心）
            self.population, self.fitness = self._environmental_selection(
                combined_population, combined_fitness
            )

        # 记录最后一代
        convergence_data.append(self._record_convergence(self.max_generations))
        end_time = time.time()

        # 评估最终结果
        result = self._evaluate_final_result(convergence_data, end_time - start_time)
        return result

    # ================================================================
    # 生成子代
    # ================================================================
    def _generate_offspring(self) -> List[np.ndarray]:
        """生成子代种群（标准交叉变异）"""
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

    # ================================================================
    # 环境选择（PRDH 核心）
    # ================================================================
    def _environmental_selection(
        self,
        combined_population: List[np.ndarray],
        combined_fitness: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        PRDH 环境选择：
        1. 非支配排序
        2. 重复解处理
        3. 问题重构 + 约束处理
        """
        # 步骤1：非支配排序
        fronts = ParetoOperations.fast_non_dominated_sort(combined_fitness)

        # 步骤2：重复解处理
        population, fitness = self._duplication_handling(
            combined_population, combined_fitness, fronts
        )

        # 重新非支配排序（处理后可能变化）
        fronts = ParetoOperations.fast_non_dominated_sort(fitness)

        # 步骤3：问题重构 + 约束处理
        return self._constraint_handling(population, fitness, fronts)

    # ================================================================
    # 重复解处理 (Algorithm 1 in paper)
    # ================================================================
    def _duplication_handling(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        处理目标空间重复解

        策略：
        - 识别目标空间重复的解（相同的 ferr 和 fratio）
        - 对每个重复解计算其与参考点的汉明距离
        - 保留非支配的重复解
        """
        n = len(population)
        if n <= 1:
            return population, fitness

        # 提取第一层非支配前沿（F1）
        if len(fronts) == 0 or len(fronts[0]) == 0:
            return population, fitness

        F1_indices = fronts[0]
        F1_fitness = fitness[F1_indices]

        # 找出所有目标空间重复的解组
        duplicates_groups = self._find_objective_duplicates(fitness)

        # 标记要删除的解
        to_remove = set()

        for dup_indices in duplicates_groups:
            if len(dup_indices) <= 1:
                continue

            # 计算每个重复解的多样性分数 (de, dr)
            diversity_scores = []

            for idx in dup_indices:
                solution = population[idx]
                ferr_val = fitness[idx, 0]
                fratio_val = fitness[idx, 1]

                # 找参考点 Re: F1中分类误差最接近的解
                Re_idx = self._find_reference_point(F1_fitness, ferr_val, objective_idx=0)
                Re = population[F1_indices[Re_idx]]

                # 找参考点 Rr: F1中特征比率最接近的解
                Rr_idx = self._find_reference_point(F1_fitness, fratio_val, objective_idx=1)
                Rr = population[F1_indices[Rr_idx]]

                # 计算汉明距离
                de = self._hamming_distance_at_positions(solution, Re, Re)
                dr = self._hamming_distance(solution, Rr)

                diversity_scores.append((idx, de, dr))

            # 使用 Pareto 支配关系筛选
            # 保留 (de, dr) 非支配的解
            non_dominated_in_group = self._get_non_dominated_by_diversity(diversity_scores)

            # 其余解标记为删除
            for idx, _, _ in diversity_scores:
                if idx not in non_dominated_in_group:
                    to_remove.add(idx)

        # 过滤掉标记的解
        if len(to_remove) > 0:
            keep_indices = [i for i in range(n) if i not in to_remove]
            if len(keep_indices) == 0:
                keep_indices = [0]  # 至少保留一个

            new_population = [population[i] for i in keep_indices]
            new_fitness = fitness[keep_indices]
            return new_population, new_fitness

        return population, fitness

    def _find_objective_duplicates(self, fitness: np.ndarray) -> List[List[int]]:
        """找出目标空间重复的解"""
        n = len(fitness)
        # 使用字典记录相同目标值的解
        objective_dict = {}

        for i in range(n):
            # 四舍五入避免浮点精度问题
            key = (round(fitness[i, 0], 10), round(fitness[i, 1], 10))
            if key not in objective_dict:
                objective_dict[key] = []
            objective_dict[key].append(i)

        # 返回重复组（大于1个解的组）
        duplicates = [indices for indices in objective_dict.values() if len(indices) > 1]
        return duplicates

    def _find_reference_point(
        self,
        F1_fitness: np.ndarray,
        target_value: float,
        objective_idx: int
    ) -> int:
        """在F1中找到目标值最接近target_value的解的索引"""
        distances = np.abs(F1_fitness[:, objective_idx] - target_value)
        return np.argmin(distances)

    def _hamming_distance_at_positions(
        self,
        x: np.ndarray,
        y: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """
        计算x和y在reference=1的位置上的汉明距离

        Args:
            x: 解1
            y: 解2
            reference: 参考解，只在其值为1的位置计算距离
        """
        positions = np.where(reference == 1)[0]
        if len(positions) == 0:
            return 0.0

        diff = np.sum(x[positions] != y[positions])
        return diff / len(positions)

    def _hamming_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算完整的汉明距离"""
        diff = np.sum(x != y)
        return diff / len(x)

    def _get_non_dominated_by_diversity(
        self,
        diversity_scores: List[Tuple[int, float, float]]
    ) -> Set[int]:
        """
        根据多样性分数 (de, dr) 筛选非支配解

        最大化 (de, dr)：距离越大越好，表示多样性越高
        """
        non_dominated = set()
        n = len(diversity_scores)

        for i in range(n):
            idx_i, de_i, dr_i = diversity_scores[i]
            is_dominated = False

            for j in range(n):
                if i == j:
                    continue
                idx_j, de_j, dr_j = diversity_scores[j]

                # j 支配 i (最大化问题)
                if (de_j >= de_i and dr_j >= dr_i) and (de_j > de_i or dr_j > dr_i):
                    is_dominated = True
                    break

            if not is_dominated:
                non_dominated.add(idx_i)

        return non_dominated

    # ================================================================
    # 问题重构和约束处理 (Algorithm 2 in paper)
    # ================================================================
    def _constraint_handling(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        问题重构 + 约束处理

        步骤：
        1. 找临界点 xt（F1中fratio最小的解）
        2. 根据 ferr <= ferr(xt) 划分可行/不可行解
        3. 优先选择可行解，不足时补充不可行解
        """
        n = len(population)

        if n <= self.pop_size:
            return population, fitness

        # 问题重构：找临界点 xt
        if len(fronts) == 0 or len(fronts[0]) == 0:
            # 如果没有非支配前沿，使用标准选择
            return self._standard_selection(population, fitness, fronts)

        F1_indices = fronts[0]
        F1_fitness = fitness[F1_indices]

        # xt = arg min fratio in F1
        xt_idx_in_F1 = np.argmin(F1_fitness[:, 1])
        xt_idx = F1_indices[xt_idx_in_F1]
        ferr_threshold = fitness[xt_idx, 0]

        # 划分可行/不可行解
        feasible_indices = []
        infeasible_indices = []

        for i in range(n):
            if fitness[i, 0] <= ferr_threshold:
                feasible_indices.append(i)
            else:
                infeasible_indices.append(i)

        # 约束处理
        selected_indices = []

        if len(feasible_indices) >= self.pop_size:
            # 情况1：可行解足够，从可行解中选择
            selected_indices = self._select_from_feasible(
                population, fitness, feasible_indices, fronts
            )
        else:
            # 情况2：可行解不足，全选可行解+补充不可行解
            selected_indices = feasible_indices.copy()

            # 从不可行解中按约束违反度排序，选最好的
            if len(infeasible_indices) > 0:
                n_needed = self.pop_size - len(selected_indices)

                # 按约束违反度排序（ferr - ferr_threshold）
                violation_scores = [
                    (idx, fitness[idx, 0] - ferr_threshold)
                    for idx in infeasible_indices
                ]
                violation_scores.sort(key=lambda x: x[1])  # 升序，违反度小的优先

                for idx, _ in violation_scores[:n_needed]:
                    selected_indices.append(idx)

        # 构建结果
        new_population = [population[i] for i in selected_indices]
        new_fitness = fitness[selected_indices]

        return new_population, new_fitness

    def _select_from_feasible(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        feasible_indices: List[int],
        fronts: List[List[int]]
    ) -> List[int]:
        """从可行解中使用非支配排序+拥挤度选择"""
        # 提取可行解
        feasible_fitness = fitness[feasible_indices]

        # 对可行解进行非支配排序
        feasible_fronts = ParetoOperations.fast_non_dominated_sort(feasible_fitness)

        # 按前沿逐层选择
        selected = []
        for front in feasible_fronts:
            if len(selected) + len(front) <= self.pop_size:
                # 整层加入
                selected.extend([feasible_indices[i] for i in front])
            else:
                # 需要截断，使用拥挤度距离
                n_needed = self.pop_size - len(selected)

                # 计算拥挤度
                front_fitness = feasible_fitness[front]
                crowding = ParetoOperations.crowding_distance(feasible_fitness, front)

                # 按拥挤度降序排序
                sorted_indices = np.argsort(-crowding)

                for i in sorted_indices[:n_needed]:
                    selected.append(feasible_indices[front[i]])
                break

        return selected[:self.pop_size]

    def _standard_selection(
        self,
        population: List[np.ndarray],
        fitness: np.ndarray,
        fronts: List[List[int]]
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """标准选择（当没有有效前沿时）"""
        selected = []

        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
            else:
                n_needed = self.pop_size - len(selected)
                crowding = ParetoOperations.crowding_distance(fitness, front)
                sorted_indices = np.argsort(-crowding)

                for i in sorted_indices[:n_needed]:
                    selected.append(front[i])
                break

        new_population = [population[i] for i in selected]
        new_fitness = fitness[selected]

        return new_population, new_fitness

    # ================================================================
    # 辅助方法
    # ================================================================
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
        return "PRDH"
