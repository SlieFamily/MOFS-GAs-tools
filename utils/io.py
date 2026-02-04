import os
import numpy as np
import pandas as pd
from typing import List, Dict

from algorithms.base import RunResult
from evaluation.pareto import ParetoOperations
from evaluation.metrics import MetricsCalculator


class ResultSaver:
    def __init__(self, output_dir: str):

        self.output_dir = output_dir
        self.pareto_dir = os.path.join(output_dir, "pareto_fronts_per_run")

        # 创建目录
        os.makedirs(self.pareto_dir, exist_ok=True)

    def save_run_pareto_front(
        self,
        run_id: int,
        result: RunResult,
        evaluator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_features: int
    ):
        """
        保存单次运行的Pareto前沿

        Args:
            run_id: 运行ID（1-30）
            result: 运行结果
            evaluator: 适应度评估器
            X_test: 测试集特征
            y_test: 测试集标签
            n_features: 总特征数
        """
        rows = []
        for i, (fitness, solution) in enumerate(zip(
            result.pareto_front, result.pareto_solutions
        )):
            # 计算各项指标
            test_error = fitness[0]
            feature_ratio = fitness[1]

            # 获取训练集错误率
            _, train_error = evaluator.evaluate_on_test(solution, X_test, y_test)

            # 选中的特征索引
            selected_indices = np.where(solution == 1)[0]
            num_features = len(selected_indices)
            selected_features = ",".join(map(str, selected_indices))

            rows.append({
                "test_error": round(test_error, 6),
                "train_error": round(train_error, 6),
                "num_features": num_features,
                "feature_ratio": round(feature_ratio, 6),
                "selected_features": selected_features
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.pareto_dir, f"run_{run_id}_pareto_front.csv")
        df.to_csv(filepath, index=False, encoding='utf-8')

    def save_all_runs_hv(self, hv_values: List[float], run_times: List[float]):
        """
        保存所有运行的HV值和运行时间

        Args:
            hv_values: HV值列表
            run_times: 运行时间列表
        """
        # 保存HV
        hv_df = pd.DataFrame({
            "run_id": list(range(1, len(hv_values) + 1)),
            "hv": [round(hv, 6) for hv in hv_values]
        })
        hv_filepath = os.path.join(self.output_dir, "all_runs_hv.csv")
        hv_df.to_csv(hv_filepath, index=False, encoding='utf-8')

        # 保存运行时间
        time_df = pd.DataFrame({
            "run_id": list(range(1, len(run_times) + 1)),
            "time(s)": [round(t, 6) for t in run_times]
        })
        time_filepath = os.path.join(self.output_dir, "all_runs_time.csv")
        time_df.to_csv(time_filepath, index=False, encoding='utf-8')

    def save_combined_pareto_front(
        self,
        all_results: List[RunResult],
        evaluator,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_features: int
    ):
        """
        保存综合Pareto前沿（30次运行合并后的非支配解）

        Args:
            all_results: 所有运行结果
            evaluator: 适应度评估器
            X_test: 测试集特征
            y_test: 测试集标签
            n_features: 总特征数
        """
        # 收集所有解和适应度
        all_fitness = []
        all_solutions = []

        for result in all_results:
            for fitness, solution in zip(result.pareto_front, result.pareto_solutions):
                all_fitness.append(fitness)
                all_solutions.append(solution)

        if len(all_fitness) == 0:
            return

        all_fitness = np.array(all_fitness)

        # 非支配排序提取第一前沿
        combined_fitness, combined_solutions = ParetoOperations.extract_pareto_front(
            all_fitness, all_solutions
        )

        # 保存
        rows = []
        for fitness, solution in zip(combined_fitness, combined_solutions):
            test_error = fitness[0]
            feature_ratio = fitness[1]

            _, train_error = evaluator.evaluate_on_test(solution, X_test, y_test)

            selected_indices = np.where(solution == 1)[0]
            num_features = len(selected_indices)
            selected_features = ",".join(map(str, selected_indices))

            rows.append({
                "test_error": round(test_error, 6),
                "train_error": round(train_error, 6),
                "num_features": num_features,
                "feature_ratio": round(feature_ratio, 6),
                "selected_features": selected_features
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "combined_pareto_front.csv")
        df.to_csv(filepath, index=False, encoding='utf-8')

    def save_metrics_summary(self, hv_values: List[float], run_times: List[float]):
        """
        保存指标汇总

        Args:
            hv_values: HV值列表
            run_times: 运行时间列表
        """
        hv_stats = MetricsCalculator.compute_statistics(np.array(hv_values))
        time_stats = MetricsCalculator.compute_statistics(np.array(run_times))

        rows = [
            {
                "metric": "hv",
                "mean": round(hv_stats["mean"], 6),
                "std": round(hv_stats["std"], 6),
                "min": round(hv_stats["min"], 6),
                "median": round(hv_stats["median"], 6),
                "max": round(hv_stats["max"], 6)
            },
            {
                "metric": "time(s)",
                "mean": round(time_stats["mean"], 6),
                "std": round(time_stats["std"], 6),
                "min": round(time_stats["min"], 6),
                "median": round(time_stats["median"], 6),
                "max": round(time_stats["max"], 6)
            }
        ]

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "metrics_summary.csv")
        df.to_csv(filepath, index=False, encoding='utf-8')

    def save_convergence(self, all_convergence_data: List[List[Dict]]):
        """
        保存收敛数据（取所有运行的平均值）

        Args:
            all_convergence_data: 所有运行的收敛数据
        """
        # 按代数聚合
        generations = [d["generation"] for d in all_convergence_data[0]]

        rows = []
        for i, gen in enumerate(generations):
            hv_values = [data[i]["hv_value"] for data in all_convergence_data]
            errors = [data[i]["best_error"] for data in all_convergence_data]
            features = [data[i]["min_features"] for data in all_convergence_data]
            ratios = [data[i]["feature_ratio"] for data in all_convergence_data]

            rows.append({
                "generation": gen,
                "hv_value": round(np.mean(hv_values), 6),
                "best_error": round(np.mean(errors), 6),
                "min_features": round(np.mean(features), 6),
                "feature_ratio": round(np.mean(ratios), 6)
            })

        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "convergence.csv")
        df.to_csv(filepath, index=False, encoding='utf-8')
