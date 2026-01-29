import os
import sys
import argparse
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig, AVAILABLE_DATASETS, AVAILABLE_ALGORITHMS
from data.loader import DataLoader
from data.preprocessor import DataPreprocessor
from algorithms.nsga2 import NSGA2
from evaluation.fitness import FitnessEvaluator
from utils.io import ResultSaver


def parse_args():
    parser = argparse.ArgumentParser(
        description="多目标特征选择实验框架",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument(
        "--dataset", type=str, default="GLIOMA",
        choices=AVAILABLE_DATASETS,
        help="数据集名称"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="datasets",
        help="数据集目录"
    )

    # 算法参数
    parser.add_argument(
        "--algorithm", type=str, default="NSGA-II",
        choices=AVAILABLE_ALGORITHMS,
        help="算法名称"
    )
    parser.add_argument(
        "--pop_size", type=int, default=100,
        help="种群大小"
    )
    parser.add_argument(
        "--max_gen", type=int, default=100,
        help="最大代数"
    )
    parser.add_argument(
        "--crossover_prob", type=float, default=1.0,
        help="交叉概率"
    )

    # 运行参数
    parser.add_argument(
        "--runs", type=int, default=30,
        help="独立运行次数"
    )
    parser.add_argument(
        "--seed_start", type=int, default=42,
        help="起始随机种子"
    )

    # 评估参数
    parser.add_argument(
        "--k_neighbors", type=int, default=5,
        help="KNN的k值"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="交叉验证折数"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3,
        help="测试集比例"
    )

    # 输出参数
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="结果输出目录"
    )

    return parser.parse_args()


def run_experiment(config: ExperimentConfig):
    """
    运行实验

    Args:
        config: 实验配置
    """
    print("=" * 60)
    print(f"实验配置:")
    print(f"  数据集: {config.dataset_name}")
    print(f"  算法: {config.algorithm}")
    print(f"  种群大小: {config.pop_size}")
    print(f"  最大代数: {config.max_generations}")
    print(f"  运行次数: {config.num_runs}")
    print(f"  随机种子: {config.seed_start} - {config.seed_start + config.num_runs - 1}")
    print("=" * 60)

    # 加载数据
    print(f"\n[1/4] 加载数据集: {config.dataset_path}")
    X, y = DataLoader.load(config.dataset_path)
    print(f"  样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

    # 预处理数据（使用固定seed=42划分）
    print("\n[2/4] 数据预处理...")
    preprocessor = DataPreprocessor(
        test_size=config.test_size,
        random_state=42  # 固定划分种子
    )
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(X, y)
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")

    n_features = X_train.shape[1]
    mutation_prob = config.get_mutation_prob(n_features)
    print(f"  变异概率: {mutation_prob:.6f} (1/{n_features})")

    # 创建结果保存器
    saver = ResultSaver(config.output_dir)
    print(f"\n[3/4] 结果将保存到: {config.output_dir}")

    # 运行多次实验
    print(f"\n[4/4] 开始运行 {config.num_runs} 次独立实验...")
    all_results = []
    hv_values = []
    run_times = []
    all_convergence_data = []

    for run_idx, seed in enumerate(config.seeds):
        run_id = run_idx + 1
        print(f"\n  运行 {run_id}/{config.num_runs} (seed={seed})...", end=" ", flush=True)

        # 创建算法实例
        if config.algorithm == "NSGA-II":
            algorithm = NSGA2(
                pop_size=config.pop_size,
                max_generations=config.max_generations,
                crossover_prob=config.crossover_prob,
                mutation_prob=mutation_prob,
                seed=seed,
                k_neighbors=config.k_neighbors,
                cv_folds=config.cv_folds,
                convergence_interval=config.convergence_interval,
                hv_ref_point=config.hv_ref_point
            )
        else:
            raise ValueError(f"未实现的算法: {config.algorithm}")

        # 设置数据
        algorithm.set_data(X_train, y_train, X_test, y_test)

        # 运行进化
        result = algorithm.evolve()

        # 记录结果
        all_results.append(result)
        hv_values.append(result.hv_value)
        run_times.append(result.run_time)
        all_convergence_data.append(result.convergence_data)

        print(f"HV={result.hv_value:.4f}, Time={result.run_time:.2f}s")

        # 保存单次运行的Pareto前沿
        evaluator = FitnessEvaluator(
            X_train, y_train,
            k_neighbors=config.k_neighbors,
            cv_folds=config.cv_folds,
            seed=seed
        )
        saver.save_run_pareto_front(
            run_id, result, evaluator, X_test, y_test, n_features
        )

    # 保存汇总结果
    print("\n保存汇总结果...")
    saver.save_all_runs_hv(hv_values, run_times)
    saver.save_metrics_summary(hv_values, run_times)
    saver.save_convergence(all_convergence_data)

    # 保存综合Pareto前沿
    evaluator = FitnessEvaluator(
        X_train, y_train,
        k_neighbors=config.k_neighbors,
        cv_folds=config.cv_folds,
        seed=42
    )
    saver.save_combined_pareto_front(
        all_results, evaluator, X_test, y_test, n_features
    )

    # 打印汇总统计
    print("\n" + "=" * 60)
    print("实验完成!")
    print(f"  HV: {np.mean(hv_values):.4f} ± {np.std(hv_values):.4f}")
    print(f"  运行时间: {np.mean(run_times):.2f} ± {np.std(run_times):.2f} 秒")
    print(f"  结果已保存到: {config.output_dir}")
    print("=" * 60)


def main():
    """主函数"""
    args = parse_args()

    # 构建配置
    config = ExperimentConfig(
        dataset_name=args.dataset,
        dataset_dir=args.dataset_dir,
        algorithm=args.algorithm,
        pop_size=args.pop_size,
        max_generations=args.max_gen,
        crossover_prob=args.crossover_prob,
        num_runs=args.runs,
        seed_start=args.seed_start,
        k_neighbors=args.k_neighbors,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        results_dir=args.results_dir
    )

    # 检查数据集是否存在
    if not os.path.exists(config.dataset_path):
        print(f"错误: 数据集不存在: {config.dataset_path}")
        sys.exit(1)

    # 运行实验
    run_experiment(config)


if __name__ == "__main__":
    main()
