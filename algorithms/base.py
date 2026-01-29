"""
算法基类模块
定义多目标特征选择算法的接口
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class RunResult:
    """单次运行结果"""
    pareto_front: np.ndarray  # Pareto前沿的目标值
    pareto_solutions: List[np.ndarray]  # Pareto前沿的解
    hv_value: float  # HV值
    convergence_data: List[Dict]  # 收敛数据
    run_time: float  # 运行时间


class BaseAlgorithm(ABC):
    """多目标特征选择算法基类"""

    def __init__(
        self,
        pop_size: int = 100,
        max_generations: int = 100,
        crossover_prob: float = 1.0,
        mutation_prob: float = None,  # 默认为1/D
        seed: int = 42
    ):
        """
        初始化算法

        Args:
            pop_size: 种群大小
            max_generations: 最大代数
            crossover_prob: 交叉概率
            mutation_prob: 变异概率（None表示使用1/D）
            seed: 随机种子
        """
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # 数据相关
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.n_features = None

        # 算法状态
        self.population = None
        self.fitness = None
        self.generation = 0

    def set_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        设置数据

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = X_train.shape[1]

        # 如果未指定变异概率，使用1/D
        if self.mutation_prob is None:
            self.mutation_prob = 1.0 / self.n_features

    @abstractmethod
    def initialize_population(self) -> List[np.ndarray]:
        """初始化种群"""
        pass

    @abstractmethod
    def evolve(self) -> RunResult:
        """执行进化过程"""
        pass

    def get_algorithm_name(self) -> str:
        """返回算法名称"""
        return self.__class__.__name__
