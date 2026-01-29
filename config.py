"""
配置管理模块
包含实验参数、路径配置和默认设置
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """实验配置"""
    # 数据集配置
    dataset_name: str = "GLIOMA"
    dataset_dir: str = "datasets"

    # 算法参数
    algorithm: str = "NSGA-II"
    pop_size: int = 100
    max_generations: int = 100
    crossover_prob: float = 1.0
    # mutation_prob 将根据特征数动态计算: 1/D

    # 运行配置
    num_runs: int = 30
    seed_start: int = 42

    # 评估配置
    k_neighbors: int = 5
    cv_folds: int = 5
    test_size: float = 0.3

    # HV参考点
    hv_ref_point: tuple = (1.0, 1.0)

    # 收敛记录间隔
    convergence_interval: int = 10

    # 输出配置
    results_dir: str = "results"

    @property
    def seeds(self) -> List[int]:
        """返回所有运行的随机种子列表"""
        return list(range(self.seed_start, self.seed_start + self.num_runs))

    @property
    def dataset_path(self) -> str:
        """返回数据集完整路径"""
        return os.path.join(self.dataset_dir, f"{self.dataset_name}.csv")

    @property
    def output_dir(self) -> str:
        """返回算法输出目录"""
        return os.path.join(self.results_dir, self.algorithm, self.dataset_name)

    def get_mutation_prob(self, num_features: int) -> float:
        """根据特征数计算变异概率"""
        return 1.0 / num_features if num_features > 0 else 0.01


# 可用数据集列表
AVAILABLE_DATASETS = [
    # 中维 (100-1000)
    "SCADI", "LSVT", "musk2", "Hill_Valley_without_noise_merged",
    "MUSK1", "semeion_digits", "isolet5", "Madelon", "drivface", "acrene",
    # 高维 (1000-10000)
    "GLIOMA", "PCMAC", "RELATHE", "DLBCL", "ALLAML", "Leukemia1",
    "Leukemia2", "SRBCT", "Brain1", "Brain2", "Carcinom",
    # 超高维 (>10000)
    "11Tumor", "9Tumor", "CLLSUB", "GLI_85", "Lung", "lymphoma",
    "nci9", "prostate", "SMK_CAN_187", "TOX_171", "tumors_C", "kits"
]

# 算法列表（便于后续扩展）
AVAILABLE_ALGORITHMS = [
    "NSGA-II",
    "SM-MOEA",
    "VGS-MOEA",
    "MFFS",
    "MFMOEA",
    "MOFS-MST",
    "PRDH",
    "DR-RPMODE",
    "DAEA"
]
