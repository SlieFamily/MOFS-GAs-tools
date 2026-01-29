"""
评估指标模块
实现HV（超体积）等指标计算
"""
import numpy as np
from typing import Tuple


class MetricsCalculator:
    """指标计算器"""

    @staticmethod
    def hypervolume(
        pareto_front: np.ndarray,
        ref_point: Tuple[float, float] = (1.0, 1.0)
    ) -> float:
        """
        计算二维Pareto前沿的超体积（HV）

        Args:
            pareto_front: Pareto前沿的目标值 (n_solutions, 2)
            ref_point: 参考点

        Returns:
            hv: 超体积值
        """
        if len(pareto_front) == 0:
            return 0.0

        # 过滤掉超出参考点的解
        valid_mask = np.all(pareto_front < ref_point, axis=1)
        valid_front = pareto_front[valid_mask]

        if len(valid_front) == 0:
            return 0.0

        # 按第一个目标排序
        sorted_indices = np.argsort(valid_front[:, 0])
        sorted_front = valid_front[sorted_indices]

        # 计算HV（二维情况下的简单算法）
        hv = 0.0
        prev_x = 0.0

        for i, point in enumerate(sorted_front):
            x, y = point
            # 计算从prev_x到x，高度为ref_point[1]-y的矩形面积
            width = x - prev_x
            height = ref_point[1] - y
            hv += width * height
            prev_x = x

        # 加上最后一个矩形
        hv += (ref_point[0] - prev_x) * (ref_point[1] - sorted_front[-1, 1])

        return hv

    @staticmethod
    def compute_statistics(values: np.ndarray) -> dict:
        """
        计算统计量

        Args:
            values: 数值数组

        Returns:
            统计量字典
        """
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "median": np.median(values),
            "max": np.max(values)
        }
