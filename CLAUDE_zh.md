# CLAUDE_zh.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个多目标特征选择（MOFS）研究项目，用于在分类任务上比较算法性能。项目评估的算法需要平衡两个冲突的目标：最小化分类错误率和最小化所选特征数量。

## 项目结构

- `datasets/` - 33个基准数据集（CSV格式），涵盖中维（100-1000特征）、高维（1000-10000）和超高维（>10000）数据
- `papers/` - 9个对比算法的参考论文
- `FS_datasets.xlsx` - 数据集元信息
- `实验文档说明.md` - 详细实验协议

## 对比算法

需实现9个算法：SM-MOEA (2022)、VGS-MOEA (2023)、MFFS (2023)、MFMOEA (2024)、MOFS-MST (2024)、PRDH (2024)、DR-RPMODE (2025)、DAEA (2020)、NSGA-II (2002基准)。

## 关键实验参数

- **种群大小**：100（多种群时平均分配）
- **最大代数**：100
- **交叉概率**：Pc = 1
- **变异概率**：Pm = 1/D（D为特征数）
- **运行次数**：每个算法在每个数据集上独立运行30次
- **随机种子**：42-71

## 评估配置

- **分类器**：KNN（k=5，欧氏距离）
- **验证方法**：5折分层交叉验证
- **目标函数**：f₁ = 分类错误率（最小化），f₂ = 特征选择率 = 选择特征数/总特征数（最小化）
- **数据划分**：70%训练集 / 30%测试集（分层采样，seed=42）
- **HV参考点**：(1.0, 1.0)

## 数据预处理

1. 缺失值处理：中位数填充
2. 特征标准化：`StandardScaler`（仅在训练集上fit，同时transform训练集和测试集）
3. 随机种子：所有预处理步骤使用固定种子42

## 输出目录结构

```
results/<算法名>/<数据集名>/
├── pareto_fronts_per_run/
│   └── run_{1-30}_pareto_front.csv
├── all_runs_hv.csv
├── combined_pareto_front.csv
├── metrics_summary.csv
└── convergence.csv
```

## 输出文件格式

**Pareto前沿CSV列**：`test_error, train_error, num_features, feature_ratio, selected_features`
- `selected_features`：逗号分隔的特征索引，从0开始

**convergence.csv**：每10代记录一次，列包括 `generation, hv_value, best_error, min_features, feature_ratio`

**计时规范**：从种群初始化完成到第100代结束。不包含数据加载、预处理和特征评价指标计算（如SU、ReliefF）。

---

> 注：更新 CLAUDE.md 时需同步更新本文件。
