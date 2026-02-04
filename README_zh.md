# MOFSA - 多目标特征选择算法框架

一个用于比较多目标特征选择算法在分类任务上性能的综合框架。本项目评估同时优化两个冲突目标的算法：**最小化分类错误率**和**最小化所选特征数量**。

## 项目概述

本研究项目实现并比较了9种最新的多目标特征选择算法，测试数据覆盖33个基准数据集，包括中维（100-1,000特征）、高维（1,000-10,000特征）和超高维（>10,000特征）数据。

### 已实现算法

| 算法 | 年份 | 描述 |
|------|------|------|
| **NSGA-II** | 2002 | 经典多目标优化基准算法 |
| SM-MOEA | 2022 | 基于对称不确定性的导向矩阵 |
| VGS-MOEA | 2023 | 变粒度搜索机制 |
| MFFS | 2023 | 多形态多任务框架 |
| MFMOEA | 2024 | 多因子辅助任务优化 |
| MOFS-MST | 2024 | 特征分组与空间变换 |
| PRDH | 2024 | 问题重构与动态约束处理 |
| DR-RPMODE | 2025 | 两阶段降维优化框架 |
| DAEA | 2020 | 重复解分析增强多样性 |


## 项目结构

```
MOFSA/
├── main.py                    # 主程序入口，命令行参数解析
├── config.py                  # 配置管理（参数、路径）
├── datasets/                  # 33个基准数据集（CSV格式）
├── papers/                    # 对比算法参考论文
├── results/                   # 实验结果输出目录
│
├── data/                      # 数据处理模块
│   ├── __init__.py
│   ├── loader.py              # CSV加载器（自动检测表头）
│   └── preprocessor.py        # 缺失值填充、标准化、数据划分
│
├── algorithms/                # 算法实现模块
│   ├── __init__.py
│   ├── base.py                # 抽象基类和RunResult数据类
│   └── nsga2.py               # NSGA-II算法实现
│
├── evaluation/                # 评估模块
│   ├── __init__.py
│   ├── fitness.py             # 基于KNN的适应度评估（5折CV）
│   ├── pareto.py              # 快速非支配排序、拥挤度距离
│   └── metrics.py             # 超体积（HV）计算
│
├── operators/                 # 遗传算子模块
│   ├── __init__.py
│   ├── crossover.py           # 单点、两点、均匀交叉
│   ├── mutation.py            # 位翻转变异
│   └── selection.py           # 二元锦标赛选择
│
└── utils/                     # 工具模块
    ├── __init__.py
    └── io.py                  # 结果保存（CSV输出）
```

## 安装

### 依赖

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn

```bash
pip install numpy pandas scikit-learn
```

## 使用方法

### 基本用法

```bash
# 使用默认配置运行（GLIOMA数据集，30次运行，100代）
python main.py --dataset GLIOMA

# 快速测试（减少参数）
python main.py --dataset musk2 --runs 1 --max_gen 10 --pop_size 20

# 完整实验
python main.py --dataset GLIOMA --algorithm NSGA-II --runs 30 --pop_size 100 --max_gen 100
```

### 命令行参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset` | GLIOMA | 数据集名称（见下方可用数据集） |
| `--dataset_dir` | datasets | 数据集目录路径 |
| `--algorithm` | NSGA-II | 算法名称 |
| `--pop_size` | 100 | 种群大小 |
| `--max_gen` | 100 | 最大代数 |
| `--crossover_prob` | 1.0 | 交叉概率 |
| `--runs` | 30 | 独立运行次数 |
| `--seed_start` | 42 | 起始随机种子 |
| `--k_neighbors` | 5 | KNN的k值 |
| `--cv_folds` | 5 | 交叉验证折数 |
| `--test_size` | 0.3 | 测试集比例 |
| `--results_dir` | results | 输出目录 |

### 可用数据集

**中维（100-1,000特征）**：
SCADI, LSVT, musk2, Hill_Valley_without_noise_merged, MUSK1, semeion_digits, isolet5, Madelon, drivface, acrene

**高维（1,000-10,000特征）**：
GLIOMA, PCMAC, RELATHE, DLBCL, ALLAML, Leukemia1, Leukemia2, SRBCT, Brain1, Brain2, Carcinom

**超高维（>10,000特征）**：
11Tumor, 9Tumor, CLLSUB, GLI_85, Lung, lymphoma, nci9, prostate, SMK_CAN_187, TOX_171, tumors_C, kits

## 输出格式

结果保存到 `results/<算法>/<数据集>/`：

```
results/NSGA-II/GLIOMA/
├── pareto_fronts_per_run/
│   ├── run_1_pareto_front.csv
│   ├── run_2_pareto_front.csv
│   └── ...（30个文件）
├── all_runs_hv.csv
├── all_runs_time.csv
├── combined_pareto_front.csv
├── metrics_summary.csv
└── convergence.csv
```

### 输出文件格式

**Pareto前沿 CSV**（`run_*_pareto_front.csv`）：
| 列名 | 描述 |
|------|------|
| test_error | 测试集分类错误率 |
| train_error | 训练集分类错误率 |
| num_features | 选中特征数量 |
| feature_ratio | 选中特征数 / 总特征数 |
| selected_features | 逗号分隔的特征索引（从0开始） |

**HV结果**（`all_runs_hv.csv`）：
| 列名 | 描述 |
|------|------|
| run_id | 运行编号（1-30） |
| hv | 超体积值 |

**指标汇总**（`metrics_summary.csv`）：
| 列名 | 描述 |
|------|------|
| metric | 指标名称（hv, time） |
| mean, std, min, median, max | 统计量 |

**收敛数据**（`convergence.csv`）：
| 列名 | 描述 |
|------|------|
| generation | 代数（0, 10, 20, ..., 100） |
| hv_value | 该代的超体积值 |
| best_error | 种群中最佳错误率 |
| min_features | 最小特征数 |
| feature_ratio | 最小特征比例 |

## 实验协议

按照实验文档规范：

- **种群大小**：100（多种群算法平均分配）
- **最大代数**：100
- **交叉概率**：Pc = 1.0
- **变异概率**：Pm = 1/D（D为特征数）
- **独立运行**：每个算法每个数据集30次
- **随机种子**：42-71
- **分类器**：KNN（k=5，欧氏距离）
- **验证方法**：5折分层交叉验证
- **数据划分**：70%训练 / 30%测试（分层，seed=42）
- **HV参考点**：(1.0, 1.0)


## 架构设计原则

### 模块化设计

1. **职责分离**：每个模块处理特定职责
   - `data/`：仅负责数据加载和预处理
   - `algorithms/`：仅负责算法逻辑
   - `evaluation/`：仅负责适应度评估和指标计算
   - `operators/`：仅负责遗传算子
   - `utils/`：仅负责I/O操作

2. **可扩展性**：添加新算法只需：
   - 继承 `algorithms/base.py` 中的 `BaseAlgorithm`
   - 实现 `initialize_population()` 和 `evolve()` 方法
   - 在 `config.py` 和 `main.py` 中注册

3. **可重现性**：所有随机操作使用种子控制
   - 数据划分：固定种子 42
   - 算法运行：种子 42-71（30次独立运行）

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| 配置管理 | `config.py` | 实验参数、数据集/算法列表 |
| 数据加载 | `data/loader.py` | CSV解析、自动表头检测 |
| 数据预处理 | `data/preprocessor.py` | 缺失值填充、标准化、训练/测试划分 |
| 算法基类 | `algorithms/base.py` | 抽象接口、`RunResult` 数据类 |
| 适应度评估 | `evaluation/fitness.py` | KNN分类器、5折分层交叉验证 |
| Pareto操作 | `evaluation/pareto.py` | 快速非支配排序、拥挤度距离 |
| 指标计算 | `evaluation/metrics.py` | 超体积计算 |
| 结果保存 | `utils/io.py` | 按规定格式输出CSV |

## 添加新算法

1. 在 `algorithms/` 中创建新文件（如 `algorithms/your_algorithm.py`）
2. 继承 `BaseAlgorithm`：

```python
from algorithms.base import BaseAlgorithm, RunResult

class YourAlgorithm(BaseAlgorithm):
    def initialize_population(self):
        # 初始化种群
        pass

    def evolve(self) -> RunResult:
        # 实现进化过程
        pass
```

3. 在 `config.py` 的 `AVAILABLE_ALGORITHMS` 列表中注册
4. 在 `main.py` 的 `run_experiment()` 函数中添加分支

## 参考文献

- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
- 更多算法参考论文见 `papers/` 目录。
