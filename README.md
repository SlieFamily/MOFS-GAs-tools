# MOFSA - Multi-Objective Feature Selection Algorithms

A comprehensive framework for comparing multi-objective feature selection algorithms on classification tasks. This project evaluates algorithms that balance two conflicting objectives: **minimizing classification error rate** and **minimizing the number of selected features**.

## Project Overview

This research project implements and compares 9 state-of-the-art multi-objective feature selection algorithms across 33 benchmark datasets covering medium (100-1,000 features), high (1,000-10,000 features), and ultra-high (>10,000 features) dimensional data.

### Implemented Algorithms

| Algorithm | Year | Description |
|-----------|------|-------------|
| **NSGA-II** | 2002 | Classic multi-objective optimization baseline |
| SM-MOEA | 2022 | Guiding matrix with symmetric uncertainty |
| VGS-MOEA | 2023 | Variable granularity search mechanism |
| MFFS | 2023 | Multi-form multi-task framework |
| MFMOEA | 2024 | Multi-factorial with auxiliary tasks |
| MOFS-MST | 2024 | Feature grouping and space transformation |
| PRDH | 2024 | Problem reformulation with dynamic constraints |
| DR-RPMODE | 2025 | Two-phase dimensionality reduction |
| DAEA | 2020 | Duplicate analysis for diversity |


## Installation

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn

```bash
pip install numpy pandas scikit-learn
```

## Usage

### Basic Usage

```bash
# Run with default settings (GLIOMA dataset, 30 runs, 100 generations)
python main.py --dataset GLIOMA

# Quick test with reduced parameters
python main.py --dataset musk2 --runs 1 --max_gen 10 --pop_size 20

# Full experiment
python main.py --dataset GLIOMA --algorithm NSGA-II --runs 30 --pop_size 100 --max_gen 100
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | GLIOMA | Dataset name (see available datasets below) |
| `--dataset_dir` | datasets | Dataset directory path |
| `--algorithm` | NSGA-II | Algorithm name |
| `--pop_size` | 100 | Population size |
| `--max_gen` | 100 | Maximum generations |
| `--crossover_prob` | 1.0 | Crossover probability |
| `--runs` | 30 | Number of independent runs |
| `--seed_start` | 42 | Starting random seed |
| `--k_neighbors` | 5 | K value for KNN classifier |
| `--cv_folds` | 5 | Cross-validation folds |
| `--test_size` | 0.3 | Test set ratio |
| `--results_dir` | results | Output directory |

### Available Datasets

**Medium Dimensional (100-1,000 features)**:
SCADI, LSVT, musk2, Hill_Valley_without_noise_merged, MUSK1, semeion_digits, isolet5, Madelon, drivface, acrene

**High Dimensional (1,000-10,000 features)**:
GLIOMA, PCMAC, RELATHE, DLBCL, ALLAML, Leukemia1, Leukemia2, SRBCT, Brain1, Brain2, Carcinom

**Ultra-High Dimensional (>10,000 features)**:
11Tumor, 9Tumor, CLLSUB, GLI_85, Lung, lymphoma, nci9, prostate, SMK_CAN_187, TOX_171, tumors_C, kits

## Output Format

Results are saved to `results/<algorithm>/<dataset>/`:

```
results/NSGA-II/GLIOMA/
├── pareto_fronts_per_run/
│   ├── run_1_pareto_front.csv
│   ├── run_2_pareto_front.csv
│   └── ... (30 files)
├── all_runs_hv.csv
├── all_runs_time.csv
├── combined_pareto_front.csv
├── metrics_summary.csv
└── convergence.csv
```

### Output File Formats

**Pareto Front CSV** (`run_*_pareto_front.csv`):
| Column | Description |
|--------|-------------|
| test_error | Classification error rate on test set |
| train_error | Classification error rate on training set |
| num_features | Number of selected features |
| feature_ratio | Selected features / Total features |
| selected_features | Comma-separated feature indices (0-based) |

**HV Results** (`all_runs_hv.csv`):
| Column | Description |
|--------|-------------|
| run_id | Run identifier (1-30) |
| hv | Hypervolume value |

**Metrics Summary** (`metrics_summary.csv`):
| Column | Description |
|--------|-------------|
| metric | Metric name (hv, time) |
| mean, std, min, median, max | Statistical measures |

**Convergence Data** (`convergence.csv`):
| Column | Description |
|--------|-------------|
| generation | Generation number (0, 10, 20, ..., 100) |
| hv_value | Hypervolume at this generation |
| best_error | Best error rate in population |
| min_features | Minimum feature count |
| feature_ratio | Minimum feature ratio |

## Experimental Protocol

Following the experimental document specifications:

- **Population size**: 100 (evenly split for multi-population algorithms)
- **Max generations**: 100
- **Crossover probability**: Pc = 1.0
- **Mutation probability**: Pm = 1/D (D = number of features)
- **Independent runs**: 30 per algorithm per dataset
- **Random seeds**: 42-71
- **Classifier**: KNN (k=5, Euclidean distance)
- **Validation**: 5-fold stratified cross-validation
- **Data split**: 70% train / 30% test (stratified, seed=42)
- **HV reference point**: (1.0, 1.0)



## Project Structure

```
MOFSA/
├── main.py                    # Main entry point with CLI argument parsing
├── config.py                  # Configuration management (parameters, paths)
├── datasets/                  # 33 benchmark datasets (CSV format)
├── papers/                    # Reference papers for comparison algorithms
├── results/                   # Output directory for experiment results
│
├── data/                      # Data handling module
│   ├── __init__.py
│   ├── loader.py              # CSV loader with auto header detection
│   └── preprocessor.py        # Missing value imputation, standardization, splitting
│
├── algorithms/                # Algorithm implementations
│   ├── __init__.py
│   ├── base.py                # Abstract base class and RunResult dataclass
│   └── nsga2.py               # NSGA-II implementation
│
├── evaluation/                # Evaluation module
│   ├── __init__.py
│   ├── fitness.py             # KNN-based fitness evaluation (5-fold CV)
│   ├── pareto.py              # Fast non-dominated sorting, crowding distance
│   └── metrics.py             # Hypervolume (HV) calculation
│
├── operators/                 # Genetic operators
│   ├── __init__.py
│   ├── crossover.py           # Single-point, two-point, uniform crossover
│   ├── mutation.py            # Bit-flip mutation
│   └── selection.py           # Binary tournament selection
│
└── utils/                     # Utility functions
    ├── __init__.py
    └── io.py                  # Result saving (CSV output)
```

## Architecture Design

### Modular Design Principles

1. **Separation of Concerns**: Each module handles a specific responsibility
   - `data/`: Data loading and preprocessing only
   - `algorithms/`: Algorithm logic only
   - `evaluation/`: Fitness evaluation and metrics only
   - `operators/`: Genetic operators only
   - `utils/`: I/O operations only

2. **Extensibility**: New algorithms can be added by:
   - Inheriting from `BaseAlgorithm` in `algorithms/base.py`
   - Implementing `initialize_population()` and `evolve()` methods
   - Registering in `config.py` and `main.py`

3. **Reproducibility**: All random operations use seeded generators
   - Data split: fixed seed 42
   - Algorithm runs: seeds 42-71 for 30 independent runs

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| Configuration | `config.py` | Experiment parameters, dataset/algorithm lists |
| Data Loading | `data/loader.py` | CSV parsing, auto header detection |
| Preprocessing | `data/preprocessor.py` | Imputation, standardization, train/test split |
| Base Algorithm | `algorithms/base.py` | Abstract interface, `RunResult` dataclass |
| Fitness Evaluation | `evaluation/fitness.py` | KNN classifier, 5-fold stratified CV |
| Pareto Operations | `evaluation/pareto.py` | Fast non-dominated sorting, crowding distance |
| Metrics | `evaluation/metrics.py` | Hypervolume calculation |
| Result Saving | `utils/io.py` | CSV output in required format |


## Adding New Algorithms

1. Create a new file in `algorithms/` (e.g., `algorithms/your_algorithm.py`)
2. Inherit from `BaseAlgorithm`:

```python
from algorithms.base import BaseAlgorithm, RunResult

class YourAlgorithm(BaseAlgorithm):
    def initialize_population(self):
        # Initialize your population
        pass

    def evolve(self) -> RunResult:
        # Implement evolution process
        pass
```

3. Register in `config.py` `AVAILABLE_ALGORITHMS` list
4. Add case in `main.py` `run_experiment()` function


## References

- Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
- See `papers/` directory for algorithm-specific references.
