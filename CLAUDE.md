# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-objective feature selection (MOFS) research project for comparing algorithms on classification tasks. The project evaluates algorithms that balance two conflicting objectives: minimizing classification error rate and minimizing the number of selected features.

## Project Structure

- `datasets/` - 33 benchmark datasets (CSV format) covering medium (100-1000 features), high (1000-10000), and ultra-high (>10000) dimensional data
- `papers/` - Reference papers for the 9 comparison algorithms
- `FS_datasets.xlsx` - Dataset metadata
- `实验文档说明.md` - Detailed experiment protocol (Chinese)

## Comparison Algorithms

9 algorithms to implement: SM-MOEA (2022), VGS-MOEA (2023), MFFS (2023), MFMOEA (2024), MOFS-MST (2024), PRDH (2024), DR-RPMODE (2025), DAEA (2020), NSGA-II (2002 baseline).

## Key Experimental Parameters

- **Population size**: 100 (split evenly if multi-population)
- **Max generations**: 100
- **Crossover probability**: Pc = 1
- **Mutation probability**: Pm = 1/D (D = feature count)
- **Runs**: 30 independent runs per algorithm per dataset
- **Random seeds**: 42-71

## Evaluation Configuration

- **Classifier**: KNN (k=5, Euclidean distance)
- **Validation**: 5-fold stratified cross-validation
- **Objectives**: f₁ = classification error rate (minimize), f₂ = feature_ratio = selected/total (minimize)
- **Data split**: 70% train / 30% test (stratified, seed=42)
- **HV reference point**: (1.0, 1.0)

## Data Preprocessing

1. Missing values: median imputation
2. Standardization: `StandardScaler` (fit on train only, transform both)
3. Random seed: 42 for all preprocessing

## Output Structure

```
results/<algorithm>/<dataset>/
├── pareto_fronts_per_run/
│   └── run_{1-30}_pareto_front.csv
├── all_runs_hv.csv
├── combined_pareto_front.csv
├── metrics_summary.csv
└── convergence.csv
```

## Output File Formats

**Pareto front CSV columns**: `test_error, train_error, num_features, feature_ratio, selected_features`
- `selected_features`: comma-separated indices starting from 0

**convergence.csv**: Record every 10 generations with columns `generation, hv_value, best_error, min_features, feature_ratio`

**Timing**: Measure from population initialization complete to generation 100 end. Exclude data loading, preprocessing, and feature evaluation metric computation (SU, ReliefF).

---

> Note: When updating this file, also update CLAUDE_zh.md (Chinese version) accordingly.
