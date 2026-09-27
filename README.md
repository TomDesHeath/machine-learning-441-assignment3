# CS441 / RW441 Assignment 3: Feedforward Neural Network Training Algorithms

**Student Name**: Tom Des Heath  
**Student ID**: 24888923  
**Module**: Machine Learning 441 (CS441 / RW441)  
**Report Title**: *Comparative Empirical Evaluation of Feedforward Neural Network Training Algorithms: Stochastic Gradient Descent, Scaled Conjugate Gradient, and LeapFrog Optimization* 

---

## 1. Project Overview

This repository contains a pure Python, from-scratch implementation and comparative evaluation of three feedforward neural network (FNN) training algorithms:
1. **Stochastic Gradient Descent (SGD)** with Momentum.
2. **Scaled Conjugate Gradient (SCG)** (Møller 1993).
3. **LeapFrog Optimization** (Snyman 1982, 1983).

The evaluation is conducted across 6 benchmark problems (3 classification and 3 regression) using 5-fold cross-validation and non-parametric statistical evaluation (Friedman test and Nemenyi Critical Difference plot).

---

## 2. Directory Layout

```
assignment_3/
├── README.md                           # System requirements and execution guide
├── run_all.py                          # Master entry script (--no-show supported)
├── run_experiments.py                  # Full experiment pipeline execution
├── 24888923RW441assignment3.tex        # Primary LaTeX report source file
├── 24888923RW441assignment3.pdf        # Compiled IEEE PDF report
├── src/                                # Pure Python algorithmic code from scratch
│   ├── __init__.py
│   ├── neural_network.py               # Feedforward Neural Network architecture & backpropagation
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── sgd.py                      # Stochastic Gradient Descent with Momentum
│   │   ├── scg.py                      # Scaled Conjugate Gradient (Møller 1993)
│   │   └── leapfrog.py                 # LeapFrog Optimizer (Snyman 1982, 1983)
│   ├── data_loader.py                  # Benchmark datasets loader
│   ├── preprocessing.py               # Min-Max scaling, z-score, one-hot encoding
│   ├── metrics.py                      # Classification & regression performance metrics
│   └── statistical_tests.py            # Friedman test & Nemenyi CD calculation
├── datasets/                           # Benchmark dataset CSV files
│   ├── classification_1_iris.csv
│   ├── classification_2_diabetes.csv
│   ├── classification_3_glass.csv
│   ├── regression_1_sine.csv
│   ├── regression_2_franke.csv
│   └── regression_3_housing.csv
├── figures/                            # High-resolution output plots (300 DPI)
│   ├── fig1_convergence_curves.png
│   ├── fig2_hidden_units_tuning.png
│   ├── fig3_critical_difference.png
│   ├── fig4_regression_fits.png
│   └── fig5_confusion_matrices.png
└── tables/                             # Formatted metrics summaries (CSV)
    ├── table1_dataset_characteristics.csv
    ├── table2_hidden_unit_optimality.csv
    ├── table3_hyperparameter_tuning.csv
    ├── table4_classification_results.csv
    ├── table5_regression_results.csv
    └── table6_statistical_ranks.csv
```

---

## 3. System Requirements & Dependencies

- **Python**: Version 3.8 or higher.
- **Python Libraries**: `numpy`, `scipy`, `matplotlib`, `pandas`, `scikit-learn`.
- **LaTeX Compiler**: `pdflatex` (TeX Live 2026 or equivalent with `IEEEtran.cls`).

---

## 4. Replication Instructions

To reproduce all experiments, generate figures, output tables, and compile the LaTeX report:

```bash
# 1. Run the entire experimental pipeline headlessly
python3 run_all.py --no-show

# 2. Compile the primary LaTeX PDF report
pdflatex -interaction=nonstopmode 24888923RW441assignment3.tex
```

All generated plots will be saved to `figures/` and metrics tables to `tables/`.

---

## 5. Summary of Main Empirical Results

1. **SCG Superiority**: Scaled Conjugate Gradient achieved the lowest overall average rank ($1.25$) across all six benchmark datasets, outperforming LeapFrog ($1.92$) and SGD ($2.83$).
2. **Statistical Significance**: The omnibus Friedman test ($F_F = 8.5849, p = 0.0068$) confirmed statistically significant performance differences at $\alpha = 0.05$. The rank difference between SCG and SGD ($1.58$) exceeded the Nemenyi critical difference threshold ($CD = 1.3527$).
3. **Line-Search-Free Convergence**: SCG achieved convergence to low error thresholds in significantly fewer function evaluations compared to first-order SGD, while LeapFrog navigated ill-conditioned error landscapes effectively using dynamic velocity resets.
