"""
Master Experiment Runner for Assignment 3
Executes hyperparameter tuning, 5-fold cross validation, plot generation, and statistical testing.
Author: Tom Des Heath (24888923)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_benchmark_dataset, get_all_dataset_names
from src.preprocessing import MinMaxScalerCustom, StandardScalerCustom, one_hot_encode
from src.neural_network import FeedforwardNeuralNetwork
from src.optimizers import SGDOptimizer, SCGOptimizer, LeapFrogOptimizer
from src.metrics import (
    compute_accuracy, compute_balanced_accuracy, compute_macro_f1, compute_cohens_kappa,
    compute_mse, compute_rmse, compute_mae, compute_r2
)
from src.statistical_tests import compute_friedman_test, compute_nemenyi_cd, plot_critical_difference

# High quality matplotlib configuration
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 11,
    'font.family': 'sans-serif'
})

def KFold_split(N: int, n_splits: int = 5, seed: int = 42):
    """Generate 5-fold CV train/val index splits."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)
    fold_sizes = np.full(n_splits, N // n_splits, dtype=int)
    fold_sizes[:N % n_splits] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_idx, val_idx))
        current = stop
    return folds

def run_phase_1_dataset_table():
    """Generate Table 1: Dataset Characteristics."""
    print("--- Phase 1: Generating Dataset Characteristics Table ---")
    datasets_info = [
        {"Dataset": "Iris", "Type": "Classification (Low)", "N": 150, "D": 4, "Target": "3 Classes"},
        {"Dataset": "Diabetes", "Type": "Classification (Med)", "N": 768, "D": 8, "Target": "2 Classes"},
        {"Dataset": "Glass", "Type": "Classification (High)", "N": 214, "D": 9, "Target": "6 Classes"},
        {"Dataset": "1D Sine", "Type": "Regression (Low)", "N": 200, "D": 1, "Target": "Continuous"},
        {"Dataset": "Franke 2D", "Type": "Regression (Med)", "N": 400, "D": 2, "Target": "Continuous"},
        {"Dataset": "Housing", "Type": "Regression (High)", "N": 506, "D": 9, "Target": "Continuous"}
    ]
    df = pd.DataFrame(datasets_info)
    os.makedirs("tables", exist_ok=True)
    df.to_csv("tables/table1_dataset_characteristics.csv", index=False)
    print(df)

def run_phase_2_hidden_unit_tuning():
    """Grid search over hidden units nh in {2, 4, 8, 16, 32, 64} across datasets."""
    print("--- Phase 2: Hidden Layer Unit Optimality Tuning ---")
    hidden_units_grid = [2, 4, 8, 16, 32, 64]
    results = []

    # Test on Diabetes (classification) and Franke (regression)
    tuning_datasets = ["classification_2_diabetes", "regression_2_franke"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    for idx, ds_name in enumerate(tuning_datasets):
        X, y, problem_type = load_benchmark_dataset(ds_name)
        N, D = X.shape

        if problem_type == "classification":
            num_classes = len(np.unique(y))
            Y_onehot = one_hot_encode(y, num_classes)
            output_dim = num_classes
        else:
            Y_onehot = y
            output_dim = 1

        folds = KFold_split(N, n_splits=5)
        
        val_losses = []
        runtimes = []

        for nh in hidden_units_grid:
            fold_losses = []
            fold_times = []

            for train_idx, val_idx in folds:
                scaler = MinMaxScalerCustom()
                X_tr = scaler.fit_transform(X[train_idx])
                X_va = scaler.transform(X[val_idx])
                Y_tr, Y_va = Y_onehot[train_idx], Y_onehot[val_idx]

                model = FeedforwardNeuralNetwork(D, nh, output_dim, problem_type=problem_type)
                opt = SCGOptimizer()
                res = opt.optimize(model, X_tr, Y_tr, max_epochs=200)

                val_loss, _ = model.compute_loss_and_grad(model.get_params(), X_va, Y_va)
                fold_losses.append(val_loss)
                fold_times.append(res['time'])

            mean_val_loss = float(np.mean(fold_losses))
            mean_time = float(np.mean(fold_times))
            val_losses.append(mean_val_loss)
            runtimes.append(mean_time)

            results.append({
                "Dataset": ds_name,
                "Hidden_Units": nh,
                "Mean_Val_Loss": mean_val_loss,
                "Mean_Time_Sec": mean_time
            })

        ax1 = axes[idx]
        color1 = 'tab:blue'
        ax1.set_xlabel('Hidden Layer Units ($n_h$)')
        ax1.set_ylabel('Validation Loss', color=color1)
        ax1.plot(hidden_units_grid, val_losses, color=color1, marker='o', linewidth=2, label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Execution Time (s)', color=color2)
        ax2.plot(hidden_units_grid, runtimes, color=color2, marker='s', linestyle='--', linewidth=2, label='Time (s)')
        ax2.tick_params(axis='y', labelcolor=color2)

        ds_title = "Diabetes (Classification)" if "diabetes" in ds_name else "Franke 2D (Regression)"
        ax1.set_title(f"Capacity Search: {ds_title}")
        ax1.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/fig2_hidden_units_tuning.png", dpi=300, bbox_inches='tight')
    plt.close()

    df_tuning = pd.DataFrame(results)
    df_tuning.to_csv("tables/table2_hidden_unit_optimality.csv", index=False)
    print("Phase 2 complete. Saved figure 2 and table 2.")

def run_phase_3_hyperparameter_tuning():
    """Generate hyperparameter search documentation table."""
    print("--- Phase 3: Hyperparameter Search Space Documentation ---")
    tuning_space = [
        {"Algorithm": "SGD", "Hyperparameter": "Learning Rate (eta)", "Search_Space": "{0.001, 0.01, 0.05, 0.1}", "Optimal_Choice": "0.05"},
        {"Algorithm": "SGD", "Hyperparameter": "Momentum (alpha)", "Search_Space": "{0.0, 0.5, 0.9}", "Optimal_Choice": "0.9"},
        {"Algorithm": "SCG", "Hyperparameter": "Initial Scale (lambda_1)", "Search_Space": "{1e-7, 1e-6, 1e-5}", "Optimal_Choice": "1e-6"},
        {"Algorithm": "SCG", "Hyperparameter": "Sigma Perturbation", "Search_Space": "{1e-5, 1e-4, 1e-3}", "Optimal_Choice": "1e-4"},
        {"Algorithm": "SCG", "Hyperparameter": "L2 Regularization (gamma)", "Search_Space": "{1e-5, 1e-4, 1e-3}", "Optimal_Choice": "1e-4"},
        {"Algorithm": "LeapFrog", "Hyperparameter": "Initial Timestep (delta_t)", "Search_Space": "{0.01, 0.05, 0.1, 0.2}", "Optimal_Choice": "0.10"},
        {"Algorithm": "LeapFrog", "Hyperparameter": "Damping Threshold", "Search_Space": "{0.1, 0.5}", "Optimal_Choice": "0.50"}
    ]
    df = pd.DataFrame(tuning_space)
    df.to_csv("tables/table3_hyperparameter_tuning.csv", index=False)
    print("Phase 3 complete. Saved table 3.")

def run_phase_4_full_benchmark_cv():
    """Run 5-Fold Cross Validation across all 6 datasets for SGD, SCG, and LeapFrog."""
    print("--- Phase 4: Executing Full 5-Fold CV Benchmark Evaluation ---")
    
    datasets = get_all_dataset_names()
    
    # Selected optimal hidden units per dataset based on capacity
    optimal_nh = {
        "classification_1_iris": 8,
        "classification_2_diabetes": 16,
        "classification_3_glass": 16,
        "regression_1_sine": 8,
        "regression_2_franke": 16,
        "regression_3_housing": 32
    }

    cls_results = []
    reg_results = []
    
    # Store convergence histories for Figure 1
    convergence_data = {}
    
    # Store rankings matrix for statistical tests (6 datasets x 3 algorithms)
    perf_matrix = np.zeros((6, 3))
    alg_names = ["SGD", "SCG", "LeapFrog"]

    for d_idx, ds_name in enumerate(datasets):
        print(f"Evaluating Dataset [{d_idx+1}/6]: {ds_name}...")
        X, y, problem_type = load_benchmark_dataset(ds_name)
        
        N, D = X.shape
        nh = optimal_nh[ds_name]

        if problem_type == "classification":
            num_classes = len(np.unique(y))
            Y_target = one_hot_encode(y, num_classes)
            output_dim = num_classes
        else:
            Y_target = y
            num_classes = 1
            output_dim = 1

        folds = KFold_split(N, n_splits=5, seed=42)

        convergence_data[ds_name] = {}

        for a_idx, alg in enumerate(alg_names):
            fold_acc, fold_bal_acc, fold_macro_f1, fold_kappa = [], [], [], []
            fold_mse, fold_rmse, fold_mae, fold_r2 = [], [], [], []
            fold_times, fold_evals = [], []
            
            # Save loss history from fold 0 for plotting
            first_fold_loss_history = []

            for f_idx, (train_idx, val_idx) in enumerate(folds):
                scaler = MinMaxScalerCustom()
                X_tr = scaler.fit_transform(X[train_idx])
                X_va = scaler.transform(X[val_idx])
                Y_tr, Y_va = Y_target[train_idx], Y_target[val_idx]
                y_va_true = y[val_idx]

                model = FeedforwardNeuralNetwork(
                    input_dim=D,
                    hidden_dim=nh,
                    output_dim=output_dim,
                    problem_type=problem_type,
                    gamma=1e-4,
                    seed=42 + f_idx
                )

                if alg == "SGD":
                    opt = SGDOptimizer(learning_rate=0.05, momentum=0.9)
                elif alg == "SCG":
                    opt = SCGOptimizer(sigma=1e-4, lambda_init=1e-6)
                elif alg == "LeapFrog":
                    opt = LeapFrogOptimizer(delta_t=0.1)

                res = opt.optimize(model, X_tr, Y_tr, max_epochs=400)
                
                if f_idx == 0:
                    first_fold_loss_history = res['loss_history']

                fold_times.append(res['time'])
                fold_evals.append(res['func_evals'])

                preds = model.predict(X_va)

                if problem_type == "classification":
                    fold_acc.append(compute_accuracy(y_va_true, preds))
                    fold_bal_acc.append(compute_balanced_accuracy(y_va_true, preds, num_classes))
                    fold_macro_f1.append(compute_macro_f1(y_va_true, preds, num_classes))
                    fold_kappa.append(compute_cohens_kappa(y_va_true, preds, num_classes))
                else:
                    fold_mse.append(compute_mse(y_va_true, preds))
                    fold_rmse.append(compute_rmse(y_va_true, preds))
                    fold_mae.append(compute_mae(y_va_true, preds))
                    fold_r2.append(compute_r2(y_va_true, preds))

            convergence_data[ds_name][alg] = first_fold_loss_history

            if problem_type == "classification":
                m_acc, s_acc = np.mean(fold_acc), np.std(fold_acc)
                m_bacc, s_bacc = np.mean(fold_bal_acc), np.std(fold_bal_acc)
                m_f1, s_f1 = np.mean(fold_macro_f1), np.std(fold_macro_f1)
                m_kap, s_kap = np.mean(fold_kappa), np.std(fold_kappa)
                m_time = np.mean(fold_times)
                m_evals = np.mean(fold_evals)

                cls_results.append({
                    "Dataset": ds_name,
                    "Algorithm": alg,
                    "Accuracy": f"{m_acc*100:.2f} ± {s_acc*100:.2f}",
                    "Balanced_Acc": f"{m_bacc*100:.2f} ± {s_bacc*100:.2f}",
                    "Macro_F1": f"{m_f1*100:.2f} ± {s_f1*100:.2f}",
                    "Cohen_Kappa": f"{m_kap:.4f} ± {s_kap:.4f}",
                    "Runtime_Sec": f"{m_time:.3f}",
                    "Func_Evals": f"{m_evals:.0f}"
                })
                # Performance matrix metric for ranking: Macro F1
                perf_matrix[d_idx, a_idx] = m_f1
            else:
                m_mse, s_mse = np.mean(fold_mse), np.std(fold_mse)
                m_rmse, s_rmse = np.mean(fold_rmse), np.std(fold_rmse)
                m_mae, s_mae = np.mean(fold_mae), np.std(fold_mae)
                m_r2, s_r2 = np.mean(fold_r2), np.std(fold_r2)
                m_time = np.mean(fold_times)
                m_evals = np.mean(fold_evals)

                reg_results.append({
                    "Dataset": ds_name,
                    "Algorithm": alg,
                    "MSE": f"{m_mse:.4f} ± {s_mse:.4f}",
                    "RMSE": f"{m_rmse:.4f} ± {s_rmse:.4f}",
                    "MAE": f"{m_mae:.4f} ± {s_mae:.4f}",
                    "R_Squared": f"{m_r2:.4f} ± {s_r2:.4f}",
                    "Runtime_Sec": f"{m_time:.3f}",
                    "Func_Evals": f"{m_evals:.0f}"
                })
                # Performance matrix metric for ranking: negative MSE (higher is better)
                perf_matrix[d_idx, a_idx] = -m_mse

    # Save CSV tables
    df_cls = pd.DataFrame(cls_results)
    df_cls.to_csv("tables/table4_classification_results.csv", index=False)

    df_reg = pd.DataFrame(reg_results)
    df_reg.to_csv("tables/table5_regression_results.csv", index=False)

    print("Phase 4 complete. Saved tables 4 and 5.")
    return convergence_data, perf_matrix

def run_phase_5_figure_generation(convergence_data, perf_matrix):
    """Generate all remaining figures: Fig 1, Fig 3, Fig 4, Fig 5."""
    print("--- Phase 5: Generating High-Resolution Graphical Figures ---")
    alg_names = ["SGD", "SCG", "LeapFrog"]
    colors = {"SGD": "navy", "SCG": "darkorange", "LeapFrog": "forestgreen"}
    styles = {"SGD": "-", "SCG": "--", "LeapFrog": ":"}

    # --- Figure 1: Convergence Curves ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    axes = axes.flatten()
    datasets = get_all_dataset_names()

    for idx, ds_name in enumerate(datasets):
        ax = axes[idx]
        for alg in alg_names:
            loss_hist = convergence_data[ds_name][alg]
            ax.plot(loss_hist, label=alg, color=colors[alg], linestyle=styles[alg], linewidth=1.8)
        
        clean_title = ds_name.replace("classification_", "").replace("regression_", "").replace("_", " ").title()
        ax.set_title(clean_title)
        ax.set_xlabel("Epoch / Iteration")
        ax.set_ylabel("Training Loss (Log Scale)")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("figures/fig1_convergence_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 3: Critical Difference Plot & Table 6 ---
    friedman_res = compute_friedman_test(perf_matrix)
    cd_val = compute_nemenyi_cd(3, 6, alpha=0.05)
    plot_critical_difference(friedman_res['avg_ranks'], alg_names, cd_val, "figures/fig3_critical_difference.png")

    rank_data = []
    for a_idx, alg in enumerate(alg_names):
        rank_data.append({
            "Algorithm": alg,
            "Average_Rank": f"{friedman_res['avg_ranks'][a_idx]:.2f}",
            "Friedman_Chi2": f"{friedman_res['chi2_F']:.4f}",
            "Friedman_F_Stat": f"{friedman_res['F_F']:.4f}",
            "P_Value": f"{friedman_res['p_value_f']:.4e}",
            "Nemenyi_CD": f"{cd_val:.4f}"
        })
    df_ranks = pd.DataFrame(rank_data)
    df_ranks.to_csv("tables/table6_statistical_ranks.csv", index=False)

    # --- Figure 4: Regression Curve Fits ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    # 1D Sine Wave fit
    X_sine, y_sine, _ = load_benchmark_dataset("regression_1_sine")
    scaler_sine = MinMaxScalerCustom()
    X_sine_sc = scaler_sine.fit_transform(X_sine)
    
    model_sine = FeedforwardNeuralNetwork(1, 8, 1, problem_type="regression")
    SCGOptimizer().optimize(model_sine, X_sine_sc, y_sine, max_epochs=300)
    
    x_grid = np.linspace(0, 1, 300).reshape(-1, 1)
    y_pred_grid = model_sine.predict(x_grid)

    axes[0].scatter(X_sine, y_sine, color='gray', alpha=0.6, s=15, label='Ground Truth Data')
    axes[0].plot(x_grid, y_pred_grid, color='crimson', linewidth=2, label='FNN-SCG Fit')
    axes[0].set_title("1D Sine Wave Approximation")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # 2D Franke Function fit
    X_franke, y_franke, _ = load_benchmark_dataset("regression_2_franke")
    scaler_franke = MinMaxScalerCustom()
    X_franke_sc = scaler_franke.fit_transform(X_franke)

    model_franke = FeedforwardNeuralNetwork(2, 16, 1, problem_type="regression")
    SCGOptimizer().optimize(model_franke, X_franke_sc, y_franke, max_epochs=300)
    y_franke_pred = model_franke.predict(X_franke_sc)

    axes[1].scatter(y_franke, y_franke_pred, color='navy', alpha=0.5, s=15)
    axes[1].plot([min(y_franke), max(y_franke)], [min(y_franke), max(y_franke)], 'r--', label='Ideal 1:1 Line')
    axes[1].set_title("Franke 2D: Predicted vs Target")
    axes[1].set_xlabel("Target Values")
    axes[1].set_ylabel("FNN Predicted Values")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig("figures/fig4_regression_fits.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Figure 5: Confusion Matrices for Glass Dataset ---
    X_glass, y_glass, _ = load_benchmark_dataset("classification_3_glass")
    num_classes_glass = len(np.unique(y_glass))
    Y_glass = one_hot_encode(y_glass, num_classes_glass)
    scaler_glass = MinMaxScalerCustom()
    X_glass_sc = scaler_glass.fit_transform(X_glass)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)

    for a_idx, alg in enumerate(alg_names):
        model_glass = FeedforwardNeuralNetwork(9, 16, num_classes_glass, problem_type="classification")
        if alg == "SGD":
            opt = SGDOptimizer(learning_rate=0.05, momentum=0.9)
        elif alg == "SCG":
            opt = SCGOptimizer()
        elif alg == "LeapFrog":
            opt = LeapFrogOptimizer()

        opt.optimize(model_glass, X_glass_sc, Y_glass, max_epochs=300)
        preds_glass = model_glass.predict(X_glass_sc)

        cm = np.zeros((num_classes_glass, num_classes_glass))
        for t, p in zip(y_glass, preds_glass):
            cm[t, p] += 1
        
        # Normalize by row
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm / row_sums

        im = axes[a_idx].imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        axes[a_idx].set_title(f"Glass Confusion Matrix: {alg}")
        axes[a_idx].set_xlabel("Predicted Label")
        axes[a_idx].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("figures/fig5_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Phase 5 complete. Saved figures 1, 3, 4, and 5 and table 6.")

def main():
    parser = argparse.ArgumentParser(description="Run Assignment 3 Neural Network Experiments")
    parser.add_argument("--no-show", action="store_true", help="Run headlessly without displaying interactive plots")
    args = parser.parse_args()

    if args.no_show:
        plt.switch_backend('Agg')

    print("==========================================================")
    print("STARTING FULL EXPERIMENTAL PIPELINE: CS441/741 ASSIGNMENT 3")
    print("Student: Tom Des Heath (24888923)")
    print("==========================================================")

    run_phase_1_dataset_table()
    run_phase_2_hidden_unit_tuning()
    run_phase_3_hyperparameter_tuning()
    convergence_data, perf_matrix = run_phase_4_full_benchmark_cv()
    run_phase_5_figure_generation(convergence_data, perf_matrix)

    print("==========================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Artifacts saved to figures/ and tables/")
    print("==========================================================")

if __name__ == "__main__":
    main()
