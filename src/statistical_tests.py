"""
Non-Parametric Statistical Evaluation: Friedman Test and Nemenyi Critical Difference (CD) Diagram.
Author: Tom Des Heath (24888923)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata, f, chi2

def compute_friedman_test(perf_matrix: np.ndarray):
    """
    Perform Friedman omnibus test across N datasets (rows) and K algorithms (columns).
    Assumes higher values in perf_matrix are better (e.g. Accuracy/MacroF1, or -MSE).
    """
    N, K = perf_matrix.shape
    # Rank each dataset (row), rank 1 = best (highest performance)
    ranks = np.zeros((N, K))
    for i in range(N):
        # Rank descending (highest value gets rank 1)
        ranks[i] = rankdata(-perf_matrix[i])

    avg_ranks = np.mean(ranks, axis=0)

    # Friedman Chi-Square statistic
    chi2_F = (12.0 * N / (K * (K + 1))) * (np.sum(avg_ranks ** 2) - (K * (K + 1) ** 2) / 4.0)

    # F_F statistic (Iman and Davenport extension)
    F_F = ((N - 1) * chi2_F) / (N * (K - 1) - chi2_F)

    p_value_chi2 = float(chi2.sf(chi2_F, K - 1))
    p_value_f = float(f.sf(F_F, K - 1, (N - 1) * (K - 1)))

    return {
        "ranks": ranks,
        "avg_ranks": avg_ranks,
        "chi2_F": float(chi2_F),
        "F_F": float(F_F),
        "p_value_chi2": p_value_chi2,
        "p_value_f": p_value_f
    }

def compute_nemenyi_cd(K: int, N: int, alpha: float = 0.05) -> float:
    """
    Compute Nemenyi Critical Difference threshold CD.
    Critical values q_alpha for alpha=0.05:
    K=2: 1.960, K=3: 2.343, K=4: 2.569, K=5: 2.728
    """
    q_alpha_table = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728
    }
    q_alpha = q_alpha_table.get(K, 2.343)
    cd = q_alpha * np.sqrt((K * (K + 1)) / (6.0 * N))
    return float(cd)

def plot_critical_difference(avg_ranks: np.ndarray, alg_names: list, cd: float, output_path: str):
    """
    Generate professional Nemenyi Critical Difference (CD) diagram.
    """
    K = len(alg_names)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)

    # Main horizontal axis
    ax.hlines(0, 1, K, colors='black', linewidth=1.5)
    for r in range(1, K + 1):
        ax.vlines(r, -0.05, 0.05, colors='black', linewidth=1)
        ax.text(r, 0.12, str(r), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.text((1 + K) / 2.0, 0.35, f"Low Rank = Superior Performance (CD = {cd:.3f})", ha='center', fontsize=11, fontweight='bold')

    # Plot algorithm ranks
    sorted_indices = np.argsort(avg_ranks)
    y_positions = [-0.25, -0.45, -0.65]

    for idx, alg_idx in enumerate(sorted_indices):
        rank_val = avg_ranks[alg_idx]
        name = alg_names[alg_idx]
        y_pos = y_positions[idx % len(y_positions)]

        ax.plot(rank_val, 0, marker='o', color='crimson', markersize=8)
        ax.plot([rank_val, rank_val], [0, y_pos], color='crimson', linestyle='--', linewidth=1)
        
        # Label position depending on rank
        ha = 'right' if rank_val > (1 + K) / 2.0 else 'left'
        x_text = rank_val - 0.05 if ha == 'right' else rank_val + 0.05
        ax.text(x_text, y_pos, f"{name} ({rank_val:.2f})", ha=ha, va='center', fontsize=10, fontweight='bold')

    # Draw CD bars connecting non-statistically different algorithms
    for i in range(K):
        for j in range(i + 1, K):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            if diff <= cd:
                r1 = avg_ranks[i]
                r2 = avg_ranks[j]
                ax.hlines(-0.85, min(r1, r2), max(r1, r2), colors='navy', linewidth=3)

    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(-1.0, 0.6)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
