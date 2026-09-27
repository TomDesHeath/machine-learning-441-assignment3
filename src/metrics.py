"""
Performance Evaluation Metrics for Classification and Regression.
Author: Tom Des Heath (24888923)
"""
import numpy as np

# --- Classification Metrics ---

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate standard classification accuracy."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    return float(np.mean(y_true == y_pred))


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calculate balanced accuracy across all target classes."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    recalls = []
    for c in range(num_classes):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            recall = np.sum((y_pred == c) & mask) / np.sum(mask)
            recalls.append(recall)
    return float(np.mean(recalls)) if recalls else 0.0


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calculate Macro F1 score averaged unweighted across classes."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    f1_scores = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Calculate Cohen's Kappa coefficient measure of agreement."""
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Observed agreement
    p_o = np.mean(y_true == y_pred)
    
    # Expected agreement
    p_e = 0.0
    for c in range(num_classes):
        count_true = np.sum(y_true == c)
        count_pred = np.sum(y_pred == c)
        p_e += (count_true / n) * (count_pred / n)
        
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


# --- Regression Metrics ---

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error."""
    return float(np.mean((np.ravel(y_true) - np.ravel(y_pred)) ** 2))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred))))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R-squared)."""
    y_t = np.ravel(y_true)
    y_p = np.ravel(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))
