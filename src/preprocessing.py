"""
Data Preprocessing Utilities
Includes Min-Max scaling, Z-score standardization, and One-Hot encoding.
Author: Tom Des Heath (24888923)
"""
import numpy as np

class StandardScalerCustom:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScalerCustom:
    """Transform features by scaling each feature to a given range [a, b]."""
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray):
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        diff = self.max_ - self.min_
        diff[diff == 0.0] = 1.0
        self.diff_ = diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        a, b = self.feature_range
        X_std = (X - self.min_) / self.diff_
        return X_std * (b - a) + a

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert categorical class labels to one-hot encoded matrix."""
    y = y.astype(int).ravel()
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot
