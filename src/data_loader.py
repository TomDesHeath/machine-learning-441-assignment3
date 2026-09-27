"""
Dataset Loader Utility
Loads and provides standardized interfaces for all 6 benchmark datasets.
Author: Tom Des Heath (24888923)
"""
import os
import pandas as pd
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")

def load_benchmark_dataset(dataset_name: str):
    """
    Load a specified benchmark dataset.
    Returns: X (features), y (target), problem_type ('classification' or 'regression')
    """
    filepath = os.path.join(DATASET_DIR, f"{dataset_name}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file {filepath} not found.")

    df = pd.read_csv(filepath)
    X = df.drop(columns=['target']).values.astype(np.float64)
    y = df['target'].values

    if "classification" in dataset_name:
        problem_type = "classification"
        y = y.astype(np.int64)
    else:
        problem_type = "regression"
        y = y.astype(np.float64).reshape(-1, 1)

    return X, y, problem_type

def get_all_dataset_names():
    return [
        "classification_1_iris",
        "classification_2_diabetes",
        "classification_3_glass",
        "regression_1_sine",
        "regression_2_franke",
        "regression_3_housing"
    ]
