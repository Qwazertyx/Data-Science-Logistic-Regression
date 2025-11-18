#!/usr/bin/env python3
"""
logreg_train.py

Usage:
    python logreg_train.py datasets/dataset_train.csv weights.json

Trains one-vs-rest logistic regression models using gradient descent and
saves weights + preprocessing info to a JSON file (weights.json).
"""
import sys
import json
import math
import numpy as np
import pandas as pd
from utils import read_dataset, mean, std

# --- Helpers -----------------------------------------------------------------
def sigmoid(z):
    # Numerically stable sigmoid
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def compute_cost_and_grad(theta, X, y):
    m = X.shape[0]
    h = sigmoid(X.dot(theta))
    # cost (not used for update directly here, but useful for logging)
    eps = 1e-15
    cost = - (1.0/m) * (y * np.log(h+eps) + (1-y) * np.log(1-h+eps)).sum()
    grad = (1.0/m) * X.T.dot(h - y)
    return cost, grad

def gradient_descent(X, y, alpha=0.1, num_iter=2000, verbose=False):
    n = X.shape[1]
    theta = np.zeros(n)  # initialize
    last_cost = None
    for it in range(num_iter):
        cost, grad = compute_cost_and_grad(theta, X, y)
        theta -= alpha * grad
        last_cost = cost
        if verbose and (it % (num_iter // 10 + 1) == 0):
            print(f" iter {it}/{num_iter} cost={cost:.6f}")
    return theta, last_cost

# --- Main --------------------------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python logreg_train.py dataset_train.csv weights.json")
        sys.exit(1)

    train_path = sys.argv[1]
    out_path = sys.argv[2]

    try:
        print(f"Loading dataset: {train_path}")
        df, numeric_cols = read_dataset(train_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Filter out index-like columns from numeric features
    numeric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in ['index', 'id'])]

    if len(numeric_cols) == 0:
        print("Error: No numeric feature columns found after filtering (excluding index/id columns)")
        sys.exit(1)

    # Ensure Hogwarts House exists
    if 'Hogwarts House' not in df.columns:
        print("Error: The training CSV must contain 'Hogwarts House' column")
        sys.exit(1)

    # Keep only rows with at least one numeric value? We'll drop rows where all numeric are NaN
    df_num = df[numeric_cols].copy()
    df_y = df['Hogwarts House'].copy()
    
    if len(df_num) == 0:
        print("Error: Dataset has no rows")
        sys.exit(1)

    # Impute missing values with column mean (train mean)
    means = {}
    for col in numeric_cols:
        col_values = df_num[col].tolist()
        col_mean = mean(col_values)
        if pd.isna(col_mean):
            means[col] = 0.0
        else:
            means[col] = col_mean
        df_num[col] = df_num[col].fillna(means[col])

    # Standardize (z-score) features
    stds = {}
    for col in numeric_cols:
        col_values = df_num[col].tolist()
        col_std = std(col_values)
        if col_std == 0 or math.isnan(col_std):
            stds[col] = 1.0
        else:
            stds[col] = col_std

    try:
        X_raw = df_num[numeric_cols].values.astype(float)
    except (ValueError, TypeError) as e:
        print(f"Error: Failed to convert numeric columns to float: {str(e)}")
        sys.exit(1)
    
    # normalize
    for i, col in enumerate(numeric_cols):
        if stds[col] == 0:
            X_raw[:, i] = 0.0
        else:
            X_raw[:, i] = (X_raw[:, i] - means[col]) / stds[col]

    m = X_raw.shape[0]
    if m == 0:
        print("Error: No valid training samples after preprocessing")
        sys.exit(1)
    
    # add intercept column
    X = np.hstack([np.ones((m, 1)), X_raw])  # shape (m, n+1)

    # classes
    classes = sorted(df_y.dropna().unique().astype(str).tolist())
    
    if len(classes) == 0:
        print("Error: No classes found in 'Hogwarts House' column")
        sys.exit(1)

    print(f"Found classes: {classes}")
    thetas = []
    training_info = {
        "classes": classes,
        "feature_names": numeric_cols,
        "means": [means[c] for c in numeric_cols],
        "stds": [stds[c] for c in numeric_cols],
        "thetas": []
    }

    # Training hyperparameters (to customize)
    alpha = 0.1
    num_iter = 4000
    verbose = True

    for cls in classes:
        print(f"\nTraining classifier for class: {cls}")
        # binary labels: 1 for this class, 0 otherwise
        y = (df_y.astype(str) == cls).astype(float).values  # shape (m,)
        theta, final_cost = gradient_descent(X, y, alpha=alpha, num_iter=num_iter, verbose=verbose)
        print(f" Done: final cost={final_cost:.6f}")
        training_info["thetas"].append(theta.tolist())

    # Save to JSON
    try:
        print(f"\nSaving weights & preprocessing to: {out_path}")
        with open(out_path, "w") as f:
            json.dump(training_info, f)
    except (IOError, OSError) as e:
        print(f"Error: Failed to save weights to {out_path}: {str(e)}")
        sys.exit(1)

    print("Training complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
