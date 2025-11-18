#!/usr/bin/env python3
"""
logreg_check.py

Usage:
    python logreg_check.py dataset_with_labels.csv weights.json

Evaluates the logistic regression model on a labeled dataset and prints
accuracy, per-class performance, and a confusion matrix.
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from utils import read_dataset, mean

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def main():
    if len(sys.argv) < 3:
        print("Usage: python logreg_check.py dataset_with_labels.csv weights.json")
        sys.exit(1)

    data_path = sys.argv[1]
    weights_path = sys.argv[2]

    # Load weights file
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
        sys.exit(1)
    
    try:
        print(f"Loading weights: {weights_path}")
        with open(weights_path, "r") as f:
            info = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in weights file {weights_path}: {str(e)}")
        sys.exit(1)
    except (IOError, OSError) as e:
        print(f"Error: Failed to read weights file {weights_path}: {str(e)}")
        sys.exit(1)
    
    # Validate weights structure
    required_keys = ["classes", "feature_names", "means", "stds", "thetas"]
    for key in required_keys:
        if key not in info:
            print(f"Error: Missing required key '{key}' in weights file")
            sys.exit(1)
    
    try:
        classes = info["classes"]
        feature_names = info["feature_names"]
        means = np.array(info["means"], dtype=float)
        stds = np.array(info["stds"], dtype=float)
        thetas = np.array(info["thetas"], dtype=float)
    except (ValueError, TypeError) as e:
        print(f"Error: Invalid data format in weights file: {str(e)}")
        sys.exit(1)
    
    if len(classes) == 0:
        print("Error: No classes found in weights file")
        sys.exit(1)
    
    if len(feature_names) == 0:
        print("Error: No feature names found in weights file")
        sys.exit(1)
    
    if len(means) != len(feature_names) or len(stds) != len(feature_names):
        print("Error: Mismatch between number of features and means/stds in weights file")
        sys.exit(1)
    
    if len(thetas) != len(classes):
        print("Error: Mismatch between number of classes and thetas in weights file")
        sys.exit(1)

    # Load dataset
    try:
        print(f"Loading dataset: {data_path}")
        df, numeric_cols = read_dataset(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    if "Hogwarts House" not in df.columns:
        print("Error: The dataset must contain a 'Hogwarts House' column for checking.")
        sys.exit(1)
    
    if df.empty:
        print("Error: Dataset is empty")
        sys.exit(1)

    y_true = df["Hogwarts House"].astype(str).values

    # --- Preprocessing same as in training ---
    X_raw_cols = {}
    for i, col in enumerate(feature_names):
        if col in df.columns:
            try:
                X_raw_cols[col] = df[col].fillna(means[i]).astype(float).values
            except (ValueError, TypeError) as e:
                print(f"Warning: feature '{col}' contains non-numeric data. Filling with mean.")
                X_raw_cols[col] = np.full(len(df), means[i], dtype=float)
        else:
            print(f"Warning: missing feature '{col}', filling with mean.")
            X_raw_cols[col] = np.full(len(df), means[i], dtype=float)

    try:
        X_raw = np.column_stack([X_raw_cols[c] for c in feature_names])
    except Exception as e:
        print(f"Error: Failed to build feature matrix: {str(e)}")
        sys.exit(1)
    
    # Prevent division by zero
    stds_safe = np.where(stds == 0, 1.0, stds)
    X_norm = (X_raw - means) / stds_safe
    m = X_norm.shape[0]
    
    if m == 0:
        print("Error: No valid samples in dataset")
        sys.exit(1)
    
    X = np.hstack([np.ones((m, 1)), X_norm])  # add intercept

    # --- Prediction ---
    try:
        probs = sigmoid(X.dot(thetas.T))
        preds_idx = np.argmax(probs, axis=1)
        y_pred = [classes[i] for i in preds_idx]
    except Exception as e:
        print(f"Error: Failed to compute predictions: {str(e)}")
        sys.exit(1)

    # --- Metrics ---
    df_result = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    # Overall accuracy
    matches = (df_result["Actual"] == df_result["Predicted"]).tolist()
    acc = mean(matches)
    print(f"\nOverall accuracy: {acc*100:.2f}%")

    # Per-class accuracy
    print("\nAccuracy per class:")
    for cls in classes:
        mask = df_result["Actual"] == cls
        if mask.sum() == 0:
            continue
        cls_matches = (df_result.loc[mask, "Actual"] == df_result.loc[mask, "Predicted"]).tolist()
        acc_cls = mean(cls_matches)
        print(f"  {cls:15s}: {acc_cls*100:.2f}%")

    # Confusion matrix
    print("\nConfusion matrix:")
    cm = pd.crosstab(df_result["Actual"], df_result["Predicted"], rownames=["Actual"], colnames=["Predicted"], dropna=False)
    print(cm)

    # Optional: precision, recall, F1
    print("\nClassification report:")
    for cls in classes:
        tp = ((df_result["Actual"] == cls) & (df_result["Predicted"] == cls)).sum()
        fp = ((df_result["Actual"] != cls) & (df_result["Predicted"] == cls)).sum()
        fn = ((df_result["Actual"] == cls) & (df_result["Predicted"] != cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {cls:15s}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
