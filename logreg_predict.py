#!/usr/bin/env python3
"""
logreg_predict.py

Usage:
    python logreg_predict.py dataset_test.csv weights.json

Produces houses.csv with predictions and the Index column.
"""
import sys
import json
import os
import numpy as np
import pandas as pd
from utils import read_dataset

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def main():
    if len(sys.argv) < 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        sys.exit(1)

    test_path = sys.argv[1]
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

    # Load test dataset
    try:
        print(f"Loading test dataset: {test_path}")
        df_test = pd.read_csv(test_path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error: Failed to load test dataset {test_path}: {str(e)}")
        sys.exit(1)
    
    if df_test.empty:
        print("Error: Test dataset is empty")
        sys.exit(1)

    # preserve index column if available (case insensitive 'Index')
    index_col = None
    if 'Index' in df_test.columns:
        index_col = 'Index'
    else:
        # try case-insensitive find
        for c in df_test.columns:
            if c.lower() == 'index':
                index_col = c
                break

    if index_col is None:
        # fallback to row numbers
        idx_values = list(range(len(df_test)))
    else:
        idx_values = df_test[index_col].tolist()

    # build numeric matrix X_raw with same feature order
    # If some feature is missing in test, fill with training mean
    X_raw_cols = {}
    for i, col in enumerate(feature_names):
        if col in df_test.columns:
            try:
                X_raw_cols[col] = df_test[col].fillna(means[i]).astype(float).values
            except (ValueError, TypeError) as e:
                print(f"Warning: feature '{col}' contains non-numeric data. Filling with training mean.")
                X_raw_cols[col] = np.full(len(df_test), means[i], dtype=float)
        else:
            # column missing in test -> use training mean
            print(f"Warning: feature '{col}' missing in test file. Filling with training mean.")
            X_raw_cols[col] = np.full(len(df_test), means[i], dtype=float)

    try:
        X_raw = np.column_stack([X_raw_cols[c] for c in feature_names])  # shape (m, n)
    except Exception as e:
        print(f"Error: Failed to build feature matrix: {str(e)}")
        sys.exit(1)
    
    # standardize using training means/stds
    # Prevent division by zero
    stds_safe = np.where(stds == 0, 1.0, stds)
    X_norm = (X_raw - means) / stds_safe

    m = X_norm.shape[0]
    if m == 0:
        print("Error: No valid samples in test dataset")
        sys.exit(1)
    
    X = np.hstack([np.ones((m, 1)), X_norm])  # add intercept

    # compute probabilities for each class: (m, n_classes)
    try:
        probs = sigmoid(X.dot(thetas.T))  # each column corresponds to a class
    except Exception as e:
        print(f"Error: Failed to compute predictions: {str(e)}")
        sys.exit(1)
    
    # choose class with highest probability
    idx_max = np.argmax(probs, axis=1)  # length m
    preds = [classes[i] for i in idx_max]

    # produce houses.csv
    try:
        out_df = pd.DataFrame({
            "Index": idx_values,
            "Hogwarts House": preds
        })
        out_df.to_csv("houses.csv", index=False)
        print("Predictions saved to houses.csv")
    except (IOError, OSError) as e:
        print(f"Error: Failed to save predictions to houses.csv: {str(e)}")
        sys.exit(1)
    
    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
