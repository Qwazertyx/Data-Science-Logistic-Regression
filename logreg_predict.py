#!/usr/bin/env python3
"""
logreg_predict.py

Usage:
    python logreg_predict.py dataset_test.csv weights.json

Produces houses.csv with predictions and the Index column.
"""
import sys
import json
import numpy as np
import pandas as pd
from utils import read_dataset

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def main():
    if len(sys.argv) < 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.json")
        return

    test_path = sys.argv[1]
    weights_path = sys.argv[2]

    print(f"📘 Loading weights: {weights_path}")
    with open(weights_path, "r") as f:
        info = json.load(f)

    classes = info["classes"]
    feature_names = info["feature_names"]
    means = np.array(info["means"], dtype=float)
    stds = np.array(info["stds"], dtype=float)
    thetas = np.array(info["thetas"], dtype=float)  # shape (n_classes, n_features+1)

    print(f"📘 Loading test dataset: {test_path}")
    df_test = pd.read_csv(test_path)

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
            X_raw_cols[col] = df_test[col].fillna(means[i]).astype(float).values
        else:
            # column missing in test -> use training mean
            print(f"⚠️ Warning: feature '{col}' missing in test file. Filling with training mean.")
            X_raw_cols[col] = np.full(len(df_test), means[i], dtype=float)

    X_raw = np.column_stack([X_raw_cols[c] for c in feature_names])  # shape (m, n)
    # standardize using training means/stds
    X_norm = (X_raw - means) / stds

    m = X_norm.shape[0]
    X = np.hstack([np.ones((m, 1)), X_norm])  # add intercept

    # compute probabilities for each class: (m, n_classes)
    probs = sigmoid(X.dot(thetas.T))  # each column corresponds to a class
    # choose class with highest probability
    idx_max = np.argmax(probs, axis=1)  # length m
    preds = [classes[i] for i in idx_max]

    # produce houses.csv
    out_df = pd.DataFrame({
        "Index": idx_values,
        "Hogwarts House": preds
    })
    out_df.to_csv("houses.csv", index=False)
    print("💾 Predictions saved to houses.csv")
    print("✅ Done.")

if __name__ == "__main__":
    main()
