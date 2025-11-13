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
import numpy as np
import pandas as pd
from utils import read_dataset

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def main():
    if len(sys.argv) < 3:
        print("Usage: python logreg_check.py dataset_with_labels.csv weights.json")
        return

    data_path = sys.argv[1]
    weights_path = sys.argv[2]

    print(f"📘 Loading weights: {weights_path}")
    with open(weights_path, "r") as f:
        info = json.load(f)

    classes = info["classes"]
    feature_names = info["feature_names"]
    means = np.array(info["means"], dtype=float)
    stds = np.array(info["stds"], dtype=float)
    thetas = np.array(info["thetas"], dtype=float)  # shape (n_classes, n_features+1)

    print(f"📘 Loading dataset: {data_path}")
    df, numeric_cols = read_dataset(data_path)

    if "Hogwarts House" not in df.columns:
        raise ValueError("The dataset must contain a 'Hogwarts House' column for checking.")

    y_true = df["Hogwarts House"].astype(str).values

    # --- Preprocessing same as in training ---
    X_raw_cols = {}
    for i, col in enumerate(feature_names):
        if col in df.columns:
            X_raw_cols[col] = df[col].fillna(means[i]).astype(float).values
        else:
            print(f"⚠️ Warning: missing feature '{col}', filling with mean.")
            X_raw_cols[col] = np.full(len(df), means[i], dtype=float)

    X_raw = np.column_stack([X_raw_cols[c] for c in feature_names])
    X_norm = (X_raw - means) / stds
    m = X_norm.shape[0]
    X = np.hstack([np.ones((m, 1)), X_norm])  # add intercept

    # --- Prediction ---
    probs = sigmoid(X.dot(thetas.T))
    preds_idx = np.argmax(probs, axis=1)
    y_pred = [classes[i] for i in preds_idx]

    # --- Metrics ---
    df_result = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    # Overall accuracy
    acc = (df_result["Actual"] == df_result["Predicted"]).mean()
    print(f"\n🎯 Overall accuracy: {acc*100:.2f}%")

    # Per-class accuracy
    print("\n🏠 Accuracy per class:")
    for cls in classes:
        mask = df_result["Actual"] == cls
        if mask.sum() == 0:
            continue
        acc_cls = (df_result.loc[mask, "Actual"] == df_result.loc[mask, "Predicted"]).mean()
        print(f"  {cls:15s}: {acc_cls*100:.2f}%")

    # Confusion matrix
    print("\n📊 Confusion matrix:")
    cm = pd.crosstab(df_result["Actual"], df_result["Predicted"], rownames=["Actual"], colnames=["Predicted"], dropna=False)
    print(cm)

    # Optional: precision, recall, F1
    print("\n📈 Classification report:")
    for cls in classes:
        tp = ((df_result["Actual"] == cls) & (df_result["Predicted"] == cls)).sum()
        fp = ((df_result["Actual"] != cls) & (df_result["Predicted"] == cls)).sum()
        fn = ((df_result["Actual"] == cls) & (df_result["Predicted"] != cls)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {cls:15s}  P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}")

    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()
