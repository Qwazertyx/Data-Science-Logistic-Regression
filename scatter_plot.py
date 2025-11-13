# scatter_plot.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import read_dataset, HOUSE_COLORS, correlation_matrix


def main():
    # 🧩 Argument parser setup
    parser = argparse.ArgumentParser(
        description="Display a scatter plot comparing two Hogwarts features."
    )
    parser.add_argument(
        "features",
        nargs="*",
        help="Two feature names to compare (optional). Example: Arithmancy Astronomy"
    )
    args = parser.parse_args()

    # 1️⃣ Load dataset
    df, numeric_cols = read_dataset("datasets/dataset_train.csv")

    # 2️⃣ Remove index-like columns
    numeric_cols = [
        col for col in numeric_cols
        if not any(keyword in col.lower() for keyword in ['index', 'id'])
    ]

    # 3️⃣ Determine which features to use
    if len(args.features) == 2:
        # User provided 2 features explicitly
        feature_x, feature_y = args.features
        if feature_x not in numeric_cols or feature_y not in numeric_cols:
            print("❌ Error: One or both provided features are not numeric columns.")
            print(f"Available numeric columns: {', '.join(numeric_cols)}")
            return
        print(f"📊 Comparing user-selected features: {feature_x} vs {feature_y}")
    else:
        # Auto-detect most correlated pair using custom correlation function
        corr_matrix = correlation_matrix(df, numeric_cols)
        
        # Find the maximum absolute correlation value and its position
        # (excluding diagonal self-correlations)
        max_corr = 0
        feature_x = None
        feature_y = None
        
        for col_x in numeric_cols:
            for col_y in numeric_cols:
                if col_x == col_y:
                    continue  # Skip diagonal (self-correlation)
                
                corr_val = corr_matrix.loc[col_x, col_y]
                if pd.isna(corr_val):
                    continue
                
                # Take absolute value manually
                abs_corr = abs(corr_val)
                if abs_corr > max_corr:
                    max_corr = abs_corr
                    feature_x = col_x
                    feature_y = col_y
        
        if feature_x is None or feature_y is None:
            print("❌ Error: Could not find correlated features.")
            return
        
        print(f"🔍 Most similar features: {feature_x} and {feature_y} (corr={max_corr:.2f})")

    # 4️⃣ Drop NaN values in the selected features
    df_plot = df[[feature_x, feature_y, "Hogwarts House"]].dropna()

    # 5️⃣ Create scatter plot
    plt.figure(figsize=(8, 6))
    for house, color in HOUSE_COLORS.items():
        subset = df_plot[df_plot["Hogwarts House"] == house]
        plt.scatter(
            subset[feature_x],
            subset[feature_y],
            label=house,
            color=color,
            alpha=0.6
        )

    plt.title(f"Scatter Plot: {feature_x} vs {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
