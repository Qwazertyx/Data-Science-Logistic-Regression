# pair_plot.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_dataset, HOUSE_COLORS


def abbreviate_label(label, max_length=9):
    """Abbreviate labels to first 9 letters, add '.' if truncated."""
    if len(label) > max_length:
        return label[:max_length] + "."
    return label


def main():
    # Load dataset
    try:
        df, numeric_cols = read_dataset("datasets/dataset_train.csv")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    if 'Hogwarts House' not in df.columns:
        print("Error: 'Hogwarts House' column not found in dataset")
        sys.exit(1)

    # Remove non-feature columns
    numeric_cols = [
        col for col in numeric_cols
        if not any(keyword in col.lower() for keyword in ['index', 'id'])
    ]
    
    if len(numeric_cols) == 0:
        print("Error: No numeric feature columns found after filtering")
        sys.exit(1)

    # Keep relevant columns
    try:
        df_plot = df[["Hogwarts House"] + numeric_cols].dropna()
    except KeyError as e:
        print(f"Error: Missing required column: {str(e)}")
        sys.exit(1)
    
    if len(df_plot) == 0:
        print("Error: No data remaining after dropping NaN values")
        sys.exit(1)

    n = len(numeric_cols)
    try:
        fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    except Exception as e:
        print(f"Error: Failed to create plot figure: {str(e)}")
        sys.exit(1)

    # Create the matrix of plots
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:
                # Histogram on diagonal
                for house, color in HOUSE_COLORS.items():
                    subset = df_plot[df_plot["Hogwarts House"] == house]
                    values = subset[numeric_cols[i]].dropna()
                    ax.hist(values, bins=20, alpha=0.6, color=color, edgecolor='black', linewidth=0.3)
            else:
                # Scatter plot for pairs
                for house, color in HOUSE_COLORS.items():
                    subset = df_plot[df_plot["Hogwarts House"] == house]
                    ax.scatter(
                        subset[numeric_cols[j]],
                        subset[numeric_cols[i]],
                        color=color,
                        alpha=0.5,
                        s=8,
                        edgecolors='none'
                    )

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Show feature names on outer labels
            if j == 0:
                label = abbreviate_label(numeric_cols[i])
                ax.set_ylabel(label, fontsize=10, labelpad=6, rotation=0, ha='right', va='center')
            else:
                ax.set_ylabel("")
            
            if i == n - 1:
                label = abbreviate_label(numeric_cols[j])
                ax.set_xlabel(label, fontsize=10, rotation=90, labelpad=15, ha='center', va='top')
            else:
                ax.set_xlabel("")

    # Add title and layout adjustments
    fig.suptitle("Pair Plot of Hogwarts Features", y=0.995, fontsize=16, fontweight='bold')
    handles = [
        plt.Line2D([], [], marker="o", color=color, linestyle="", label=house, markersize=6)
        for house, color in HOUSE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.02), 
               title="Hogwarts House", fontsize=8, title_fontsize=9, framealpha=0.9)

    # Adjust layout for spacing
    plt.subplots_adjust(
        left=0.10, right=0.98, top=0.96, bottom=0.15, wspace=0.35, hspace=0.35
    )

    try:
        plt.show()
    except Exception as e:
        print(f"Error: Failed to display plot: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)
