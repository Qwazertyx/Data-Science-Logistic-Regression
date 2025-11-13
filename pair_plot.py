# pair_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from utils import read_dataset, HOUSE_COLORS


def abbreviate_label(label, max_length=15):
    """Abbreviate long labels to prevent overlapping."""
    # Common abbreviations for Hogwarts subjects
    abbreviations = {
        'Defense Against the Dark Arts': 'Defense DADA',
        'Care of Magical Creatures': 'Magical Creatures',
        'History of Magic': 'History Magic',
        'Muggle Studies': 'Muggle Studies',
        'Ancient Runes': 'Ancient Runes',
        'Transfiguration': 'Transfiguration',
        'Divination': 'Divination',
        'Herbology': 'Herbology',
        'Astronomy': 'Astronomy',
        'Arithmancy': 'Arithmancy',
        'Potions': 'Potions',
        'Charms': 'Charms',
        'Flying': 'Flying',
        'Score': 'Score'
    }
    
    if label in abbreviations:
        return abbreviations[label]
    
    # If label is still too long, wrap it
    if len(label) > max_length:
        return '\n'.join(textwrap.wrap(label, max_length))
    return label


def main():
    # 1️⃣ Load dataset
    df, numeric_cols = read_dataset("datasets/dataset_train.csv")

    # 2️⃣ Remove non-feature columns
    numeric_cols = [
        col for col in numeric_cols
        if not any(keyword in col.lower() for keyword in ['index', 'id'])
    ]

    # 3️⃣ Keep relevant columns
    df_plot = df[["Hogwarts House"] + numeric_cols].dropna()

    n = len(numeric_cols)
    # Increase figure size for better readability
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))

    # 4️⃣ Create the matrix of plots
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

            # Show only feature names on outer labels with better formatting
            if j == 0:
                label = abbreviate_label(numeric_cols[i])
                ax.set_ylabel(label, fontsize=10, labelpad=12, rotation=0, ha='right', va='center')
            else:
                ax.set_ylabel("")
            
            if i == n - 1:
                label = abbreviate_label(numeric_cols[j])
                ax.set_xlabel(label, fontsize=10, rotation=90, labelpad=12, ha='center', va='top')
            else:
                ax.set_xlabel("")

    # 5️⃣ Add title and layout adjustments
    fig.suptitle("Pair Plot of Hogwarts Features", y=0.995, fontsize=16, fontweight='bold')
    handles = [
        plt.Line2D([], [], marker="o", color=color, linestyle="", label=house, markersize=8)
        for house, color in HOUSE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", title="Hogwarts House", 
               fontsize=11, title_fontsize=12, framealpha=0.9)

    # Adjust layout for spacing - more space for labels
    plt.subplots_adjust(
        left=0.08, right=0.96, top=0.96, bottom=0.08, wspace=0.35, hspace=0.35
    )

    plt.show()


if __name__ == "__main__":
    main()
