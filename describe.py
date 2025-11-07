import sys
import pandas as pd
from tabulate import tabulate
from utils import (
    read_dataset, count, mean, std, minimum, maximum, percentile
)


def compute_summary(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    """Compute descriptive statistics for each numeric column."""
    stats = {
        "count": [],
        "mean": [],
        "std": [],
        "min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "max": [],
    }

    for col in numeric_cols:
        values = df[col].tolist()
        stats["count"].append(count(values))
        stats["mean"].append(mean(values))
        stats["std"].append(std(values))
        stats["min"].append(minimum(values))
        stats["25%"].append(percentile(values, 25))
        stats["50%"].append(percentile(values, 50))
        stats["75%"].append(percentile(values, 75))
        stats["max"].append(maximum(values))

    return stats


def print_summary_table(numeric_cols: list[str], stats: dict) -> None:
    """Format and print the computed statistics as a table."""
    table = []
    for i, col in enumerate(numeric_cols):
        row = [
            col,
            _safe_round(stats["count"][i]),
            _safe_round(stats["mean"][i]),
            _safe_round(stats["std"][i]),
            _safe_round(stats["min"][i]),
            _safe_round(stats["25%"][i]),
            _safe_round(stats["50%"][i]),
            _safe_round(stats["75%"][i]),
            _safe_round(stats["max"][i]),
        ]
        table.append(row)

    headers = ["Feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))


def _safe_round(value, digits=6):
    """Round only if value is a valid number."""
    return round(value, digits) if value == value else "NaN"


def main():
    """Entry point: load dataset and display manual descriptive stats."""
    if len(sys.argv) < 2:
        print("Usage: python describe.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    df, numeric_cols = read_dataset(dataset_path)

    print("\n📊 Computing summary statistics...\n")

    stats = compute_summary(df, numeric_cols)
    print_summary_table(numeric_cols, stats)


if __name__ == "__main__":
    main()
