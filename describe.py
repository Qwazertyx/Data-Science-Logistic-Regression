# describe.py
import sys
import pandas as pd
from utils import load_dataset, clean_dataset, describe_numeric, check_missing_values

# Optional dependency for pretty-printing
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False


def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset_path>")
        return

    path = sys.argv[1]

    print(f"\n📘 Loading dataset: {path}")
    df = load_dataset(path)
    df = clean_dataset(df)

    print("\n🔍 Checking for missing values...")
    check_missing_values(df)

    print("\n📊 Computing summary statistics...\n")
    desc = describe_numeric(df)

    # Round numeric values for better readability
    desc = desc.round(6)

    # Reorder columns to match classical describe order
    columns_order = [
        "count", "mean", "std", "min", "25%", "50%", "75%", "max"
    ]
    desc = desc[[col for col in columns_order if col in desc.columns]]

    # Display using tabulate if available
    if TABULATE_AVAILABLE:
        print(tabulate(desc, headers='keys', tablefmt='fancy_grid', showindex=True))
    else:
        # fallback: aligned plain text
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
            print(desc)


if __name__ == "__main__":
    main()
