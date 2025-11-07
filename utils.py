import math
import pandas as pd

# ===============================
# VISUAL CONSTANTS
# ===============================

HOUSE_COLORS = {
    'Gryffindor': 'red',
    'Slytherin': 'green',
    'Ravenclaw': 'blue',
    'Hufflepuff': 'gold'
}


# ===============================
# BASIC STATS FUNCTIONS
# ===============================

import math
import pandas as pd

# ===============================
# BASIC STATS IMPLEMENTATION
# ===============================

def _to_floats(values):
    """Convert values to float when possible, skip non-numeric or NaN."""
    clean = []
    for v in values:
        if pd.isna(v):
            continue
        try:
            clean.append(float(v))
        except (ValueError, TypeError):
            continue
    return clean

def count(values):
    clean = _to_floats(values)
    return len(clean)

def mean(values):
    clean = _to_floats(values)
    n = len(clean)
    if n == 0:
        return float('nan')
    total = sum(clean)
    return total / n

def variance(values):
    clean = _to_floats(values)
    n = len(clean)
    if n <= 1:
        return float('nan')
    m = mean(clean)
    return sum((v - m) ** 2 for v in clean) / (n - 1)

def std(values):
    var = variance(values)
    return math.sqrt(var) if not math.isnan(var) else float('nan')

def minimum(values):
    clean = _to_floats(values)
    return min(clean) if clean else float('nan')

def maximum(values):
    clean = _to_floats(values)
    return max(clean) if clean else float('nan')

def percentile(values, p):
    clean = sorted(_to_floats(values))
    n = len(clean)
    if n == 0:
        return float('nan')
    k = (n - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return clean[int(k)]
    d0 = clean[int(f)] * (c - k)
    d1 = clean[int(c)] * (k - f)
    return d0 + d1


# ===============================
# DATA HELPERS
# ===============================

def read_dataset(path):
    """Load dataset and return only numeric columns."""
    print(f"📘 Loading dataset: {path}")
    df = pd.read_csv(path)
    numeric_cols = [col for col in df.columns if is_numeric_series(df[col])]
    return df, numeric_cols


def is_numeric_series(series):
    """Detect if a pandas series is numeric (even if stored as string)."""
    try:
        pd.to_numeric(series.dropna().iloc[:10])
        return True
    except Exception:
        return False


# ===============================
# HOGWARTS STATS
# ===============================

def compute_house_stats(df, courses):
    """Compute mean score per house per course."""
    houses = df['Hogwarts House'].unique()
    stats = {}
    for course in courses:
        stats[course] = {
            house: mean(df.loc[df['Hogwarts House'] == house, course])
            for house in houses
        }
    return stats


def homogeneity_score(house_means):
    """Return the variance of house means (lower = more homogeneous)."""
    return variance(list(house_means.values()))
