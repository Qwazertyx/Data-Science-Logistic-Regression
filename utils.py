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


# ===============================
# CORRELATION IMPLEMENTATION
# ===============================

def correlation(x_values, y_values):
    """
    Compute Pearson correlation coefficient between two series.
    
    Formula: r = Σ((x - x̄)(y - ȳ)) / sqrt(Σ(x - x̄)² * Σ(y - ȳ)²)
    
    Args:
        x_values: First series of values
        y_values: Second series of values (must be same length as x_values)
    
    Returns:
        Correlation coefficient (float) or NaN if insufficient data
    """
    # Pair values together and remove any NaN pairs
    pairs = []
    for i in range(len(x_values)):
        x_val = x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i]
        y_val = y_values.iloc[i] if hasattr(y_values, 'iloc') else y_values[i]
        
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        try:
            pairs.append((float(x_val), float(y_val)))
        except (ValueError, TypeError):
            continue
    
    if len(pairs) < 2:
        return float('nan')
    
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    
    # Calculate means
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    
    if math.isnan(x_mean) or math.isnan(y_mean):
        return float('nan')
    
    # Calculate numerator: Σ((x - x̄)(y - ȳ))
    numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(len(pairs)))
    
    # Calculate denominators: sqrt(Σ(x - x̄)² * Σ(y - ȳ)²)
    x_variance_sum = sum((x_vals[i] - x_mean) ** 2 for i in range(len(pairs)))
    y_variance_sum = sum((y_vals[i] - y_mean) ** 2 for i in range(len(pairs)))
    
    denominator = math.sqrt(x_variance_sum * y_variance_sum)
    
    if denominator == 0:
        return float('nan')
    
    return numerator / denominator


def correlation_matrix(df, columns):
    """
    Compute correlation matrix for given columns.
    
    Args:
        df: DataFrame containing the data
        columns: List of column names to compute correlations for
    
    Returns:
        DataFrame with correlation matrix (same structure as pandas corr())
    """
    n = len(columns)
    corr_data = {}
    
    for col_x in columns:
        corr_data[col_x] = {}
        for col_y in columns:
            if col_x == col_y:
                corr_data[col_x][col_y] = 1.0
            else:
                corr = correlation(df[col_x], df[col_y])
                corr_data[col_x][col_y] = corr
    
    return pd.DataFrame(corr_data, index=columns, columns=columns)