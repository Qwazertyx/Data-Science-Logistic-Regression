# 🧙‍♂️ Data Science & Logistic Regression Project

A comprehensive data science project demonstrating **exploratory data analysis**, **statistical computing**, and **machine learning** implementation from scratch. This project analyzes Hogwarts student data and implements a multi-class logistic regression classifier to predict house assignments.

## 📊 Project Overview

This project showcases hands-on experience in:
- **Data Analysis**: Custom statistical functions (mean, variance, correlation) implemented from scratch
- **Data Visualization**: Pair plots, histograms, and scatter plots with proper formatting
- **Machine Learning**: Logistic regression with gradient descent implemented from scratch
- **Classification**: Multi-class prediction using one-vs-rest strategy

## 🎯 Key Features

### 1. Statistical Analysis (`describe.py`)
- Custom implementation of descriptive statistics (mean, std, percentiles, etc.)
- No reliance on pandas statistical functions
- Clean tabular output using `tabulate`

### 2. Data Visualization

#### Pair Plot Matrix (`pair_plot.py`)
![Pair Plot](images/pair_plot.png)
- 14×14 matrix showing relationships between all features
- Diagonal: Histograms showing score distributions by house
- Off-diagonal: Scatter plots showing feature correlations
- Optimized labels and spacing for readability

#### Histogram Analysis (`histogram.py`)
![Score Distribution](images/score_distribution.png)
![Most Homogeneous Course](images/most_homogeneous_course.png)
- Score distributions by Hogwarts house for each course
- Identification of most homogeneous course (smallest variance between houses)
- Mean indicators on histograms

#### Scatter Plot (`scatter_plot.py`)
![Most Similar Features](images/most_similar_features.png)
- Custom correlation function implementation
- Auto-detection of most correlated feature pairs
- Color-coded by house for pattern recognition

### 3. Machine Learning Implementation

#### Logistic Regression Training (`logreg_train.py`)
- **Gradient descent** implemented from scratch
- **One-vs-rest** multi-class classification strategy
- Feature standardization (z-score normalization)
- Missing value imputation with mean
- Saves trained weights and preprocessing parameters to JSON

#### Prediction (`logreg_predict.py`)
- Loads trained model weights
- Applies same preprocessing pipeline
- Generates predictions for test set
- Outputs results to `houses.csv`

## 📈 Results

The logistic regression model successfully:
- Trains on 14 numeric features
- Classifies students into 4 Hogwarts houses (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)
- Uses one-vs-rest strategy for multi-class classification
- Generates predictions with proper preprocessing pipeline

### Data Analysis Results

Descriptive statistics computed using custom implementations (no pandas statistical functions):

| Feature | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|-------|------|-----|-----|-----|-----|-----|-----|
| **Index** | 1600 | 799.5 | 462.025 | 0 | 399.75 | 799.5 | 1199.25 | 1599 |
| **Arithmancy** | 1566 | 49634.6 | 16679.8 | -24370 | 38511.5 | 49013.5 | 60811.2 | 104956 |
| **Astronomy** | 1568 | 39.80 | 520.30 | -966.74 | -489.55 | 260.29 | 524.77 | 1016.21 |
| **Herbology** | 1567 | 1.14 | 5.22 | -10.30 | -4.31 | 3.47 | 5.42 | 11.61 |
| **Defense Against the Dark Arts** | 1569 | -0.39 | 5.21 | -10.16 | -5.26 | -2.59 | 4.90 | 9.67 |
| **Divination** | 1561 | 3.15 | 4.16 | -8.73 | 3.10 | 4.62 | 5.67 | 10.03 |
| **Muggle Studies** | 1565 | -224.59 | 486.35 | -1086.5 | -577.58 | -419.16 | 254.99 | 1092.39 |
| **Ancient Runes** | 1565 | 495.75 | 106.29 | 283.87 | 397.51 | 463.92 | 597.49 | 745.40 |
| **History of Magic** | 1557 | 2.96 | 4.43 | -8.86 | 2.22 | 4.38 | 5.83 | 11.89 |
| **Transfiguration** | 1566 | 1030.10 | 44.13 | 906.63 | 1026.21 | 1045.51 | 1058.44 | 1098.96 |
| **Potions** | 1570 | 5.95 | 3.15 | -4.70 | 3.65 | 5.87 | 8.25 | 13.54 |
| **Care of Magical Creatures** | 1560 | -0.05 | 0.97 | -3.31 | -0.67 | -0.04 | 0.59 | 3.06 |
| **Charms** | 1600 | -243.37 | 8.78 | -261.05 | -250.65 | -244.87 | -232.55 | -225.43 |
| **Flying** | 1600 | 21.96 | 97.63 | -181.47 | -41.87 | -2.52 | 50.56 | 279.07 |

**Key Observations:**
- Dataset contains 1,600 students with 14 numeric features
- Some features have missing values (counts range from 1557 to 1600)
- Wide range of scales across features (e.g., Arithmancy: ~50K, Charms: ~-243)
- Feature standardization is crucial for model performance

### Model Performance

The trained model achieves excellent performance on the test set:

**🎯 Overall Accuracy: 98.19%**

**🏠 Accuracy per Class:**
- **Gryffindor**: 97.25%
- **Hufflepuff**: 99.24%
- **Ravenclaw**: 98.19%
- **Slytherin**: 97.34%

**📊 Confusion Matrix:**

| Actual \ Predicted | Gryffindor | Hufflepuff | Ravenclaw | Slytherin |
|-------------------|------------|------------|-----------|-----------|
| **Gryffindor**    | 318        | 4          | 5         | 0         |
| **Hufflepuff**    | 2          | 525        | 1         | 1         |
| **Ravenclaw**     | 2          | 4          | 435       | 2         |
| **Slytherin**     | 0          | 3          | 5         | 293       |

**📈 Classification Report:**

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Gryffindor  | 0.988     | 0.972  | 0.980    |
| Hufflepuff | 0.979     | 0.992  | 0.986    |
| Ravenclaw   | 0.975     | 0.982  | 0.979    |
| Slytherin  | 0.990     | 0.973  | 0.982    |

The model demonstrates strong performance across all classes with balanced precision and recall, indicating effective classification without significant bias toward any particular house.


## 🛠️ Technical Implementation

### Custom Statistical Functions
All statistical computations are implemented from scratch:
- `mean()`, `variance()`, `std()` - Basic statistics
- `percentile()` - Quantile computation
- `correlation()` - Pearson correlation coefficient
- `correlation_matrix()` - Full correlation matrix computation

### Machine Learning Components
- **Sigmoid activation** with numerical stability
- **Gradient descent** with configurable learning rate and iterations
- **Cost function**: Binary cross-entropy loss
- **Feature engineering**: Standardization and imputation

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Data-Science-Logistic-Regression

# Install dependencies
make install
# or manually:
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start (Full Pipeline)
```bash
make run
```

### Individual Commands

**Data Analysis:**
```bash
make describe      # Compute descriptive statistics
make histogram     # Generate histogram visualizations
make pair_plot     # Generate pair plot matrix
make scatter_plot  # Generate scatter plot
```

**Machine Learning:**
```bash
make train         # Train the logistic regression model
make predict       # Make predictions on test set
```

### Manual Execution

```bash
# Descriptive statistics
python describe.py datasets/dataset_train.csv

# Visualizations
python histogram.py
python pair_plot.py
python scatter_plot.py

# Training
python logreg_train.py datasets/dataset_train.csv weights.json

# Prediction
python logreg_predict.py datasets/dataset_test.csv weights.json
```

## 🔧 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `tabulate` - Pretty table formatting

## 📝 Notes

- All statistical functions are implemented from scratch (no pandas `.corr()`, `.mean()`, etc.)
- The project follows educational best practices by implementing core algorithms manually
- Visualizations are optimized for readability with proper label formatting and spacing

---

