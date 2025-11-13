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

## 📁 Project Structure

```
Data-Science-Logistic-Regression/
├── datasets/
│   ├── dataset_train.csv    # Training data
│   └── dataset_test.csv     # Test data
├── images/                  # Generated visualizations
│   ├── pair_plot.png
│   ├── score_distribution.png
│   ├── most_homogeneous_course.png
│   └── most_similar_features.png
├── describe.py              # Statistical analysis
├── histogram.py             # Histogram visualizations
├── pair_plot.py             # Pair plot matrix
├── scatter_plot.py          # Scatter plot with correlation
├── logreg_train.py          # Model training
├── logreg_predict.py        # Model prediction
├── utils.py                 # Shared utilities & custom functions
├── weights.json             # Trained model weights
├── houses.csv               # Predictions output
├── requirements.txt         # Python dependencies
├── Makefile                 # Build automation
└── README.md                # This file
```

## 🎓 Skills Demonstrated

### Data Science
- ✅ Exploratory Data Analysis (EDA)
- ✅ Statistical computing from scratch
- ✅ Data visualization (matplotlib)
- ✅ Feature correlation analysis
- ✅ Data preprocessing and cleaning

### Machine Learning
- ✅ Logistic regression implementation
- ✅ Gradient descent optimization
- ✅ Multi-class classification
- ✅ Model training and evaluation
- ✅ Feature standardization

### Software Engineering
- ✅ Clean, modular code structure
- ✅ Custom utility functions
- ✅ Command-line interfaces
- ✅ Makefile automation
- ✅ Documentation

## 📈 Results

The logistic regression model successfully:
- Trains on 14 numeric features
- Classifies students into 4 Hogwarts houses (Gryffindor, Slytherin, Ravenclaw, Hufflepuff)
- Uses one-vs-rest strategy for multi-class classification
- Generates predictions with proper preprocessing pipeline

## 🔧 Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `tabulate` - Pretty table formatting

## 📝 Notes

- All statistical functions are implemented from scratch (no pandas `.corr()`, `.mean()`, etc.)
- The project follows educational best practices by implementing core algorithms manually
- Visualizations are optimized for readability with proper label formatting and spacing

## 👤 Author

Demonstrating proficiency in data science, statistical analysis, and machine learning implementation.

---

**Built with ❤️ for data science and machine learning**
