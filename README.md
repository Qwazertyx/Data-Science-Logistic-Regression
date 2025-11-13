# 🧙‍♂️ Data Science & Logistic Regression

**End-to-end data science project**: Custom statistical implementations, EDA, and logistic regression from scratch. Multi-class classification of Hogwarts students into 4 houses.

## 🎯 Key Achievements

- **Custom Statistics**: Mean, variance, std, percentiles, correlation — all implemented from scratch (no pandas shortcuts)
- **Machine Learning**: Logistic regression with gradient descent, one-vs-rest classification
- **Model Performance**: **98.19% accuracy** on test set

## 📊 Data Science Implementation

### Custom Statistical Functions
All computations implemented manually:
- `mean()`, `variance()`, `std()`, `percentile()`
- `correlation()` - Pearson correlation coefficient
- `correlation_matrix()` - Full correlation analysis

### Data Visualization

**Pair Plot Matrix** - Feature relationships across 14 dimensions
<img src="images/pair_plot.png" width="300" alt="Pair Plot">

**Score Distribution** - Histograms by house for each course
<img src="images/score_distribution.png" width="300" alt="Score Distribution">

**Most Homogeneous Course** - Course with smallest variance between houses
<img src="images/most_homogeneous_course.png" width="300" alt="Most Homogeneous Course">

**Most Similar Features** - Auto-detected highest correlation pair
<img src="images/most_similar_features.png" width="300" alt="Most Similar Features">

## 🤖 Machine Learning

### Training (`logreg_train.py`)
- **Gradient descent** from scratch
- **One-vs-rest** multi-class strategy
- Feature standardization (z-score)
- Missing value imputation
- Saves weights + preprocessing to JSON

### Results

**🎯 Overall Accuracy: 98.19%**

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Gryffindor  | 0.988     | 0.972  | 0.980    |
| Hufflepuff | 0.979     | 0.992  | 0.986    |
| Ravenclaw   | 0.975     | 0.982  | 0.979    |
| Slytherin  | 0.990     | 0.973  | 0.982    |

**Confusion Matrix:**
| Actual \ Predicted | Gryffindor | Hufflepuff | Ravenclaw | Slytherin |
|-------------------|------------|------------|-----------|-----------|
| **Gryffindor**    | 318        | 4          | 5         | 0         |
| **Hufflepuff**    | 2          | 525        | 1         | 1         |
| **Ravenclaw**     | 2          | 4          | 435       | 2         |
| **Slytherin**     | 0          | 3          | 5         | 293       |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
make run

# Or individual steps
make train      # Train model
make predict    # Generate predictions
```
---
