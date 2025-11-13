# Makefile for Data Science Logistic Regression Project 🧙‍♂️

PYTHON = python3
DATASET_TRAIN = datasets/dataset_train.csv
DATASET_TEST = datasets/dataset_test.csv
WEIGHTS = weights.json

.PHONY: help install clean describe histogram pair_plot scatter_plot train predict run all

# === Help ===
help:
	@echo "📚 Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make describe      - Compute descriptive statistics"
	@echo "  make histogram     - Generate histogram visualizations"
	@echo "  make pair_plot     - Generate pair plot matrix"
	@echo "  make scatter_plot  - Generate scatter plot (auto-detects correlation)"
	@echo "  make train         - Train logistic regression model"
	@echo "  make predict       - Make predictions on test set"
	@echo "  make run           - Run full pipeline (describe → visualize → train → predict)"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make all           - Install + run full pipeline"

# === Setup ===
install:
	@echo "📦 Installing dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Installation complete."

# === Data Analysis ===
describe:
	@echo "📊 Computing descriptive statistics..."
	@$(PYTHON) describe.py $(DATASET_TRAIN)

histogram:
	@echo "📈 Generating histogram visualizations..."
	@$(PYTHON) histogram.py

pair_plot:
	@echo "🔗 Generating pair plot matrix..."
	@$(PYTHON) pair_plot.py

scatter_plot:
	@echo "📉 Generating scatter plot..."
	@$(PYTHON) scatter_plot.py

# === Machine Learning ===
train:
	@echo "🤖 Training logistic regression model..."
	@$(PYTHON) logreg_train.py $(DATASET_TRAIN) $(WEIGHTS)

predict:
	@echo "🔮 Making predictions..."
	@$(PYTHON) logreg_predict.py $(DATASET_TEST) $(WEIGHTS)

# === Full Pipeline ===
run: describe histogram pair_plot train predict
	@echo "🎉 Full pipeline completed successfully!"

all: install run

# === Cleanup ===
clean:
	@echo "🧹 Cleaning up cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✨ Cleanup complete."
