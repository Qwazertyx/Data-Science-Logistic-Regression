# Makefile for Data Science Logistic Regression Project

PYTHON = python3
DATASET_TRAIN = datasets/dataset_train.csv
DATASET_TEST = datasets/dataset_test.csv
WEIGHTS = weights.json

.PHONY: install clean describe histogram pair_plot scatter_plot train predict run all

install:
	@echo "Installing dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "Installation complete."

describe:
	@echo "Computing descriptive statistics..."
	@$(PYTHON) describe.py $(DATASET_TRAIN)

histogram:
	@echo "Generating histogram visualizations..."
	@$(PYTHON) histogram.py

pair_plot:
	@echo "Generating pair plot matrix..."
	@$(PYTHON) pair_plot.py

scatter_plot:
	@echo "Generating scatter plot..."
	@$(PYTHON) scatter_plot.py

train:
	@echo "Training logistic regression model..."
	@$(PYTHON) logreg_train.py $(DATASET_TRAIN) $(WEIGHTS)

predict:
	@echo "Making predictions..."
	@$(PYTHON) logreg_predict.py $(DATASET_TEST) $(WEIGHTS)

run: describe histogram pair_plot train predict
	@echo "Full pipeline completed successfully."

all: install run

clean:
	@echo "Cleaning up cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "Cleanup complete."
