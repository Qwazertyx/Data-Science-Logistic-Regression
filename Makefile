# Makefile for DSLR Project 🧙‍♂️

PYTHON = python
DATASET = datasets/dataset_train.csv

# === Setup ===
install:
	@echo "📦 Installing dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Installation complete."

# === Data Description ===
describe:
	@echo "📊 Running describe.py..."
	@$(PYTHON) describe.py $(DATASET)

# === Cleanup ===
clean:
	@echo "🧹 Cleaning up cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@echo "✨ Cleanup complete."

# === Full pipeline (for later use) ===
run:
	@echo "🏁 Running full DSLR pipeline..."
	@$(PYTHON) describe.py $(DATASET)
	@$(PYTHON) histogram.py $(DATASET)
	@$(PYTHON) pair_plot.py $(DATASET)
	@$(PYTHON) logreg_train.py $(DATASET)
	@$(PYTHON) logreg_predict.py
	@echo "🎉 All scripts executed successfully."
