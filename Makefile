.PHONY: install run dashboard clean help

PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

run: ## Run certification pipeline (SKILL=, WORKSPACE=, ITERATION= optional)
	$(BIN)/python run.py $(SKILL) $(WORKSPACE) $(if $(ITERATION),--iteration $(ITERATION))

dashboard: ## Launch the Streamlit dashboard
	$(BIN)/streamlit run streamlit_app.py

clean: ## Remove venv, caches, and results
	rm -rf $(VENV) __pycache__ app/**/__pycache__ results/ .mypy_cache .pytest_cache
