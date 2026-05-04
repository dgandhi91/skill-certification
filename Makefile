.PHONY: help install test quality local dashboard clean

# Variables
PYTHON ?= python3
VENV := venv
BIN := $(VENV)/bin

# Colors
RESET := \033[0m
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[36m
CYAN := \033[96m

.DEFAULT_GOAL := help

help: ## Show available commands
	@echo "$(BOLD)$(CYAN)Skill Certification Pipeline$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-12s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BOLD)$(YELLOW)Examples:$(RESET)"
	@echo "  make install                    # First time setup"
	@echo "  make local SKILL=csv-analyzer   # Run certification (auto-setup)"
	@echo "  make quality                    # Format & lint code"
	@echo "  make dashboard                  # Open web UI"
	@echo ""

install: ## Setup environment, install dependencies + dev tools
	@echo "$(BLUE)Setting up environment...$(RESET)"
	@$(PYTHON) -m venv $(VENV)
	@$(BIN)/pip install --upgrade pip setuptools wheel
	@$(BIN)/pip install -r requirements.txt
	@$(BIN)/pip install black isort flake8 pytest pytest-cov pytest-anyio
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)✓ Created .env file - please add your API keys$(RESET)"; \
	fi
	@echo "$(GREEN)✓ Installation complete!$(RESET)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	@$(BIN)/pytest tests/ -v && echo "$(GREEN)✓ All tests passed!$(RESET)" || echo "$(YELLOW)⚠ Some tests failed$(RESET)"

quality: ## Format with Black & isort, check with flake8
	@echo "$(BLUE)Sorting imports with isort...$(RESET)"
	@$(BIN)/isort app/ tests/ *.py
	@echo "$(BLUE)Formatting code with Black...$(RESET)"
	@$(BIN)/black app/ tests/ *.py
	@echo "$(BLUE)Checking code quality with flake8...$(RESET)"
	@$(BIN)/flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203,W503 && echo "$(GREEN)✓ Code looks good$(RESET)" || echo "$(YELLOW)⚠ Found some issues$(RESET)"

local: ## Run certification locally with auto-setup (make local SKILL=csv-analyzer)
	@# Check if venv exists, if not run install
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)Virtual environment not found, running install...$(RESET)"; \
		$(MAKE) install; \
	fi
	@# Check if .env exists, if not create it
	@if [ ! -f .env ]; then \
		echo "$(YELLOW).env not found, creating from .env.example...$(RESET)"; \
		cp .env.example .env; \
		echo "$(YELLOW)⚠ Please edit .env and add your API keys$(RESET)"; \
		exit 1; \
	fi
	@# Run certification
	@if [ -z "$(SKILL)" ]; then \
		echo "$(YELLOW)Usage: make local SKILL=csv-analyzer$(RESET)"; \
		echo ""; \
		echo "$(BLUE)Available skills:$(RESET)"; \
		find skills -name "SKILL.md" -exec dirname {} \; | sed 's|skills/||' | grep -v workspace | sed 's/^/  - /'; \
		exit 1; \
	fi
	@echo "$(BLUE)Running certification for: $(SKILL)$(RESET)"
	@$(BIN)/python run.py skills/$(SKILL) skills/$(SKILL)-workspace && echo "$(GREEN)✓ Certification complete!$(RESET)" || echo "$(YELLOW)⚠ Certification failed$(RESET)"

dashboard: ## Launch interactive web dashboard with auto-setup
	@# Check if venv exists, if not run install
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(YELLOW)Virtual environment not found, running install...$(RESET)"; \
		$(MAKE) install; \
	fi
	@# Check if .env exists
	@if [ ! -f .env ]; then \
		echo "$(YELLOW).env not found, creating from .env.example...$(RESET)"; \
		cp .env.example .env; \
		echo "$(YELLOW)⚠ Please edit .env and add your API keys, then run 'make dashboard' again$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Launching Streamlit dashboard...$(RESET)"
	@echo "$(CYAN)→ Opening http://localhost:8501$(RESET)"
	@$(BIN)/streamlit run streamlit_app.py

clean: ## Remove cache files and virtual environment
	@echo "$(BLUE)Cleaning up...$(RESET)"
	@rm -rf $(VENV) __pycache__ app/**/__pycache__ results/ .pytest_cache .mypy_cache htmlcov/
	@echo "$(GREEN)✓ Cleanup complete$(RESET)"
