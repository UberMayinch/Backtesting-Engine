.PHONY: help install dev test lint format clean demo sync

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies using uv
	uv sync

dev: ## Install with development dependencies
	uv sync --extra dev

test: ## Run tests
	uv run pytest

lint: ## Run linting checks
	uv run ruff check .
	uv run mypy .

format: ## Format code
	uv run black .
	uv run ruff check --fix .

demo: ## Run the demo script
	uv run demo.py

sync: ## Sync dependencies (equivalent to install)
	uv sync

clean: ## Clean up cache and temporary files
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf .venv/
	rm -rf *.egg-info/
	rm *.png
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

check: ## Run all checks (lint + test)
	make lint
	make test

build: ## Build the package
	uv build

publish: ## Publish to PyPI (requires authentication)
	uv publish

lock: ## Update uv.lock file
	uv lock

add: ## Add a new dependency (usage: make add PKG=package-name)
	@if [ -z "$(PKG)" ]; then echo "Usage: make add PKG=package-name"; exit 1; fi
	uv add $(PKG)

add-dev: ## Add a development dependency (usage: make add-dev PKG=package-name)
	@if [ -z "$(PKG)" ]; then echo "Usage: make add-dev PKG=package-name"; exit 1; fi
	uv add --dev $(PKG)
