.PHONY: help install dev test lint typecheck format check serve clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

dev:  ## Install with dev dependencies
	uv sync --all-extras

test:  ## Run tests with coverage
	uv run pytest tests/ -v --cov=papers_mcp --cov-report=term-missing

test-fast:  ## Run tests without coverage
	uv run pytest tests/ -v

lint:  ## Run linting checks
	uv run ruff check src tests

typecheck:  ## Run type checking
	uv run mypy src tests

format:  ## Format code
	uv run ruff format src tests
	uv run ruff check --fix src tests

check:  ## Run all checks (lint, typecheck, test)
	@echo "Running linting..."
	uv run ruff check src tests
	@echo "\nRunning type checking..."
	uv run mypy src tests
	@echo "\nRunning tests..."
	uv run pytest tests/ -v --cov=papers_mcp --cov-report=term-missing

serve:  ## Start the MCP server
	uv run papers-mcp

serve-debug:  ## Start server with debug logging
	uv run papers-mcp --debug

clean:  ## Clean build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
