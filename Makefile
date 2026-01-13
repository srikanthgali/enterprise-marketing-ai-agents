# Makefile for Enterprise Marketing AI Agents

.PHONY: help install test clean api dashboard all

help:
	@echo "Enterprise Marketing AI Agents - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup        - Full setup (install + initialize RAG)"
	@echo ""
	@echo "Running:"
	@echo "  make api          - Start FastAPI server"
	@echo "  make dashboard    - Start Streamlit dashboard"
	@echo "  make all          - Start API + Dashboard (parallel)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make test-api     - Test API endpoints"
	@echo "  make test-dashboard - Test dashboard integration"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Data:"
	@echo "  make data         - Run data extraction pipeline"
	@echo "  make rag-init     - Initialize RAG pipeline"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache and logs"
	@echo "  make clean-data   - Remove generated data"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✅ Installation complete"

setup: install
	@echo "Initializing system..."
	python scripts/run_data_extraction.py
	python scripts/initialize_rag_pipeline.py
	@echo "✅ Setup complete"

api:
	@echo "Starting FastAPI server..."
	python scripts/run_api.py

dashboard:
	@echo "Starting Streamlit dashboard..."
	@echo "Make sure API is running at http://localhost:8000"
	python scripts/run_dashboard.py

all:
	@echo "Starting API and Dashboard..."
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@./start_dashboard.sh

test:
	@echo "Running tests..."
	pytest tests/ -v

test-api:
	@echo "Testing API endpoints..."
	python scripts/test_api.py

test-dashboard:
	@echo "Testing dashboard integration..."
	python scripts/test_dashboard_integration.py

lint:
	@echo "Running linters..."
	flake8 src/ api/ ui/
	pylint src/ api/ ui/

format:
	@echo "Formatting code..."
	black src/ api/ ui/ tests/
	isort src/ api/ ui/ tests/

data:
	@echo "Running data extraction..."
	python scripts/run_data_extraction.py

rag-init:
	@echo "Initializing RAG pipeline..."
	python scripts/initialize_rag_pipeline.py

clean:
	@echo "Cleaning cache and logs..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf logs/*.log
	@echo "✅ Cleanup complete"

clean-data:
	@echo "Cleaning generated data..."
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/embeddings/*
	@echo "✅ Data cleanup complete"

.DEFAULT_GOAL := help
