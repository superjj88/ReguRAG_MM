.PHONY: setup test lint type coverage up down health package help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dev deps + pre-commit hooks
	python3 -m pip install -r backend/requirements.txt -r frontend/requirements.txt -r requirements-dev.txt
	python3 -m pre_commit install

test: ## Run offline test suite
	python3 -m pytest -q

lint: ## Ruff lint
	python3 -m ruff check backend/ tests/

type: ## Mypy type check
	python3 -m mypy --no-site-packages backend/

coverage: ## Tests with coverage report
	python3 -m pytest -q --cov=backend --cov-report=term-missing

up: ## Build & start the full stack
	docker compose up -d --build

down: ## Stop the stack
	docker compose down

health: ## Smoke-test the stack
	curl -sf http://127.0.0.1:8000/api/health && echo

package: ## Build the offline bundle
	bash scripts/package_for_offline.sh
