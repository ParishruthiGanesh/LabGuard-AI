# LabGuard AI — common tasks.
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := backend/.venv/bin/python
PIP := backend/.venv/bin/pip

.PHONY: help setup backend-setup frontend-setup demo api dashboard test lint format check build clean deploy

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: backend-setup frontend-setup ## Install everything

backend-setup: ## Create the Python venv and install dependencies
	python3 -m venv backend/.venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r backend/requirements-dev.txt

frontend-setup: ## Install dashboard dependencies
	cd frontend && npm install

demo: ## Run the full workflow headless and print the verdict
	$(PY) scripts/run_demo.py

api: ## Start the API on :8080 (demo mode)
	cd backend && LABGUARD_MODE=demo .venv/bin/python -m uvicorn labguard.api.app:app \
		--host 127.0.0.1 --port 8080 --reload

dashboard: ## Start the dashboard on :3000
	cd frontend && npm run dev

test: ## Run the backend test suite
	cd backend && .venv/bin/python -m pytest tests/ -q

lint: ## Lint backend and frontend
	cd backend && .venv/bin/ruff check . && .venv/bin/ruff format --check .
	cd frontend && npx tsc --noEmit && npx next lint

format: ## Auto-format the backend
	cd backend && .venv/bin/ruff format . && .venv/bin/ruff check . --fix

check: lint test ## Lint, typecheck and test everything

build: ## Production build of the dashboard
	cd frontend && npm run build

deploy: ## Provision Google Cloud and deploy (needs gcloud and a project)
	./deploy/provision.sh $${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT} $${GOOGLE_CLOUD_REGION:-us-central1}

clean: ## Remove build output and caches
	rm -rf backend/artifacts artifacts frontend/.next
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.pytest_cache backend/.ruff_cache
