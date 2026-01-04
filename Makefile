.PHONY: dev test lint check clean \
        dashboard-dev dashboard-dev-backend dashboard-dev-frontend \
        dashboard-build dashboard-generate dashboard-test dashboard-test-e2e \
        dashboard-install dashboard-install-e2e dashboard-lint

# ============================================================================
# Main Project Commands
# ============================================================================

test:
	uv run pytest

lint:
	uv run ruff check
	uv run ty check

check: lint test

build: dashboard-build test-all 
	uv build

clean:
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf dist/
	rm -rf src/huldra/dashboard/frontend/dist/
	rm -rf dashboard-frontend/src/api/
	rm -rf e2e/playwright-report/
	rm -rf e2e/test-results/
	rm -f openapi.json
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

# ============================================================================
# Dashboard Commands
# ============================================================================

# Linting
dashboard-lint:
	cd dashboard-frontend && bun run lint

# Development
dashboard-dev:
	@echo "Starting development servers..."
	@make -j2 dashboard-dev-backend dashboard-dev-frontend

dashboard-dev-backend:
	uv run uvicorn huldra.dashboard.main:app --reload --host 0.0.0.0 --port 8000

dashboard-dev-frontend:
	cd dashboard-frontend && bun run dev

# Generate OpenAPI spec and TypeScript client
dashboard-generate:
	uv run python -c "from huldra.dashboard.main import app; import json; print(json.dumps(app.openapi()))" > openapi.json
	cd dashboard-frontend && bun run generate

# Build
dashboard-build: dashboard-generate
	cd dashboard-frontend && bun run build

# Testing
dashboard-test:
	uv run pytest tests/dashboard/ -v

dashboard-test-frontend:
	cd dashboard-frontend && bun test

dashboard-test-e2e:
	cd e2e && bun run test

dashboard-test-all: dashboard-test dashboard-test-frontend dashboard-test-e2e

# Installation
dashboard-install:
	uv sync --all-extras
	cd dashboard-frontend && bun install

dashboard-install-e2e:
	cd e2e && bun install && bunx playwright install chromium

# Full setup for dashboard development
dashboard-setup: dashboard-install dashboard-install-e2e dashboard-generate

# Build and serve (production mode)
dashboard-serve: dashboard-build
	uv run huldra-dashboard serve

# ============================================================================
# All tests (project + dashboard)
# ============================================================================

test-all: lint dashboard-lint test dashboard-test-all


