.PHONY: dev test lint check clean \
        dashboard-dev dashboard-dev-backend dashboard-dev-frontend \
        dashboard-test dashboard-test-e2e \
        dashboard-install dashboard-install-e2e \
        frontend-lint frontend-test frontend-build frontend-generate \
        release release-patch release-minor release-major

# ============================================================================
# Main Project Commands
# ============================================================================

test:
	uv run pytest tests/ --ignore=tests/dashboard

lint:
	uv run ruff check
	uv run ty check

check: lint test

build: frontend-build test-all 
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
frontend-lint:
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
frontend-generate:
	uv run python -c "from huldra.dashboard.main import app; import json; print(json.dumps(app.openapi()))" > openapi.json
	cd dashboard-frontend && bun run generate

# Build
frontend-build: frontend-generate
	cd dashboard-frontend && bun run build

# Testing
dashboard-test:
	uv run pytest tests/dashboard

frontend-test:
	cd dashboard-frontend && bun test

dashboard-test-e2e:
	cd e2e && bun run test

dashboard-test-all: dashboard-test frontend-test dashboard-test-e2e

# Installation
dashboard-install:
	uv sync --all-extras
	cd dashboard-frontend && bun install

dashboard-install-e2e:
	cd e2e && bun install && bunx playwright install chromium

# Full setup for dashboard development
dashboard-setup: dashboard-install dashboard-install-e2e frontend-generate

# Build and serve (production mode)
dashboard-serve: frontend-build
	uv run huldra-dashboard serve

# ============================================================================
# All tests (project + dashboard)
# ============================================================================

test-all: lint frontend-lint test dashboard-test-all

# ============================================================================
# Release Commands
# ============================================================================

# Get current version from pyproject.toml
CURRENT_VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')

# Release with a specific version: make release VERSION=1.2.3
release: test-all
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=1.2.3)
endif
	@echo "Current version: $(CURRENT_VERSION)"
	@echo "New version: $(VERSION)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ]
	sed -i '' 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml
	git add pyproject.toml
	git commit -m "release v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push && git push --tags
	@echo ""
	@echo "Release v$(VERSION) pushed. GitHub Actions will create the release."

# Convenience targets for semver bumps
release-patch:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}') && \
	$(MAKE) release VERSION=$$NEW_VERSION

release-minor:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2+1".0"}') && \
	$(MAKE) release VERSION=$$NEW_VERSION

release-major:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1+1".0.0"}') && \
	$(MAKE) release VERSION=$$NEW_VERSION
