.PHONY: dev test lint check clean \
        dashboard-dev dashboard-dev-backend dashboard-dev-frontend \
        dashboard-test dashboard-test-e2e \
        dashboard-install dashboard-install-e2e \
        frontend-lint frontend-test frontend-build frontend-generate \
        release release-pr release-patch release-minor release-major

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
	rm -rf src/furu/dashboard/frontend/dist/
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
dashboard-dev: frontend-build
	@echo "Starting development servers..."
	@make -j2 dashboard-dev-backend dashboard-dev-frontend

dashboard-dev-backend:
	uv run uvicorn furu.dashboard.main:app --reload --host 0.0.0.0 --port 8000

dashboard-dev-frontend:
	cd dashboard-frontend && bun run dev

# Generate OpenAPI spec and TypeScript client
frontend-generate:
	uv run python -c "from furu.dashboard.main import app; import json; print(json.dumps(app.openapi()))" > openapi.json
	cd dashboard-frontend && bun run generate

# Build
frontend-build: frontend-generate
	cd dashboard-frontend && bunx @tanstack/router-cli generate && bun run build

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
	uv run furu-dashboard serve

# ============================================================================
# All tests (project + dashboard)
# ============================================================================

test-all: lint frontend-lint test dashboard-test-all

# ============================================================================
# Release Commands
# ============================================================================

# Get current version from pyproject.toml
CURRENT_VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# Tag current version: make release
release: test-all
	@echo "Tagging version: $(CURRENT_VERSION)"
	git tag -a "v$(CURRENT_VERSION)" -m "Release v$(CURRENT_VERSION)"
	git push --tags
	@echo ""
	@echo "Release v$(CURRENT_VERSION) tag pushed. GitHub Actions will create the release."

# Create release PR with a specific version: make release-pr VERSION=1.2.3 MESSAGE="What changed"
release-pr:
ifndef VERSION
	$(error VERSION is required. Usage: make release-pr VERSION=1.2.3)
endif
	@echo "Current version: $(CURRENT_VERSION)"
	@echo "New version: $(VERSION)"
	@read -p "Continue? [y/N] " confirm && [ "$$confirm" = "y" ]
	git checkout -b "release/v$(VERSION)"
	sed -i '' 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml
	uv sync
	git add pyproject.toml uv.lock
	git commit -m "bump version to v$(VERSION)"
	git push --set-upstream origin "release/v$(VERSION)"
	gh pr create --title "Release v$(VERSION)" --body "$(or $(MESSAGE),Release v$(VERSION))" --base main
	@echo ""
	@echo "Release PR created for v$(VERSION)."

# Convenience targets for semver bumps
release-patch:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}') && \
	CHANGELOG_LINE=$$(sed -n '3p' CHANGELOG.md) && \
	if [ "$$CHANGELOG_LINE" != "## v$$NEW_VERSION" ]; then \
		echo "Error: CHANGELOG.md line 3 must be '## v$$NEW_VERSION', but found '$$CHANGELOG_LINE'"; \
		exit 1; \
	fi && \
	$(MAKE) release-pr VERSION=$$NEW_VERSION

release-minor:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1"."$$2+1".0"}') && \
	CHANGELOG_LINE=$$(sed -n '3p' CHANGELOG.md) && \
	if [ "$$CHANGELOG_LINE" != "## v$$NEW_VERSION" ]; then \
		echo "Error: CHANGELOG.md line 3 must be '## v$$NEW_VERSION', but found '$$CHANGELOG_LINE'"; \
		exit 1; \
	fi && \
	$(MAKE) release-pr VERSION=$$NEW_VERSION

release-major:
	@NEW_VERSION=$$(echo $(CURRENT_VERSION) | awk -F. '{print $$1+1".0.0"}') && \
	CHANGELOG_LINE=$$(sed -n '3p' CHANGELOG.md) && \
	if [ "$$CHANGELOG_LINE" != "## v$$NEW_VERSION" ]; then \
		echo "Error: CHANGELOG.md line 3 must be '## v$$NEW_VERSION', but found '$$CHANGELOG_LINE'"; \
		exit 1; \
	fi && \
	$(MAKE) release-pr VERSION=$$NEW_VERSION
