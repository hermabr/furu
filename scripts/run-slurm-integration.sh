#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
compose_file="${repo_root}/tests/slurm/docker-compose.yml"
project_name="${COMPOSE_PROJECT_NAME:-furu-slurm-integration}"

cleanup() {
    docker compose \
        --project-name "${project_name}" \
        -f "${compose_file}" \
        down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose \
    --project-name "${project_name}" \
    -f "${compose_file}" \
    build

docker compose \
    --project-name "${project_name}" \
    -f "${compose_file}" \
    run --rm slurm-test "$@"
