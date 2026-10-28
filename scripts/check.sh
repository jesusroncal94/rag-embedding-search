#!/usr/bin/env bash
# The one check agents and CI run before calling a change done.
set -euo pipefail
cd "$(dirname "$0")/.."

# Compose interpolates every service, and api and ingest require GEMINI_API_KEY. The test
# services never use it, so a placeholder lets the check run without a .env (as in CI).
export GEMINI_API_KEY="${GEMINI_API_KEY:-not-used-by-the-test-services}"

docker compose run --rm --build tests ruff check .
docker compose run --rm tests ruff format --check .
docker compose run --rm tests
docker compose run --rm --build ui-tests
