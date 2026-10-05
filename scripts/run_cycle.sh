#!/usr/bin/env bash
# Wrapper that cron/launchd calls. Runs ONE agent cycle from the repo directory.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec ./.venv/bin/python -m agent run >> logs/cron.log 2>&1
