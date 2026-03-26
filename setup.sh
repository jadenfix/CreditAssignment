#!/usr/bin/env bash
set -euo pipefail

# CreditLab local setup script.
# Requires Python 3.11+. Run from the repo root.

REQUIRED_MINOR=11

python_version=$(python3 --version 2>&1 | awk '{print $2}')
minor=$(echo "$python_version" | cut -d. -f2)

if [ "$minor" -lt "$REQUIRED_MINOR" ]; then
    echo "ERROR: Python 3.${REQUIRED_MINOR}+ required. Found: $python_version" >&2
    exit 1
fi

echo "Python $python_version OK."
echo "Installing CreditLab in editable mode with dev extras..."
python3 -m pip install -e ".[dev]"

echo ""
echo "Verifying install..."
creditlab --help > /dev/null && echo "creditlab CLI: OK"

echo ""
echo "Running tests..."
python3 -m pytest tests/ -v

echo ""
echo "Setup complete."
echo "  make smoke       -- validate config + collect one run"
echo "  make run-paper   -- sweep all 6 paper conditions"
echo "  make clean-data  -- wipe experiment state for a fresh slate"
