.PHONY: help setup test smoke run-paper clean clean-data

PYTHON ?= python3
PIP    ?= $(PYTHON) -m pip

help:
	@echo "CreditLab targets:"
	@echo "  make setup        Install package + dev deps in editable mode"
	@echo "  make test         Run all unit tests"
	@echo "  make smoke        Validate config + collect one run (diagnostic, seed=7)"
	@echo "  make run-paper    Sweep all 6 paper conditions (local, diagnostic)"
	@echo "  make clean        Remove Python cache files"
	@echo "  make clean-data   Archive SQLite DB then wipe runs/ and artifacts/ contents"

setup:
	$(PIP) install -e ".[dev]"
	@echo "Install complete. Run 'make test' to verify."

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	creditlab validate-config --config configs/diagnostic_smoke.yaml
	creditlab collect --config configs/diagnostic_smoke.yaml

run-paper:
	creditlab sweep --config configs/diagnostic_paper.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true

clean-data:
	@echo "Archiving existing run data before deletion..."
	@if [ -f runs/creditlab.sqlite ]; then \
	    cp runs/creditlab.sqlite runs/creditlab.sqlite.bak.$$(date +%Y%m%d_%H%M%S); \
	    echo "  Backed up runs/creditlab.sqlite"; \
	fi
	@rm -f runs/creditlab.sqlite
	@rm -rf artifacts/runs artifacts/paper artifacts/policies
	@echo "Done. Fresh slate ready."
