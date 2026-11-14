.DEFAULT_GOAL := test
VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
UV ?= $(if $(wildcard $(HOME)/.local/bin/uv),$(HOME)/.local/bin/uv,uv)
.PHONY: setup test demo-arizona fetch-arizona

# Explicitly recreate the environment; do not run during an active Episode.
setup:
	"$(UV)" venv -p 3.13 --clear "$(VENV)"
	"$(UV)" pip install --python "$(VENV)/bin/python" -r requirements-dev.txt -e .

test:
	$(PYTHON) -m pytest tests -q

fetch-arizona:
	$(PYTHON) scripts/arizona_demo.py --fetch-only $(ARGS)

demo-arizona:
	$(PYTHON) scripts/arizona_demo.py $(ARGS)
