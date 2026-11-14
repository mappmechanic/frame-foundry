PYTHON ?= .venv/bin/python
.PHONY: test demo-arizona fetch-arizona

test:
	$(PYTHON) -m pytest tests -q

fetch-arizona:
	$(PYTHON) scripts/arizona_demo.py --fetch-only $(ARGS)

demo-arizona:
	$(PYTHON) scripts/arizona_demo.py $(ARGS)
