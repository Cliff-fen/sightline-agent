PYTHON ?= python

.PHONY: test agent-build agent-test python-test generation-check

test: agent-build agent-test python-test

agent-build:
	cd agent && npm run build

agent-test:
	cd agent && npm test

python-test:
	$(PYTHON) -m pytest -q tests

generation-check:
	$(PYTHON) -m py_compile data/generation/*.py
