PYTHON ?= python

.PHONY: test smoke-data agent-build agent-test python-test generation-check

test: agent-build agent-test python-test

agent-build:
	cd agent && npm run build

agent-test:
	cd agent && npm test

python-test:
	$(PYTHON) -m pytest -q training_tests

smoke-data:
	$(PYTHON) data/make_smoke_data.py

generation-check:
	$(PYTHON) -m py_compile data/generation/*.py
