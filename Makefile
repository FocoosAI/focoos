.PHONY: test install install-dev install-pre-commit run-pre-commit

install:
	@pip install . --no-cache-dir
install-dev:
	@pip install -e ".[dev]" --no-cache-dir

install-pre-commit:
	@pre-commit install
lint:
	@isort ./focoos ./tests  --profile=black
	@black ./focoos ./tests
run-pre-commit:
	@pre-commit run --all-files
test:
	@pytest -s --cov=focoos --cov-report="xml:tests/coverage.xml" --cov-report=html --junitxml=./tests/junit.xml   && rm -f .coverage
clean:
	@rm -rf build dist *.egg-info .tox .nox .coverage .coverage.* .cache .pytest_cache htmlcov
	@find . -type d -name "__pycache__" -exec rm -r {} +
