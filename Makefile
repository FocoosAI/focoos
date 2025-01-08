.PHONY: test install install-dev install-pre-commit run-pre-commit .uv .pre-commit tox

.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

install: .uv .pre-commit
	@uv venv
	@uv pip install -e ".[cpu,dev]"
	@pre-commit install

install-gpu: .uv .pre-commit
	@uv venv
	@uv pip install -e ".[dev,gpu]"
	@pre-commit install

lint:
	@isort ./focoos ./tests  --profile=black
	@black ./focoos ./tests

run-pre-commit: .pre-commit
	@pre-commit run --all-files

test:
	@pytest -s --cov=focoos --cov-report="xml:tests/coverage.xml" --cov-report=html --junitxml=./tests/junit.xml   && rm -f .coverage

tox:
	tox

clean:
	@rm -rf build dist *.egg-info .tox .nox .coverage .coverage.* .cache .pytest_cache htmlcov */coverage.xml */junit.xml .venv
	@find . -type d -name "__pycache__" -exec rm -r {} +
