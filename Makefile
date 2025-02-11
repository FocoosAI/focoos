.PHONY: venv test install install-gpu run-pre-commit .uv .pre-commit tox

.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

venv:
	@uv venv --python=python3.12

install: .uv .pre-commit
	@uv pip install -e ".[dev]" --no-cache-dir
	@pre-commit install

install-gpu: .uv .pre-commit
	@uv pip install -e ".[dev,cuda,tensorrt,torch]" --no-cache-dir
	@pre-commit install

lint:
	@ruff check ./focoos ./tests ./notebooks  --fix
	@ruff format ./focoos ./tests ./notebooks

run-pre-commit: .pre-commit
	@pre-commit run --all-files

test:
	@pytest -s --cov=focoos --cov-report="xml:tests/coverage.xml" --cov-report=html --junitxml=./tests/junit.xml   && rm -f .coverage

tox:
	tox

clean:
	@rm -rf build dist *.egg-info .tox .nox .coverage .coverage.* .cache .pytest_cache htmlcov */coverage.xml */junit.xml .venv
	@find . -type d -name "__pycache__" -exec rm -r {} +
