install:
	uv sync

run:
	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
clean:
	rm -rf src/__pycache__
	rm -rf llm_sdk/llm_sdk/__pycache__

lint:
	flake8 . --exclude llm_sdk,.venv
	mypy . --exclude llm_sdk,.venv --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 . --exclude llm_sdk,.venv
	mypy . --exclude llm_sdk,.venv --strict --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
