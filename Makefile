all:

	uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
clean:
	rm -rf src/__pycache__
	rm -rf llm_sdk/llm_sdk/__pycache__