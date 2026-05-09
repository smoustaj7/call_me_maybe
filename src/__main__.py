import argparse
import json
from pathlib import Path
from sys import stderr
from pydantic import ValidationError
from .parsing import FunctionLibrary, parse_prompts
from llm_sdk import Small_LLM_Model


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Translate natural language prompts \
            into structured function calls."
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output results JSON file."
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(args.functions_definition, 'r') as f:
            func_data = json.load(f)
            definitions_obj = FunctionLibrary.model_validate(func_data)
            definitions = definitions_obj.model_dump()
            print(f"Loaded definitions from: {args.functions_definition}")
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        print(f"Error processing definitions: {e}", file=stderr)
        return

    try:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
            prompts = parse_prompts(input_data)
            print(f"Loaded prompts from: {args.input}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing input: {e}", file=stderr)
        return

    model = Small_LLM_Model()


if __name__ == "__main__":
    main()
