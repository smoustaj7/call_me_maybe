import argparse
import json
from pathlib import Path
from sys import stderr
from pydantic import ValidationError
from .parsing import FunctionLibrary, parse_prompts
from llm_sdk import Small_LLM_Model


def llm_prompt(func_def: list[dict], user_prompt: str) -> str:
    base_instruction = (
        "You are an AI assistant that selects the best function to answer a user's question.\n"
        "You must choose from the following available functions and provide your answer as a JSON object.\n\n"
    )
    formatted_functions = "Available Functions:\n" + json.dumps(func_def, indent=2)
    question_section = f"\n\nUser Question: {user_prompt}\n"
    json_priming = "Answer:\n{"
    final_llm_input = base_instruction + formatted_functions + question_section + json_priming
    return final_llm_input

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
    try:
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            print(f"Loaded vocab from: {vocab_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing vocabulary file: {e}", file=stderr)
        return
    print(definitions)



if __name__ == "__main__":
    main()
