import argparse
import json
from pathlib import Path
from sys import stderr
from pydantic import ValidationError
from .parsing import FunctionLibrary, parse_prompts
from .json_generation import select_function, extract_arguments
from .tokenizer import bytes_to_unicode, load_merges
from llm_sdk import Small_LLM_Model
import os
from pyfiglet import Figlet


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
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier to use for generation."
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

    model = Small_LLM_Model(model_name=args.model)
    try:
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            print(f"Loaded vocab from: {vocab_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing vocabulary file: {e}", file=stderr)
        return

    merges = load_merges(model.get_path_to_merges_file())
    byte_encoder = bytes_to_unicode()
    inv_vocab = {v: k for k, v in vocab.items()}
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    functions_map = {f["name"]: f for f in definitions}
    results = []

    os.system("clear")
    f = Figlet(font='banner3-D')
    print(f.renderText('CALL    ME    MAYBE    !'))
    for prompt_str in prompts:
        function_name = select_function(
            prompt_str, definitions, model, vocab,
            byte_encoder, merges, inv_vocab, byte_decoder
        )
        selected = functions_map.get(function_name)
        if selected is None:
            print(
                f"prompt: {prompt_str} - function not found, "
                "skipping prompt"
            )
            continue
        arguments = extract_arguments(
            prompt_str, selected, model, vocab,
            byte_encoder, merges, inv_vocab, byte_decoder
        )
        results.append({
            "prompt": prompt_str,
            "name": function_name,
            "parameters": arguments
        })
        print("\n" + "➖" * 60)
        print(f"Prompt: {prompt_str}")
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        print("➖" * 60)
        print("\n")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
