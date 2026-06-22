import argparse
import json
from pathlib import Path
from sys import stderr
from pydantic import ValidationError
from .parsing import FunctionLibrary, parse_prompts
from llm_sdk import Small_LLM_Model
from .tockenizer import encode, bytes_to_unicode, load_merges, decode
from .json_shema import JSONSchemaTracker, STATE_DONE


def llm_prompt(func_def: list[dict], user_prompt: str) -> str:
    base_instruction = (
        "You are an AI assistant that selects the best "
        "function to answer a user's question.\n"
        "You must choose from the following available "
        "functions and provide your answer as a JSON object.\n\n"
    )
    formatted_functions = "Available Functions:\n" + \
        json.dumps(func_def, indent=2)
    question_section = f"\n\nUser Question: {user_prompt}\n"
    json_priming = "Answer:\n{"
    final_llm_input = base_instruction + formatted_functions + \
        question_section + json_priming
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
    merges = load_merges(
        "/goinfre/smoustaj/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
        "snapshots/c1899de289a04d12100db370d81485cdf75e47ca/merges.txt"
    )
    inv_vocab = {v: k for k, v in vocab.items()}
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    results = []
    byte_encoder = bytes_to_unicode()

    for prompt_str in prompts:
        prompt = llm_prompt(definitions, prompt_str)
        input_ids = encode(prompt, byte_encoder, merges, vocab)
        prompt_length = len(input_ids)
        tracker = JSONSchemaTracker(definitions, vocab)
        tracker.update_states("{")

        max_steps = 100
        steps = 0
        while tracker.current_state != STATE_DONE and steps < max_steps:
            logits = model.get_logits_from_input_ids(input_ids)
            allowed_ids = set(tracker.get_allowed_token_ids())
            print(f"state: {tracker.current_state} | allowed: ", end="")
            print(
                f"{len(allowed_ids)} | generated: {tracker.generated_text!r}"
            )
            masked_logits = [
                l if i in allowed_ids else float('-inf')
                for i, l in enumerate(logits)
            ]
            next_id = max(
                range(len(masked_logits)), key=lambda i: masked_logits[i]
                )
            next_token = decode([next_id], inv_vocab, byte_decoder)
            tracker.update_states(next_token)
            input_ids.append(next_id)
            steps += 1

        generated_ids = input_ids[prompt_length:]
        generated_json = "{" + decode(generated_ids, inv_vocab, byte_decoder)

        results.append({
            "prompt": prompt_str,
            "result": generated_json
        })

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
