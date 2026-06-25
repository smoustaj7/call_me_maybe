from .constraints import (
    get_valid_tokens_for_boolean,
    get_valid_tokens_for_function_name,
    get_valid_tokens_for_numbers,
    get_valid_tokens_for_string,
)
from .tokenizer import encode, decode


def select_function(
    prompt: str,
    functions: list[dict],
    model,
    vocab: dict[str, int],
    byte_encoder: dict,
    merges: dict,
    inv_vocab: dict,
    byte_decoder: dict,
) -> str:
    """generate the function name using constrained decoding."""
    general_prompt = "Given the following functions:\n"
    function_names = []
    for function in functions:
        general_prompt += f" -{function['name']}: {function['description']}\n"
        function_names.append(function["name"])
    general_prompt += f"User request: '{prompt}'\nFunction to call:\n"
    input_ids = encode(general_prompt, byte_encoder, merges, vocab)
    generated: list[int] = []
    while len(generated) < 50:
        logits = model.get_logits_from_input_ids(input_ids)
        generated_str = decode(generated, inv_vocab, byte_decoder) \
            if generated else ""
        if generated_str in function_names:
            break
        valid_tokens = get_valid_tokens_for_function_name(
            generated_str, function_names, vocab
        )
        valid_set = set(valid_tokens)
        for i in range(len(logits)):
            if i not in valid_set:
                logits[i] = -float('inf')
        next_token = logits.index(max(logits))
        if next_token == 198:
            break
        generated.append(next_token)
        input_ids.append(next_token)
    result = decode(generated, inv_vocab, byte_decoder) \
        if generated else ""
    return result


def extract_arguments(
    prompt: str,
    function: dict,
    model,
    vocab: dict[str, int],
    byte_encoder: dict,
    merges: dict,
    inv_vocab: dict,
    byte_decoder: dict,
) -> dict[str, float | bool | str]:
    """generate each argument separately using constrained decoding."""
    arguments: dict[str, float | bool | str] = {}
    number_tokens = set(get_valid_tokens_for_numbers(vocab)) | {198}
    string_tokens = set(get_valid_tokens_for_string(vocab))
    for param_name, param in function["parameters"].items():
        param_type = param["type"]
        if param_type == "string":
            context = (
                f"Function: {function['description']}\n"
                "You are extracting raw input values to call this function. "
                "The function itself performs any computation when it runs. "
                "Never compute or transform the result yourself — provide the "
                "value exactly as the function needs it as input.\n"
            )
            param_prompt = (
                context
                + "If the requested value already appears"
                "in the request, copy "
                "it exactly as written, preserving case."
                "If it does not appear "
                "and must be created from a description, derive it.\n\n"
                "Request: Reverse the string 'hello'\n"
                "s: \"hello\"\n\n"
                "Request: Greet shrek\n"
                "name: \"shrek\"\n\n"
                "Request: Match any sequence of digits in the text\n"
                "pattern: \"\\d+\"\n\n"
                "Request: Use asterisks to mark each vowel\n"
                "symbol: \"*\"\n\n"
                "Request: Swap the word 'up' with 'down' in the text\n"
                "pattern: \"up\"\n\n"
                "Request: Replace every space with a single dash\n"
                "replacement: \"-\"\n\n"
                f"Request: {prompt}\n"
                f"{param_name}: \""
            )
        else:
            if arguments:
                already = "\n".join(f"{k}={v}" for k, v in arguments.items())
                param_prompt = (
                    f"From: '{prompt}'\n"
                    f"{already}\n"
                    f"{param_name}="
                )
            else:
                param_prompt = (
                    f"From: '{prompt}'\n"
                    f"The value of {param_name} is:\n"
                    f"{param_name}="
                )
        input_ids = encode(param_prompt, byte_encoder, merges, vocab)
        generated: list[int] = []
        while len(generated) < 20:
            logits = model.get_logits_from_input_ids(input_ids)
            generated_str = decode(generated, inv_vocab, byte_decoder) \
                if generated else ""
            if param_type == "number":
                valid_set = number_tokens
            elif param_type == "string":
                valid_set = string_tokens
            elif param_type == "boolean":
                valid_tokens = get_valid_tokens_for_boolean(
                    generated_str, vocab
                )
                valid_set = set(valid_tokens)
            for i in range(len(logits)):
                if i not in valid_set:
                    logits[i] = -float('inf')
            next_token = logits.index(max(logits))
            if next_token == 198:
                break
            generated.append(next_token)
            input_ids.append(next_token)
            if param_type == "string":
                generated_str = decode(generated, inv_vocab, byte_decoder)
                if '"' in generated_str:
                    break
        if param_type == "number":
            decoded_val = decode(generated, inv_vocab, byte_decoder) \
                .rstrip('.')
            arguments[param_name] = float(decoded_val)
        elif param_type == "boolean":
            is_true = decode(generated, inv_vocab, byte_decoder) == "true"
            arguments[param_name] = True if is_true else False
        elif param_type == "string":
            raw_decoded = decode(generated, inv_vocab, byte_decoder)
            arguments[param_name] = raw_decoded.split('"')[0].strip()
    return arguments
