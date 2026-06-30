from .get_tokens import (
    get_valid_tokens_for_boolean,
    get_valid_tokens_for_function_name,
    get_valid_tokens_for_numbers,
    get_valid_tokens_for_string,
)
from .tokenizer import encode, decode
from llm_sdk import Small_LLM_Model


def select_function(
    prompt: str,
    functions: list[dict],
    model: Small_LLM_Model,
    vocab: dict[str, int],
    byte_encoder: dict,
    merges: dict,
    inv_vocab: dict,
    byte_decoder: dict,
) -> str:
    """generate the function name using constrained decoding."""
    general_prompt = (
        "You are an assistant that chooses a single function to call.\n"
        "Read the function list and the user request, then "
        "respond with exactly one "
        "function name from the list. Do not add any extra "
        "text or explanation.\n\n"
        "Available functions:\n"
    )
    function_names = []
    for function in functions:
        params = []
        for param_name, param in function["parameters"].items():
            params.append(f"{param_name}: {param['type']}")
        params_str = ", ".join(params)
        general_prompt += (
            f" - {function['name']}: {function['description']}\n"
            f"   parameters: {params_str}\n"
        )
        function_names.append(function["name"])
    general_prompt += (
        "\nChoose the best function from these exact names.\n"
        f"User request: {prompt}\n\n"
        "Function to call:\n"
    )
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
    model: Small_LLM_Model,
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
                "You are extracting a raw string argument "
                "for a function call.\n"
                "Do not compute or transform the answer. If the exact value "
                "appears in the request, copy it without "
                "changing case or formatting.\n"
                "If the value must be inferred from the request and "
                "function description, produce it directly.\n"
                "Return only the string content inside quotes.\n\n"
            )
            param_prompt = (
                context +
                f"Function description: {function['description']}\n"
                f"Parameter: {param_name}\n"
                f"Request: {prompt}\n"
                f"{param_name}: \""
            )
        else:
            if arguments:
                already = "\n".join(f"{k}={v}" for k, v in arguments.items())
                param_prompt = (
                    f"Function description: {function['description']}\n"
                    f"Request: {prompt}\n"
                    f"Already extracted: {already}\n"
                    f"Provide only the raw value "
                    f"for {param_name} ({param_type}):\n"
                    f"{param_name}="
                )
            else:
                param_prompt = (
                    f"Function description: {function['description']}\n"
                    f"Request: {prompt}\n"
                    f"Provide only the value"
                    f"for {param_name} ({param_type}):\n"
                    f"{param_name}="
                )
        input_ids = encode(param_prompt, byte_encoder, merges, vocab)
        generated: list[int] = []
        while len(generated) < 20:
            logits = model.get_logits_from_input_ids(input_ids)
            generated_str = decode(generated, inv_vocab, byte_decoder) \
                if generated else ""
            if param_type == "number" or param_type == "integer":
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
        elif param_type == "integer":
            decoded_val = decode(generated, inv_vocab, byte_decoder)
            arguments[param_name] = int(decoded_val) 
        elif param_type == "boolean":
            is_true = decode(generated, inv_vocab, byte_decoder) == "true"
            arguments[param_name] = True if is_true else False
        elif param_type == "string":
            raw_decoded = decode(generated, inv_vocab, byte_decoder)
            arguments[param_name] = raw_decoded.split('"')[0].strip()
    return arguments
