*This project has been created as part of the 42 curriculum by smoustaj.*

# call me maybe

## Description

`call me maybe` is an introduction to **function calling in LLMs**, implemented from scratch without relying on any provider's built-in function-calling API. The goal is to take a small local language model (Qwen3-0.6B) and force it to reliably select the correct function and generate correctly-typed parameters for that function, given a natural language question — while guaranteeing that the output is always syntactically valid JSON.

The core idea explored in this project is **constrained decoding**: rather than letting the model generate freely and hoping the output parses, the program intervenes at every single token generation step, masking out any token that would break JSON syntax or violate the expected function-calling schema. This means the model is structurally incapable of producing malformed output — even if it sometimes picks the wrong function or hallucinates a parameter value.

The pipeline:
1. Load a library of available functions (name, description, parameters, return type) from a JSON file.
2. Load a list of natural language prompts to test against.
3. For each prompt, build an instruction prompt that lists the available functions and asks the model to answer with a JSON function call.
4. Generate the JSON response **token by token**, using a custom state machine to constrain which tokens are legal at each step.
5. Write all results to an output JSON file.

## Instructions

### Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) for dependency management and execution
- A working internet connection on first run (to download the `Qwen/Qwen3-0.6B` model weights from the Hugging Face Hub)

### Installation

```bash
git clone <repository-url>
cd call_me_maybe
uv sync
```

### Execution

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

| Flag | Description | Default |
|---|---|---|
| `--functions_definition` | Path to the JSON file describing the available functions | `data/input/functions_definition.json` |
| `--input` | Path to the JSON file containing the natural language test prompts | `data/input/function_calling_tests.json` |
| `--output` | Path where the generated function calls will be written | `data/output/function_calling_results.json` |

The first run will download and cache the model weights and tokenizer files from the Hugging Face Hub, which may take a moment. Setting an `HF_TOKEN` environment variable is optional but recommended to avoid rate limiting.

## Algorithm explanation

The project implements **grammar-constrained decoding** through a hand-written finite state machine, `JSONSchemaTracker`.

A language model normally generates text by repeatedly: (1) computing a probability distribution (logits) over its entire vocabulary for the next token, and (2) sampling or picking the highest-scoring token. Left unconstrained, nothing stops the model from generating malformed JSON, inventing a function that doesn't exist, or putting a string where a number was expected.

The generation loop instead does the following at every step:

1. **Get logits** — feed the current token sequence to the model and retrieve a score for every token in the vocabulary.
2. **Ask the tracker** — query `JSONSchemaTracker.get_allowed_token_ids()`, which inspects the tracker's current state (e.g. *"expecting a key"*, *"inside a string value"*, *"expecting a comma or closing brace"*) and returns only the token IDs that would keep the output both syntactically valid JSON **and** consistent with the function-calling schema.
3. **Mask** — every token *not* in that allowed set has its logit forced to `-inf`, making it mathematically impossible to be selected.
4. **Select** — take the `argmax` of the masked logits. The model's preference is preserved among the legal options; only illegal options are removed.
5. **Update** — decode the chosen token, feed it character-by-character into the tracker's `update_states()` method to advance the state machine, and append it to the running sequence.
6. **Repeat** until the tracker reaches a `DONE` state (the JSON object has been validly closed).

### The state machine

`JSONSchemaTracker` walks through states such as `START → EXPECT_KEY → IN_KEY → EXPECT_COLON → EXPECT_VALUE → IN_VALUE / IN_NUMBER_VALUE / IN_BOOL_VALUE → EXPECT_COMMA_OR_CLOSE → DONE`, processing the generated output one character at a time. On top of pure JSON-syntax enforcement, it layers **schema awareness**:

- The `"name"` field's value is restricted, character by character, to remain a valid *prefix* of one of the known function names — so the model cannot drift into hallucinating a function that doesn't exist.
- Once a function has been selected, subsequent keys are restricted to that function's declared parameters (tracked via a `keys_seen` set, so a key cannot be repeated and the object cannot be closed early while required keys remain).
- Each parameter's value is constrained to the correct JSON type (`string`, `number`/`integer`, `boolean`) as declared in the function's definition.

## Design decisions

- **Character-level state machine over a full parser.** Token boundaries from the model's tokenizer rarely line up with JSON syntax boundaries (a single token can be `","`, half a word, or a punctuation cluster). Processing the *generated text* character-by-character inside `update_states`, while still operating on whole tokens for masking, sidesteps a large class of tokenizer-alignment bugs.
- **Greedy decoding (argmax) instead of sampling.** Simpler to reason about and debug, and sufficient to demonstrate that constrained decoding guarantees structural validity regardless of the underlying sampling strategy.
- **Explicit `keys_seen` tracking with a sentinel for the top-level `name` field.** Because a function parameter can itself be named `name` (e.g. `fn_greet`'s `name` parameter), the top-level function-selector key and a same-named parameter key would otherwise collide in a single "seen keys" set. This was an actual bug encountered during development (see Challenges).
- **A capped maximum length for open-ended string values.** JSON syntax places no constraint on the *content* of a string, so without an upper bound a small model can ramble indefinitely inside a string value (observed concretely with regex-pattern parameters). A length cap forces the closing quote once exceeded, guaranteeing termination at the cost of occasionally truncating content.
- **A safety step cap on the overall generation loop**, independent of the string-length cap, as a last-resort guard against any unforeseen state-machine deadlock during testing.

## Performance analysis

- **Structural validity: effectively 100%.** Every test prompt produced output that parses successfully as JSON (`json.loads` never raised). This is the core guarantee the constrained decoding approach is designed to provide, and it held across all tested function signatures (string, number, and multi-parameter functions).
- **Function selection accuracy: 11/11 on the provided test set.** The model correctly identified the intended function for every test prompt, including prompts with no obvious keyword overlap with the function name.
- **Parameter extraction accuracy: high for simple types, weaker for open-ended generation.** Numeric and short string parameters (e.g. extracting `2` and `3` from "What is the sum of 2 and 3?", or `"hello"` from a reverse-string prompt) were extracted correctly and consistently. Parameters requiring the model to *generate* novel structured content — specifically regex patterns for `fn_substitute_string_with_regex` — were unreliable; the syntax was always valid JSON, but the regex content itself was sometimes semantically incorrect or malformed as a regular expression.
- **Speed.** Generation is necessarily slow: each output token requires one full forward pass through the model, and `get_allowed_token_ids()` iterates the entire vocabulary (~150k tokens) on every step to build the mask. This is the expected cost of constrained decoding implemented at this level, and is acceptable for a learning project on a 0.6B parameter model but would need vectorized masking for production use.
- **Reliability.** With the step cap and string-length cap in place, the generation loop is guaranteed to terminate on every input; no hangs were observed in the final version.

## Challenges faced

- **Tokenizer/character mismatch.** Tokens are not characters — feeding whole tokens into a character-oriented state machine without iterating each token's characters individually caused early state-transition bugs (e.g. multi-character tokens skipping states entirely).
- **Whitespace handling.** Real generated JSON includes spaces around colons and after commas (e.g. `{"name": "fn_add_numbers", ...}`), which the early version of the tracker did not account for, causing valid generations to be rejected as having no allowed next token.
- **Premature object closing.** An early version of `EXPECT_COMMA_OR_CLOSE` allowed `}` unconditionally once any value had been written, letting the model close the object before required parameters were generated. This was fixed by tracking which required keys had already been seen (`keys_seen`) and only allowing `}` once that set covers every required key for the chosen function.
- **The `name`-key collision bug.** Because `keys_seen` stored raw key strings, the top-level `"name"` field (function selector) and a parameter also called `name` (as in `fn_greet`) were treated as the same key. Once the function name had been written, the tracker incorrectly believed the `name` *parameter* had already been satisfied too, producing incomplete output (`{"name":"fn_greet"}` with the required `name` parameter missing). Fixed by tracking the top-level name field under a distinct internal sentinel, separate from any same-named parameter key.
- **Unbounded string generation.** JSON syntax does not constrain what characters may appear inside a string, so the tracker had nothing to say about *content* — only entry and exit. On open-ended parameters like a regex pattern, the model sometimes never produced a closing quote on its own, generating an ever-growing pattern. Solved pragmatically with a maximum string length that forces closure once exceeded.
- **`uv` environment / nested package import issues.** The local `llm_sdk` package was not always recognized by `uv`'s isolated environment despite being listed as a dependency, requiring an explicit editable install (`uv pip install -e llm_sdk/`) to resolve `ModuleNotFoundError: No module named 'torch'` despite torch being installed system-wide.

## Testing strategy

- **Incremental state-machine testing.** Before wiring the tracker into the actual generation loop, it was validated standalone: a hand-written character/token sequence representing a full, valid function call (e.g. `{"name":"get_weather","city":"London","days":3,"metric":true}`) was fed in step by step, asserting at each position that the expected next token was present in `get_allowed_token_ids()`, and asserting the final state was `DONE` with the correct function captured.
- **Progressive complexity.** Testing started with JSON-syntax enforcement alone (`{"name": "..."}`), then added schema-aware constraints (valid function names only), then added parameter type handling (string, number, boolean) one type at a time, each verified before moving to the next.
- **End-to-end testing on real prompts.** Once the tracker passed isolated tests, the full pipeline was run against the project's actual test prompt set, with verbose per-step logging of the tracker's state, number of allowed tokens, and generated text so far — this was instrumental in diagnosing the deadlock, premature-closing, and key-collision bugs described above.
- **JSON validity verification.** Every generated output is checked with `json.loads()` to confirm syntactic validity is actually achieved in practice, not just in theory.

## Example usage

Given the following function definition (excerpt from `data/input/functions_definition.json`):

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers together and return their sum.",
  "parameters": {
    "a": { "type": "number" },
    "b": { "type": "number" }
  },
  "returns": { "type": "number" }
}
```

And the prompt:

```
"What is the sum of 2 and 3?"
```

Running the program produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "result": "{\"name\":\"fn_add_numbers\",\"a\":2,\"b\":3}"
}
```

The `"result"` field is always valid, parseable JSON conforming to the function-calling schema, regardless of which function or parameter types are involved.

## Resources

### Classic references

- [JSON specification (ECMA-404)](https://www.ecma-international.org/publications-and-standards/standards/ecma-404/)
- [Hugging Face — Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling)
- [Guidance — constrained generation library](https://github.com/guidance-ai/guidance)
- [Outlines — structured generation library](https://github.com/dottxt-ai/outlines)
- Wikipedia — [Finite-state machine](https://en.wikipedia.org/wiki/Finite-state_machine)

### Use of AI

An AI assistant (Claude) was used as a guide throughout this project, in the following ways:

- **Conceptual explanations**: clarifying what constrained decoding is, how token-level masking works, and how a finite state machine applies to JSON generation, including a visual diagram of a simple FSM example unrelated to the project (a traffic light) to build intuition before applying the concept to JSON parsing.
- **Debugging support**: interpreting stack traces (e.g. `uv`/`torch` import errors, infinite-loop diagnosis via state-trace logging) and proposing diagnostic steps (such as adding per-step logging) rather than directly fixing the issue.

