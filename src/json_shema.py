
STATE_START = "START"
STATE_EXPECT_KEY = "EXPECT_KEY"
STATE_IN_KEY = "IN_KEY"
STATE_EXPECT_COLON = "EXPECT_COLON"
STATE_EXPECT_VALUE = "EXPECT_VALUE"
STATE_IN_VALUE = "IN_VALUE"
STATE_DONE = "DONE"
STATE_EXPECT_COMMA_OR_CLOSE = "EXPECT_COMMA_OR_CLOSE"
STATE_IN_NUMBER_VALUE = "IN_NUMBER_VALUE"
STATE_IN_BOOL_VALUE = "IN_BOOL_VALUE"


class JSONSchemaTracker:
    def __init__(self, function_definitions: list[dict], vocabulary: dict) \
            -> None:
        self.functions = function_definitions
        self.vocab = vocabulary
        self.current_state = "START"
        self.generated_text = ""
        self.chosen_function = None
        self.current_key_buffer = ""
        self.current_value_buffer = ""
        self.keys_seen = set()

    def update_states(self, last_token: str):
        for char in last_token:
            self.generated_text += char

            if char == ' ' and self.current_state in (
                STATE_EXPECT_KEY, STATE_EXPECT_COLON,
                STATE_EXPECT_VALUE, STATE_EXPECT_COMMA_OR_CLOSE
            ):
                continue

            if self.current_state == STATE_START:
                if char == "{":
                    self.current_state = STATE_EXPECT_KEY

            elif self.current_state == STATE_EXPECT_KEY:
                if char == '"':
                    self.current_state = STATE_IN_KEY
                    self.current_key_buffer = ""

            elif self.current_state == STATE_IN_KEY:
                if char == '"':
                    self.current_state = STATE_EXPECT_COLON
                else:
                    self.current_key_buffer += char

            elif self.current_state == STATE_EXPECT_COLON:
                if char == ":":
                    self.current_state = STATE_EXPECT_VALUE

            elif self.current_state == STATE_EXPECT_VALUE:
                param_type = self._get_current_param_type()
                if param_type == "string" and char == '"':
                    self.current_state = STATE_IN_VALUE
                    self.current_value_buffer = ""
                elif param_type in ("number", "integer") and (
                    char.isdigit() or char == '-'
                        ):
                    self.current_state = STATE_IN_NUMBER_VALUE
                    self.current_value_buffer = char
                elif param_type == "boolean" and char in ('t', 'f'):
                    self.current_state = STATE_IN_BOOL_VALUE
                    self.current_value_buffer = char

            elif self.current_state == STATE_IN_VALUE:
                if char == '"':
                    if self.current_key_buffer == "name":
                        self.chosen_function = self.current_value_buffer

                    self.keys_seen.add(self.current_key_buffer)
                    self.current_state = STATE_EXPECT_COMMA_OR_CLOSE
                else:
                    self.current_value_buffer += char

            elif self.current_state == STATE_EXPECT_COMMA_OR_CLOSE:
                if char == ',':
                    self.current_state = STATE_EXPECT_KEY
                elif char == '}':
                    self.current_state = STATE_DONE

            elif self.current_state == STATE_IN_NUMBER_VALUE:
                if char.isdigit() or char == '.':
                    self.current_value_buffer += char
                else:
                    self.keys_seen.add(self.current_key_buffer)
                    if char == ',':
                        self.current_state = STATE_EXPECT_KEY
                    elif char == '}':
                        self.current_state = STATE_DONE
                    else:
                        self.current_state = STATE_EXPECT_COMMA_OR_CLOSE

            elif self.current_state == STATE_IN_BOOL_VALUE:
                target = "true" if self.current_value_buffer.startswith('t') \
                    else "false"
                self.current_value_buffer += char
                if self.current_value_buffer == target:
                    self.keys_seen.add(self.current_key_buffer)
                    self.current_state = STATE_EXPECT_COMMA_OR_CLOSE

    def get_allowed_token_ids(self) -> list[int]:
        allowed_ids = []
        valid_function_names = [func["name"] for func in self.functions]

        for token_str, token_id in self.vocab.items():
            if token_str == ' ' and self.current_state in (
                STATE_EXPECT_KEY, STATE_EXPECT_COLON,
                STATE_EXPECT_VALUE, STATE_EXPECT_COMMA_OR_CLOSE
            ):
                allowed_ids.append(token_id)
                continue

            if self.current_state == STATE_START:
                if token_str == "{":
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_IN_KEY:
                valid_keys = [
                    k for k in self._get_required_keys()
                    if k not in self.keys_seen
                    ]
                if token_str == '"':
                    if self.current_key_buffer in valid_keys:
                        allowed_ids.append(token_id)
                elif not any(
                        c in token_str for c in ['"', '{', '}', ':', ',']
                        ):
                    potential_key = self.current_key_buffer + token_str
                    if any(k.startswith(potential_key) for k in valid_keys):
                        allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_KEY:
                if token_str == '"':
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_COLON:
                if token_str == ":":
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_VALUE:
                param_type = self._get_current_param_type()
                if param_type == "string":
                    if token_str == '"':
                        allowed_ids.append(token_id)
                elif param_type == "number" or param_type == "integer":
                    if token_str.lstrip('-').isdigit():
                        allowed_ids.append(token_id)
                elif param_type == "boolean":
                    if token_str in (
                        't', 'f', 'tr', 'fa', 'tru', 'fal', 'true', 'false'
                            ):
                        allowed_ids.append(token_id)

            elif self.current_state == STATE_IN_VALUE:
                if self.current_key_buffer == "name":
                    if token_str == '"':
                        allowed_ids.append(token_id)
                    else:
                        potential_string = self.current_value_buffer \
                            + token_str
                        is_valid_prefix = any(
                            func.startswith(potential_string)
                            for func in valid_function_names
                            )
                        if is_valid_prefix:
                            allowed_ids.append(token_id)
                else:
                    MAX_STRING_LEN = 25
                    if len(self.current_value_buffer) >= MAX_STRING_LEN:
                        if token_str == '"':
                            allowed_ids.append(token_id)
                    else:
                        if token_str == '"':
                            allowed_ids.append(token_id)
                        elif not any(c in token_str for c in ['"']):
                            allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_COMMA_OR_CLOSE:
                remaining = set(self._get_required_keys()) - self.keys_seen
                if remaining:
                    if token_str == ',':
                        allowed_ids.append(token_id)
                else:
                    if token_str in (',', '}'):
                        allowed_ids.append(token_id)

            elif self.current_state == STATE_IN_NUMBER_VALUE:
                if token_str.isdigit() or token_str == '.':
                    allowed_ids.append(token_id)
                else:
                    remaining = set(self._get_required_keys()) \
                        - self.keys_seen - {self.current_key_buffer}
                    if remaining and token_str == ',':
                        allowed_ids.append(token_id)
                    elif not remaining and token_str == '}':
                        allowed_ids.append(token_id)

            elif self.current_state == STATE_IN_BOOL_VALUE:
                target = "true" if self.current_value_buffer.startswith('t') \
                    else "false"
                remaining = target[len(self.current_value_buffer):]
                if remaining.startswith(token_str):
                    allowed_ids.append(token_id)

                pass
        return allowed_ids

    def _get_current_param_type(self):
        if not self.chosen_function or self.current_key_buffer == "name":
            return "string"
        func_def = next(
            f for f in self.functions if f["name"] == self.chosen_function
            )
        props = func_def["parameters"]
        return props.get(self.current_key_buffer, {}).get("type", "string")

    def _get_required_keys(self):
        if not self.chosen_function:
            return ["name"]
        func_def = next(f for f in self.functions
                        if f["name"] == self.chosen_function)
        return ["name"] + list(func_def["parameters"].keys())
