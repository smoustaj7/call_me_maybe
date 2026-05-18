
STATE_START = "START"
STATE_EXPECT_KEY = "EXPECT_KEY"
STATE_IN_KEY = "IN_KEY"
STATE_EXPECT_COLON = "EXPECT_COLON"
STATE_EXPECT_VALUE = "EXPECT_VALUE"
STATE_IN_VALUE = "IN_VALUE"
STATE_DONE = "DONE"


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

    def update_states(self, last_token: str):
        for char in last_token:
            self.generated_text += char

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
                if char == '"':
                    self.current_state = STATE_IN_VALUE
                    self.current_value_buffer = ""

            elif self.current_state == STATE_IN_VALUE:
                if char == '"':
                    if self.current_key_buffer == "name":
                        self.chosen_function = self.current_value_buffer

                    self.current_state = STATE_EXPECT_KEY
                else:
                    self.current_value_buffer += char

    def get_allowed_token_ids(self) -> list[int]:
        allowed_ids = []
        valid_function_names = [func["name"] for func in self.functions]

        for token_str, token_id in self.vocab.items():

            if self.current_state == STATE_START:
                if token_str == "{":
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_KEY:
                if token_str == '"':
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_COLON:
                if token_str == ":":
                    allowed_ids.append(token_id)

            elif self.current_state == STATE_EXPECT_VALUE:
                if token_str == '"':
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

                pass

        return allowed_ids
