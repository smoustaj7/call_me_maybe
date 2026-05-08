from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, ValidationError, RootModel
from sys import stderr
import json


class ParameterDetail(BaseModel):
    type: str
    description: Optional[str] = None


class FunctionModel(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Union[ParameterDetail, Dict[str, Any]]]
    returns: Dict[str, Any]


class FunctionLibrary(RootModel):
    root: List[FunctionModel]


def parse_prompts(data: list) -> list:
    return [
        dic.get("prompt") for dic in data
        if isinstance(dic.get("prompt"), str)
    ]


def main():
    try:
        with open('data/input/function_calling_tests.json', 'r') as file1:
            data1 = json.load(file1)
            prompts = parse_prompts(data1)
    except KeyError as e:
        print(f"Error: invalid prompts - {e}", file=stderr)
        return
    except FileNotFoundError:
        print("Error: functions_calling_tests.json not found", file=stderr)
        return

    try:
        with open('data/input/functions_definition.json', 'r') as file2:
            data2 = json.load(file2)
            definitions_obj = FunctionLibrary.model_validate(data2)
            definitions = definitions_obj.model_dump()
            print(
                "Successfully validated all function defintions and prompts!"
                )
    except ValidationError as e:
        print(f"Error: wrong function definition\n{e.errors()}", file=stderr)
        return
    except FileNotFoundError:
        print("Error: functions_definition.json not found", file=stderr)
        return
    print(prompts)
    print()
    print(definitions)


if __name__ == "__main__":
    main()
