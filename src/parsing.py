from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, RootModel, ValidationError


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


class PromptModel(BaseModel):
    prompt: str


class PromptList(RootModel):
    root: List[PromptModel]


def parse_prompts(data: list) -> list:
    try:
        PromptList.model_validate(data)
    except ValidationError as e:
        print(f"Error processing prompts: {e}")
        return []
    return [
        dic.get("prompt") for dic in data
        if isinstance(dic.get("prompt"), str)
    ]
