from typing import List, Dict, Union, Any, Optional
from pydantic import BaseModel, RootModel


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
