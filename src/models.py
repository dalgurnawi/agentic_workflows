from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

class ActionEnum(str, Enum):
    create = 'create'
    write = 'write'
    read = 'read'


class ToolCall(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    path: str = Field(description='Name of the file')
    action: ActionEnum = Field(description='Content to save inside a file.')
    content: str = Field(description='Arguments to pass on the command.')

class ResponseModel(BaseModel):
    tool_calls: list[ToolCall]