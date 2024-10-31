
from enum import StrEnum

from backend.core.model_enum import ModelEnum


class ModelType(StrEnum):
    INFERABLE = "inferable"
    EXPORTABLE = "exportable"


def model_list(model_type: ModelType) -> list[ModelEnum]:
    match model_type:
        case ModelType.INFERABLE:
            return ModelEnum.inferable()
        case ModelType.EXPORTABLE:
            return ModelEnum.exportable()
        case _:
            raise ValueError(f"ModelType not supported: {model_type}")