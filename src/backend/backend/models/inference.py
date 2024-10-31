from backend.core.model_enum import ModelEnum
from pydantic import BaseModel, ConfigDict


class InferenceRequest(BaseModel):
    model_name: ModelEnum
    confidence: float = 0.5
    """Base64-encoded image"""
    image_base64: str

    model_config = ConfigDict(protected_namespaces="")


class InferenceResponse(BaseModel):
    """Base64-encoded annotated image"""
    annotated_image_base64: str
    metrics: dict
