from pydantic import BaseModel


class ExportResponse(BaseModel):
    file_name: str
    """File size in MB."""
    file_size: int
    metrics: dict
