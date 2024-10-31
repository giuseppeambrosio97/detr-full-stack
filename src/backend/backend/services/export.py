import os

import backend.core.export as core_export
from backend.core.model_enum import ModelEnum
from backend.models.export import ExportResponse


def export(model_name: ModelEnum) -> ExportResponse:
    export_path, metrics = core_export.export(model_name)
    file_size = os.path.getsize(export_path) // (1024 * 1024)
    return ExportResponse(
        file_name=export_path.name,
        file_size=file_size,
        metrics=metrics,
    )
