from fastapi import APIRouter, HTTPException, status

import backend.services.export as export_svc
from backend.core.model_enum import ModelEnum
from backend.models.export import ExportResponse

router = APIRouter(prefix="/export", tags=["export"])


@router.get("", status_code=status.HTTP_200_OK, response_model=ExportResponse)
async def get_export(model_name: ModelEnum) -> ExportResponse:
    try:
        return export_svc.export(model_name)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )
