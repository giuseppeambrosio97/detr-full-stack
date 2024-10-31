from fastapi import APIRouter, HTTPException, status

import backend.services.inference as inference_svc
from backend import IMAGES_DATA_LOCATION
from backend.core.exceptions import ModelNotExportedError
from backend.helpers.image import ImageDecodeError, ImageEncodeError
from backend.models.inference import InferenceRequest, InferenceResponse

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("", status_code=status.HTTP_200_OK, response_model=InferenceResponse)
async def post_inference(request: InferenceRequest) -> InferenceResponse:
    try:
        return inference_svc.inference(request)
    except (ImageDecodeError, ImageEncodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except ModelNotExportedError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_message": str(exc),
                "export_required": True,
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


@router.get("/examples", status_code=status.HTTP_200_OK)
async def get_examples() -> list[str]:
    return [file.name for file in IMAGES_DATA_LOCATION.iterdir() if file.is_file()]
