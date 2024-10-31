from fastapi import APIRouter, HTTPException, status

from backend.services.model_list import ModelType, model_list

router = APIRouter(prefix="/model-list", tags=["model-list"])


@router.get("", status_code=status.HTTP_200_OK)
def get_models(modeltype: ModelType) -> list[str]:
    try:
        return model_list(modeltype)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
