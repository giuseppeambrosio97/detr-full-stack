import backend.core.inference as core_inference
import backend.helpers.image as image_helpers
from backend import IMAGES_DATA_LOCATION
from backend.models.inference import InferenceRequest, InferenceResponse


def inference(request: InferenceRequest) -> InferenceResponse:
    annotated_image_pil, metrics = core_inference.inference(
        model_name=request.model_name,
        confidence=request.confidence,
        image=image_helpers.decode_image(request.image_base64),
    )
    return InferenceResponse(
        annotated_image_base64=image_helpers.encode_image(annotated_image_pil),
        metrics=metrics,
    )


if __name__ == "__main__":
    from PIL import Image

    image_file_name = "000000039769.jpg"
    image_path = IMAGES_DATA_LOCATION / image_file_name
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to open image: {e}")
        exit(1)

    request = InferenceRequest(
        model_name="detr_simple_demo",
        confidence=0.5,
        image_base64=image_helpers.encode_image(image),
    )
    resp = inference(request)

    print(resp)
