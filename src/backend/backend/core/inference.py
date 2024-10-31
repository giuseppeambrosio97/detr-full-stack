import logging
import os
from time import perf_counter

from backend.core.exceptions import ModelNotExportedError
import numpy as np
from PIL import Image

from backend import IMAGES_DATA_LOCATION
from backend.core.detr_models import (
    ONNX_DIR,
    PanopticDetrResenet101,
    SimpleDetr,
    SimpleDetrOnnx,
)

logger = logging.getLogger(__name__)    


def inference(model_name: str, confidence: float, image: Image):
    if not image:
        raise ValueError("Image must be specified.")

    t0 = perf_counter()

    logger.info("Model loading...")

    model = None
    match model_name:
        case "detr_simple_demo":
            model = SimpleDetr()
        case "detr_resnet101_panoptic":
            model = PanopticDetrResenet101()
        case "detr_simple_demo_onnx":
            if not os.path.exists(f"{ONNX_DIR}/detr_simple_demo_onnx.onnx"):
                raise ModelNotExportedError(model_name="detr_simple_demo_onnx")
            model = SimpleDetrOnnx()
        case _:
            raise ValueError("Model not supported.")
    t1 = perf_counter()
    logger.info("Model inference...")
    annotated_image_ndarray = model.detect(image, confidence)
    annotated_image_pil = Image.fromarray(np.uint8(annotated_image_ndarray))
    t2 = perf_counter()
    return annotated_image_pil, {"load_model": t1 - t0, "inference": t2 - t1}


if __name__ == "__main__":
    image_file_name = "000000039769.jpg"
    image_path = IMAGES_DATA_LOCATION / image_file_name

    try:
        image = Image.open(image_path)
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        exit(1)

    model_name = "detr_resnet101_panoptic"
    confidence_threshold = 0.5

    # Call the inference function
    try:
        annotated_image_pil, timing_info = inference(
            model_name, confidence_threshold, image
        )
        logger.info("Inference completed successfully.")
        logger.info(f"Timing info: {timing_info}")

        # Optionally, save or display the annotated image
        annotated_image_pil.show()  # Display the image
        annotated_image_pil.save("output.jpg")  # Save the annotated image to a file

    except ValueError as e:
        logger.error(f"Error during inference: {e}")
