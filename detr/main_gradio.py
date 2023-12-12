import gradio as gr
import supervision as sv
import os
from time import perf_counter

from detr_models import SimpleDetr, PanopticDetrResenet101, SimpleDetrOnnx, ONNX_DIR

IMAGES_DIR = os.path.abspath(os.curdir) + "/data/images"
ASSETS_DIR = os.path.abspath(os.curdir) + "/data/assets"
print("images:", IMAGES_DIR)


def run_inference(image, confidence, model_name, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, "loading model..")
    if not image:
        raise gr.Error("Provide image.")
    t0 = perf_counter()
    if model_name == "detr_simple_demo":
        model = SimpleDetr()
    elif model_name == "detr_resnet101_panoptic":
        model = PanopticDetrResenet101()
    elif model_name == "detr_simple_demo_onnx":
        if not os.path.exists(f"{ONNX_DIR}/detr_simple_demo_onnx.onnx"):
            raise gr.Error("ONNX model not found, please export it first!")
        model = SimpleDetrOnnx()
    t1 = perf_counter()
    progress(0.1, "Inference..")

    annotated_img = model.detect(image, confidence)
    t2 = perf_counter()
    return annotated_img, {"load_model": t1 - t0, "inference": t2 - t1}, None


def export_model(model_name, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, "Conversion..")
    t0 = perf_counter()
    if model_name == "detr_simple_demo":
        model = SimpleDetr()
    elif model_name == "detr_resnet101_panoptic":
        model = PanopticDetrResenet101()

    model_path = model.export()
    t1 = perf_counter()
    return model_path, {"export_time": t1 - t0}


with gr.Blocks() as demo:
    gr.Markdown("# DETR: Detection Transformer")
    # gr.Image(value=f"{ASSETS_DIR}/detr_architecture.png")
    with gr.Tab("Torch Inference"):
        with gr.Row():
            with gr.Column():
                img_file = gr.Image(type="pil")
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[
                        "detr_simple_demo",
                        "detr_resnet101_panoptic",
                    ],
                    value="detr_simple_demo",
                )

                conf = gr.Slider(label="Confidence", minimum=0, maximum=0.99, value=0.5)

                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
            with gr.Column():
                annotated_img = gr.Image(label="Annotated Image")
                speed = gr.JSON(label="speed")
        examples = gr.Examples(
            examples=[
                [path]
                for path in sv.list_files_with_extensions(
                    directory=IMAGES_DIR, extensions=["jpeg", "jpg", "png"]
                )
            ],
            inputs=[img_file],
        )
        start_btn.click(
            fn=run_inference,
            inputs=[img_file, conf, model_name],
            outputs=[annotated_img, speed],
        )
    with gr.Tab("ONNX Inference"):
        with gr.Row():
            with gr.Column():
                img_file = gr.Image(type="pil")
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[
                        "detr_simple_demo_onnx",
                    ],
                    value="detr_simple_demo_onnx",
                )
                conf = gr.Slider(label="Confidence", minimum=0, maximum=0.99, value=0.7)
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
            with gr.Column():
                annotated_img = gr.Image(label="Annotated Image")
                speed = gr.JSON(label="speed")
        examples = gr.Examples(
            examples=[
                [path]
                for path in sv.list_files_with_extensions(
                    directory=IMAGES_DIR, extensions=["jpeg", "jpg", "png"]
                )
            ],
            inputs=[img_file],
        )
        start_btn.click(
            fn=run_inference,
            inputs=[img_file, conf, model_name],
            outputs=[annotated_img, speed],
        )
    with gr.Tab("ONNX export"):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label="Model",
                    choices=[
                        "detr_simple_demo",
                        "detr_resnet101_panoptic",
                    ],
                    value="detr_simple_demo",
                )
                with gr.Row():
                    export_btn = gr.Button("Export", variant="primary")
            with gr.Column():
                onnx_file = gr.File()
                result = gr.JSON(label="result")
        export_btn.click(
            fn=export_model,
            inputs=[model_name],
            outputs=[onnx_file, result],
        )

if __name__ == "__main__":
    demo.queue(2).launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7000,
    )
