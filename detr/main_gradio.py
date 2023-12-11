import gradio as gr
import supervision as sv
import os
from detr import SimpleDetr, PanopticDetrResenet101

ASSETS_DIR = os.path.abspath(os.curdir) + "/data/assets"

print("Assets:", ASSETS_DIR)


def run_inference(image, confidence, model_name, progress=gr.Progress(track_tqdm=True)):
    progress(0.1, "loading model..")

    if model_name == "detr_demo_boxes":
        model = SimpleDetr()
    else:
        model = PanopticDetrResenet101()
    progress(0.1, "Inference..")

    annotated_img = model.detect(image, confidence)
    return annotated_img, None, None


with gr.Blocks() as inference_gradio:
    gr.Markdown("# DETR inference")
    with gr.Row():
        with gr.Column():
            img_file = gr.Image(type="pil")
            # with gr.Row():
            model_name = gr.Dropdown(
                label="Model",
                scale=3,
                choices=["detr_demo_boxes", "detr_resnet101_panoptic"],
                value="detr_demo_boxes",
            )

            conf = gr.Slider(label="Confidence", minimum=0, maximum=0.99, value=0.5)

            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")

        with gr.Column():
            annotated_img = gr.Image(label="Annotated Image")
            speed = gr.JSON(label="speed")
            json_out = gr.JSON(label="output")
    examples = gr.Examples(
        examples=[
            [path]
            for path in sv.list_files_with_extensions(
                directory=ASSETS_DIR, extensions=["jpeg", "jpg"]
            )
        ],
        inputs=[img_file],
    )
    start_btn.click(
        fn=run_inference,
        inputs=[img_file, conf, model_name],
        outputs=[annotated_img, speed, json_out],
    )

if __name__ == "__main__":
    inference_gradio.queue(2).launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7000,
    )
