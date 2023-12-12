from functools import cache
import io
import itertools
import torch
import torchvision.transforms as T
import os
import numpy as np
import seaborn as sns
from torch import nn
from torchvision.models import resnet50
from panopticapi.utils import id2rgb, rgb2id
from supervision import Detections, BoxAnnotator, MaskAnnotator
import onnx
import onnxruntime
from PIL import Image
from pathlib import Path

torch.set_grad_enabled(False)


# https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=cfCcEYjg7y46

DETR_DEMO_WEIGHTS_URI = "https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth"
TORCH_HOME = os.path.abspath(os.curdir) + "/data/cache"
ONNX_DIR = os.path.abspath(os.curdir) + "/data/onnx"
os.environ["TORCH_HOME"] = TORCH_HOME

Path(TORCH_HOME).mkdir(exist_ok=True)
Path(ONNX_DIR).mkdir(exist_ok=True)


print("Torch home:", TORCH_HOME)


# standard PyTorch mean-std input image normalization


def normalize_img(image):
    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


def normalize_img_800_800(image):
    transform = T.Compose(
        [
            T.Resize((800, 800)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # propagate through the transformer
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
        ).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }


class SimpleDetr:
    @cache
    def __init__(self):
        self.model = DETRdemo(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(
            url=DETR_DEMO_WEIGHTS_URI,
            map_location="cpu",
            check_hash=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.box_annotator: BoxAnnotator = BoxAnnotator()

    def detect(self, image, conf):
        # mean-std normalize the input image (batch-size: 1)
        img = normalize_img(image)

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert (
            img.shape[-2] <= 1600 and img.shape[-1] <= 1600
        ), "demo model only supports images up to 1600 pixels on each side"

        # propagate through the model
        outputs = self.model(img)
        # keep only predictions with 0.7+ confidence
        scores = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = scores.max(-1).values > conf
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], image.size)
        probas = scores[keep]
        class_id = []
        confidence = []
        for prob in probas:
            cls_id = prob.argmax()
            c = prob[cls_id]
            class_id.append(int(cls_id))
            confidence.append(float(c))
        print(class_id, confidence)
        detections = Detections(
            xyxy=bboxes_scaled.cpu().detach().numpy(),
            class_id=np.array(class_id),
            confidence=np.array(confidence),
        )
        annotated = self.box_annotator.annotate(
            scene=np.array(image),
            skip_label=False,
            detections=detections,
            labels=[
                f"{CLASSES[cls_id]} {conf:.2f}"
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ],
        )
        return annotated

    def export(self):
        model_path = f"{ONNX_DIR}/detr_simple_demo_onnx.onnx"
        dummy_image = torch.ones(1, 3, 800, 800, device="cpu")
        input_names = ["inputs"]
        output_names = ["pred_logits", "pred_boxes"]
        torch.onnx.export(
            self.model,
            dummy_image,
            model_path,
            input_names=input_names,
            output_names=output_names,
            # dynamic_axes={input_names[0]: {0: "batch_size", 2: "height", 3: "width"}}, #!TODO
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            opset_version=14,
        )
        onnx_model = onnx.load(model_path)

        # Check the model
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(f"The model is invalid: {e}")
        else:
            print("The model is valid!")
        return model_path


class SimpleDetrOnnx:
    @cache
    def __init__(self):
        self.box_annotator: BoxAnnotator = BoxAnnotator()
        onnx_sess_opts = onnxruntime.SessionOptions()
        onnx_sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            # onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        onnx_sess_opts.enable_mem_pattern = True
        onnx_sess_opts.enable_cpu_mem_arena = True
        self.ort_session = onnxruntime.InferenceSession(
            f"{ONNX_DIR}/detr_simple_demo_onnx.onnx",
            sess_options=onnx_sess_opts,
            providers=[
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        self.classes = {}
        self.metadata = self.ort_session.get_modelmeta()
        self.providers = self.ort_session.get_providers()
        print(f"[OnnxRuntime] Providers:{self.providers}")
        print(
            f"[OnnxRuntime] medatadata: {self.metadata.custom_metadata_map} {type(self.metadata.custom_metadata_map)}"
        )

    def detect(self, image, conf):
        # dummy_image = np.ones((1, 3, 600, 800), dtype=np.float32)
        im = normalize_img_800_800(image).numpy()
        print("SHAPE", im.shape)
        ort_inputs = {self.ort_session.get_inputs()[0].name: im}
        outputs = self.ort_session.run(None, ort_inputs)
        pred_logits = torch.tensor(
            outputs[0]
        )  # conversion to torch for simplicity (softmax etc)
        pred_boxes = torch.tensor(outputs[1])
        scores = pred_logits.softmax(-1)[0, :, :-1]
        keep = scores.max(-1).values > conf
        bboxes_scaled = rescale_bboxes(pred_boxes[0, keep], image.size)
        probas = scores[keep]
        class_id = []
        confidence = []
        for prob in probas:
            cls_id = prob.argmax()
            c = prob[cls_id]
            class_id.append(int(cls_id))
            confidence.append(float(c))
        print(class_id, confidence)
        detections = Detections(
            xyxy=bboxes_scaled.cpu().detach().numpy(),
            class_id=np.array(class_id),
            confidence=np.array(confidence),
        )
        annotated = self.box_annotator.annotate(
            scene=np.array(image),
            skip_label=False,
            detections=detections,
            labels=[
                f"{CLASSES[cls_id]} {conf:.2f}"
                for cls_id, conf in zip(detections.class_id, detections.confidence)
            ],
        )
        return annotated


class PanopticDetrResenet101:
    @cache
    def __init__(self):
        self.model, self.postprocessor = torch.hub.load(
            "facebookresearch/detr",
            "detr_resnet101_panoptic",
            pretrained=True,
            return_postprocessor=True,
            num_classes=250,
        )
        self.model.eval()

    def detect(self, image, conf):
        # mean-std normalize the input image (batch-size: 1)
        img = normalize_img(image)

        outputs = self.model(img)
        result = self.postprocessor(
            outputs, torch.as_tensor(img.shape[-2:]).unsqueeze(0)
        )[0]
        print(result.keys())
        palette = itertools.cycle(sns.color_palette())

        # The segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8).copy()
        # We retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb2id(panoptic_seg)

        # Finally we color each mask individually
        panoptic_seg[:, :, :] = 0
        for id in range(panoptic_seg_id.max() + 1):
            panoptic_seg[panoptic_seg_id == id] = np.asarray(next(palette)) * 255
        return panoptic_seg

    def export(self):
        model_path = f"{ONNX_DIR}/detr_resnet101_panoptic.onnx"
        dummy_image = torch.ones(1, 3, 800, 800, device="cpu")
        input_names = ["inputs"]
        output_names = ["pred_logits", "pred_boxes", "pred_masks"]
        torch.onnx.export(
            self.model,
            dummy_image,
            model_path,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            opset_version=14,
        )
        onnx_model = onnx.load(model_path)

        # Check the model
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print(f"The model is invalid: {e}")
        else:
            print("The model is valid!")
        return model_path


# COCO classes
CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# model = SimpleDetr()
# model.export()
