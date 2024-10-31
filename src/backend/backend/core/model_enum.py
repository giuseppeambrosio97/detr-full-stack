from enum import StrEnum


class ModelEnum(StrEnum):
    DETR_SIMPLE_DEMO = "detr_simple_demo"
    DETR_RESTNET101_PANOPTIC = "detr_resnet101_panoptic"
    DETR_SIMPLE_DEMO_ONNX = "detr_simple_demo_onnx"

    @staticmethod
    def inferable():
        return list(ModelEnum)
    
    @staticmethod
    def exportable():
        return [ModelEnum.DETR_SIMPLE_DEMO, ModelEnum.DETR_RESTNET101_PANOPTIC]