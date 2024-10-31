import logging
import pathlib as pl
from time import perf_counter

from backend.core.detr_models import PanopticDetrResenet101, SimpleDetr
from backend.core.model_enum import ModelEnum

logger = logging.getLogger(__name__)


def export(model_name: ModelEnum):
    t0 = perf_counter()

    logger.info("Model loading...")
    model = None
    match model_name:
        case ModelEnum.DETR_SIMPLE_DEMO:
            model = SimpleDetr()
        case ModelEnum.DETR_RESTNET101_PANOPTIC:
            model = PanopticDetrResenet101()
        case _:
            raise ValueError("Model not supported.")

    t1 = perf_counter()
    logger.info("Model exporting...")
    path_exported_model = model.export()
    t2 = perf_counter()
    return pl.Path(path_exported_model), {"load_model": t1 - t0, "export_model": t2 - t1}
