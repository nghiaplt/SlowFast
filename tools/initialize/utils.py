
import numpy as np
import torch

from slowfast.models import model_builder
from slowfast.utils import logging
from slowfast.utils import misc
import slowfast.utils.checkpoint as cu

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

logger = logging.get_logger(__name__)
np.random.seed(20)


def build_and_switch_demo_model(cfg):
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    model.eval()
    misc.log_model_info(model)

    return model


def load_checkpoint(cfg, model):
    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2="caffe2" in [
            cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
    )


def init_model(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)

    model = build_and_switch_demo_model(cfg)

    return model


def load_object_detector(cfg):
    # Load object detector from detectron2
    dtron2_cfg_file = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
    dtron2_cfg = get_cfg()
    dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
    dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
    dtron2_cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
    object_predictor = DefaultPredictor(dtron2_cfg)

    return object_predictor


def load_labels(cfg):
    if cfg.DETECTION.ENABLE:
        # Load the labels of AVA dataset
        with open(cfg.DEMO.LABEL_FILE_PATH) as f:
            labels = f.read().split('\n')[:-1]
    else:
        # Load the labels of Kinectics-400 dataset
        labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
        labels = labels_df['name'].values

    return labels


def init_params(cfg):
    if cfg.DETECTION.ENABLE:
        object_predictor = load_object_detector(cfg)

    labels = load_labels(cfg)
    palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
    boxes = []

    frame_provider = VideoReader(cfg)
    frame_writer = VideoWriter(
        cfg, frame_provider.display_width, frame_provider.display_height)

    seq_len = cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE
    frames = []
    pred_labels = []
    s = 0.
    frame_counter = 0

    return frame_provider, frame_writer, seq_len, frames, pred_labels, s, frame_counter, palette, boxes, labels, object_predictor
