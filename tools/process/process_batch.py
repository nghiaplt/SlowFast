import torch
import numpy as np


import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.datasets import cv2_transform


logger = logging.get_logger(__name__)
np.random.seed(20)


def get_person_boxes(cfg, object_predictor, mid_frame, frame_provider):
    outputs = object_predictor(mid_frame)
    fields = outputs["instances"]._fields
    pred_classes = fields["pred_classes"]
    selection_mask = pred_classes == 0
    # acquire person boxes
    pred_classes = pred_classes[selection_mask]
    pred_boxes = fields["pred_boxes"].tensor[selection_mask]
    scores = fields["scores"][selection_mask]
    boxes = cv2_transform.scale_boxes(cfg.DATA.TEST_CROP_SIZE,
                                      pred_boxes,
                                      frame_provider.display_height,
                                      frame_provider.display_width)
    boxes = torch.cat(
        [torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1
    )

    return boxes, scores


def extract_slow_fast_path_from_frames(cfg, frames):
    inputs = torch.as_tensor(np.array(frames)).float()

    inputs = inputs / 255.0
    # Perform color normalization.
    inputs = inputs - torch.tensor(cfg.DATA.MEAN)
    inputs = inputs / torch.tensor(cfg.DATA.STD)
    # T H W C -> C T H W.
    inputs = inputs.permute(3, 0, 1, 2)

    # 1 C T H W.
    inputs = inputs.unsqueeze(0)

    # Sample frames for the fast pathway.
    index = torch.linspace(
        0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
    fast_pathway = torch.index_select(inputs, 2, index)
    # logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))

    # Sample frames for the slow pathway.
    index = torch.linspace(0, fast_pathway.shape[2] - 1,
                           fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
    slow_pathway = torch.index_select(fast_pathway, 2, index)

    return slow_pathway, fast_pathway


def process_frames_batch(cfg, frames, mid_frame, frame_provider, object_predictor, model, labels):
    if cfg.DETECTION.ENABLE:
        boxes, scores = get_person_boxes(
            cfg, object_predictor, mid_frame, frame_provider)

    slow_pathway, fast_pathway = extract_slow_fast_path_from_frames(
        cfg, frames)

    # logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
    inputs = [slow_pathway, fast_pathway]

    # Transfer the data to the current GPU device.
    if isinstance(inputs, (list,)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
    else:
        inputs = inputs.cuda(non_blocking=True)

    # Perform the forward pass.
    if cfg.DETECTION.ENABLE:
        # When there is nothing in the scene,
        #   use a dummy variable to disable all computations below.
        if not len(boxes):
            preds = torch.tensor([])
        else:
            preds = model(inputs, boxes)
    else:
        preds = model(inputs)

    # Gather all the predictions across all the devices to perform ensemble.
    if cfg.NUM_GPUS > 1:
        preds = du.all_gather(preds)[0]

    if cfg.DETECTION.ENABLE:
        # This post processing was intendedly assigned to the cpu since my laptop GPU
        #   RTX 2080 runs out of its memory, if your GPU is more powerful, I'd recommend
        #   to change this section to make CUDA does the processing.
        preds = preds.cpu().detach().numpy()
        pred_masks = preds > .1
        label_ids = [np.nonzero(pred_mask)[0]
                     for pred_mask in pred_masks]
        pred_labels = [
            [labels[label_id] for label_id in perbox_label_ids]
            for perbox_label_ids in label_ids
        ]
        # I'm unsure how to detectron2 rescales boxes to image original size, so I use
        #   input boxes of slowfast and rescale back it instead, it's safer and even if boxes
        #   was not rescaled by cv2_transform.rescale_boxes, it still works.
        boxes = boxes.cpu().detach().numpy()
        ratio = np.min(
            [frame_provider.display_height, frame_provider.display_width]
        ) / cfg.DATA.TEST_CROP_SIZE
        boxes = boxes[:, 1:] * ratio
    else:
        # Option 1: single label inference selected from the highest probability entry.
        # label_id = preds.argmax(-1).cpu()
        # pred_label = labels[label_id]
        # Option 2: multi-label inferencing selected from probability entries > threshold
        label_ids = torch.nonzero(
            preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
        pred_labels = labels[label_ids]
        logger.info(pred_labels)
        if not list(pred_labels):
            pred_labels = ['Unknown']

    return boxes, pred_labels
