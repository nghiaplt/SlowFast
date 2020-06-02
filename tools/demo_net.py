import os
import sys
from time import time

import numpy as np
import cv2
import torch

from slowfast.utils import logging

from display.utils import display_boxes, display_text_label
from initialize.utils import init_model, init_params

from process.process_batch import process_frames_batch
from process.process_frame import get_processed_frame

logger = logging.get_logger(__name__)
np.random.seed(20)


def handle_end_frame(frame_provider):
    # If Source is string means video
    if isinstance(frame_provider.source, str):
        print("Process video done!")
        return True
    # when reaches the end frame, clear the buffer and continue to the next one.
    return False


def demo(cfg):
    model = init_model(cfg)
    frame_provider, frame_writer, seq_len, frames, pred_labels, s, frame_counter, palette, boxes, labels, object_predictor = init_params(
        cfg)

    for able_to_read, frame in frame_provider:
        if not able_to_read and handle_end_frame(frame_provider):
            break

        if len(frames) != seq_len:
            frame_processed = get_processed_frame(cfg, frame)

            if cfg.DETECTION.ENABLE and len(frames) == seq_len//2 - 1:
                mid_frame = frame

            frames.append(frame_processed)

        if len(frames) == seq_len:
            start = time()
            boxes, pred_labels = process_frames_batch(cfg, frames, mid_frame,
                                                      frame_provider, object_predictor, model, labels)
            # # option 1: remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            # option 2: empty the buffer
            frames = []
            s = time() - start
            print(
                'Detection + Classification for {} frames: {} seconds'.format(seq_len, s))

        if cfg.DETECTION.ENABLE and pred_labels and boxes.any():
            frame = display_boxes(cfg, boxes, pred_labels,
                                  frame, palette, labels)
        if not cfg.DETECTION.ENABLE:
            frame = display_text_label(cfg, frame, pred_labels)

        # Display prediction speed
        cv2.putText(frame, 'Speed: {:.2f}s'.format(s), (10, 25),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=.65, color=(0, 235, 0), thickness=2)

        frame_writer.write(frame)
        frame_counter += 1
        print('Writing frame {} ...'.format(frame_counter))
        # Display the frame
        # cv2.imshow('SlowFast', frame)
        # hit Esc to quit the demo.
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

    # frame_provider.clean()
