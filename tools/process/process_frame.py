import cv2
from slowfast.datasets.cv2_transform import scale


def get_processed_frame(cfg, frame, frames, seq_len):
    mid_frame = None
    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
    if cfg.DETECTION.ENABLE and len(frames) == seq_len//2 - 1:
        mid_frame = frame

    return frame_processed, mid_frame
