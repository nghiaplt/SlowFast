import cv2
from slowfast.datasets.cv2_transform import scale


def get_processed_frame(cfg, frame):
    frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)

    return frame_processed
