import cv2


class VideoWriter(object):

    def __init__(self, cfg, out_width=0, out_height=0):
        # self._fourcc = VideoWriter_fourcc('M', 'J', 'P', 'G')
        self._fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        if out_width > 0 and out_height > 0:
            self._out = cv2.VideoWriter(
                cfg.DEMO.OUT_VIDEO, self._fourcc, cfg.DEMO.OUT_FRAME_RATE, (out_width, out_height))
        else:
            self._out = cv2.VideoWriter(cfg.DEMO.OUT_VIDEO, self._fourcc, cfg.DEMO.OUT_FRAME_RATE, (
                cfg.DEMO.OUT_FRAME_WIDTH, cfg.DEMO.OUT_FRAME_HEIGHT))

    def write(self, frame):
        self._out.write(frame)
