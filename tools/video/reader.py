import cv2


class VideoReader(object):

    def __init__(self, cfg):
        self.source = cfg.DEMO.DATA_SOURCE if cfg.DEMO.DATA_SOURCE != - \
            1 else cfg.DEMO.DATA_VIDEO
        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT

        try:  # OpenCV needs int to read from webcam
            self.source = int(self.source)
        except ValueError:
            pass

        self.cap = cv2.VideoCapture(self.source)

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))

    def __iter__(self):
        return self

    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            # reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None

        return was_read, frame

    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()
