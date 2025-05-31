import cv2
from typing import Optional, Tuple
import numpy as np


class CameraManager:
    def __init__(self, settings):
        self.cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self.setup_camera(settings)

    def setup_camera(self, settings):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()