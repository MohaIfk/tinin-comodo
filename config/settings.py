import os
from pathlib import Path
import torch


class Settings:
    # GPU/CUDA Settings
    USE_GPU = True  # Set to False to force CPU usage even if GPU is available
    DEVICE = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"

    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 1920
    CAMERA_HEIGHT = 1080
    TARGET_FPS = 30

    # Detection
    MP_DETECTION_CONFIDENCE = 0.7
    MP_FACE_DETECTION_CONFIDENCE = 0.7
    MP_FACE_TRACKING_CONFIDENCE = 0.5
    MP_MAX_NUM_FACES = 3  # Limited to support 2-3 people
    YOLO_MODEL_PATH = "yolov8s.pt"

    # Entity validation
    MIN_ENTITY_SIZE = 0.05  # Minimum size as a fraction of frame dimensions
    MAX_ENTITY_SIZE = 0.95  # Maximum size as a fraction of frame dimensions
    VALID_POSITION_MARGIN = 0.0  # Margin from frame edges as a fraction of frame dimensions

    # Avatar API
    AVATAR_API_URL = "https://api.readyplayerme.com/v1/avatars"
    AVATAR_API_KEY = os.getenv("AVATAR_API_KEY", "")  # Use environment variable for security
    AVATAR_API_TIMEOUT = 10  # 10 seconds timeout for API requests

    @property
    def has_valid_api_key(self):
        """Check if a valid API key is available"""
        return bool(self.AVATAR_API_KEY)

    # TouchDesigner
    TD_IP = "127.0.0.1"
    TD_OSC_PORT = 7000

    # Motion Analysis
    MOTION_HISTORY_LENGTH = 30  # Number of frames to keep in history for motion analysis

    # Debug
    DEBUG_MODE = True
