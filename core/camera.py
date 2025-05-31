import cv2
from typing import Optional, Tuple
import numpy as np
from utils.logger import app_logger as logger


class CameraManager:
    def __init__(self, settings):
        self.settings = settings
        self.cap = cv2.VideoCapture(settings.CAMERA_INDEX)
        self.setup_camera(settings)

        # Check if GPU acceleration is available for OpenCV
        self.use_gpu = settings.USE_GPU and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            # Initialize GPU-specific resources
            self.gpu_frame = cv2.cuda_GpuMat()
            logger.info("Camera: Using GPU acceleration for frame preprocessing")
        else:
            if settings.USE_GPU:
                logger.warning("Camera: GPU acceleration requested but OpenCV CUDA support not available")
            else:
                logger.info("Camera: Using CPU for frame preprocessing (GPU disabled in settings)")

    def setup_camera(self, settings):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, settings.TARGET_FPS)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()

        if not ret:
            return None

        # Apply GPU-accelerated preprocessing if available
        if self.use_gpu:
            try:
                # Upload frame to GPU
                self.gpu_frame.upload(frame)

                # Apply GPU-accelerated operations
                # Example: Gaussian blur for noise reduction
                gpu_blurred = cv2.cuda.createGaussianFilter(
                    cv2.CV_8UC3, cv2.CV_8UC3, (3, 3), 0)
                gpu_blurred = gpu_blurred.apply(self.gpu_frame)

                # Download result back to CPU
                result = gpu_blurred.download()
                return result
            except cv2.error as e:
                # Fallback to CPU if GPU processing fails
                logger.warning(f"GPU processing error: {e}. Falling back to CPU.")
                return frame

        return frame

    def release(self):
        self.cap.release()
