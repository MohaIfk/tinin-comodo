import torch
import cv2
import numpy as np
from config.settings import Settings
from core.detector import EntityDetector
from core.camera import CameraManager
from utils.logger import app_logger as logger

def test_gpu_acceleration():
    """Test GPU acceleration for the 3DMesh project."""
    logger.info("=== GPU Acceleration Test ===")
    
    # Check CUDA availability
    logger.info(f"CUDA is available: {torch.cuda.is_available()}")
    
    # Initialize settings
    settings = Settings()
    logger.info(f"Using device: {settings.DEVICE}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        if settings.DEVICE == "cpu":
            logger.warning("GPU is available but not being used (USE_GPU is set to False)")
    else:
        logger.warning("CUDA is not available, using CPU only")
    
    # Check OpenCV CUDA support
    opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    logger.info(f"OpenCV CUDA support: {opencv_cuda}")
    
    # Test camera with GPU acceleration
    logger.info("Testing camera with GPU acceleration...")
    try:
        camera = CameraManager(settings)
        # Get a frame to test GPU preprocessing
        frame = camera.get_frame()
        if frame is not None:
            logger.info(f"Camera frame captured: {frame.shape}")
            logger.info(f"Camera using GPU: {camera.use_gpu}")
        else:
            logger.error("Failed to capture camera frame")
    except Exception as e:
        logger.error(f"Camera test failed: {e}")
    finally:
        if 'camera' in locals():
            camera.release()
    
    # Test YOLO with GPU acceleration
    logger.info("Testing YOLO with GPU acceleration...")
    try:
        detector = EntityDetector(settings)
        # Create a test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        # Process the test image
        entities = detector.process_frame(test_image)
        logger.info(f"YOLO processed test image, detected {len(entities)} entities")
        logger.info(f"YOLO model device: {detector.animal_detector.device}")
    except Exception as e:
        logger.error(f"YOLO test failed: {e}")
    
    logger.info("=== GPU Acceleration Test Complete ===")

if __name__ == "__main__":
    test_gpu_acceleration()