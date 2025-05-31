import cv2
import time
from core.camera import CameraManager
from core.detector import EntityDetector
from config.settings import Settings
from utils.logger import app_logger as logger

def test_webcam_detection():
    """Test webcam capture and MediaPipe face/pose detection"""
    logger.info("Starting webcam detection test")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize components
    try:
        logger.info("Initializing camera and detector...")
        camera = CameraManager(settings)
        detector = EntityDetector(settings)
        
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {str(e)}")
        return False
    
    # Process frames from webcam
    try:
        logger.info("Processing frames from webcam...")
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame, skipping")
                continue
                
            # Detect entities
            entities = detector.process_frame(frame)
            logger.info(f"Frame {frame_count+1}: Detected {len(entities)} entities")
            
            # Display annotated frame
            debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
            cv2.putText(debug_frame, 
                        f"FPS: {frame_count / (time.time() - start_time):.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, 
                        f"Entities: {len(entities)}", 
                        (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            cv2.imshow('Webcam Detection Test', debug_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                logger.info("ESC key pressed, exiting")
                break
                
            frame_count += 1
            
        logger.info(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False
    finally:
        # Clean up
        logger.info("Cleaning up resources")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    success = test_webcam_detection()
    if success:
        logger.info("Webcam detection test completed successfully")
    else:
        logger.error("Webcam detection test failed")