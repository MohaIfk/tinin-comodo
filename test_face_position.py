import cv2
import time
from config.settings import Settings
from core.detector import EntityDetector
from utils.logger import app_logger as logger

def test_face_position():
    """
    Test the improved face position and limited multi-person detection.
    
    This script tests the changes made to address the following issues:
    1. Face position shift issue
    2. Limitation of the number of people detected to 2-3
    
    The script displays the detected humans and faces with annotations
    to allow visual verification that the issues have been resolved.
    """
    logger.info("=== Face Position and Limited Multi-Person Detection Test ===")
    
    # Initialize settings
    settings = Settings()
    logger.info(f"Maximum number of faces: {settings.MP_MAX_NUM_FACES}")
    
    # Initialize detector
    detector = EntityDetector(settings)
    
    # Initialize camera
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    
    # FPS calculation variables
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            # Process frame to detect entities
            entities = detector.process_frame(frame)
            
            # Get annotated frame
            annotated_frame = detector.get_annotated_frame(frame, entities)
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display FPS and entity count
            human_count = sum(1 for entity in entities if entity.type == "human")
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f} | Humans: {human_count} (Limited to {settings.MP_MAX_NUM_FACES})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display instructions
            cv2.putText(
                annotated_frame,
                "Press ESC to exit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow("Face Position and Limited Multi-Person Detection Test", annotated_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        logger.error(f"Error during test: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("=== Face Position and Limited Multi-Person Detection Test Complete ===")

if __name__ == "__main__":
    test_face_position()