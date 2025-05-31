import cv2
from config.settings import Settings
from core.detector import EntityDetector
from utils.logger import app_logger as logger

def test_multi_person_detection():
    """Test multi-person detection in the 3DMesh project."""
    logger.info("=== Multi-Person Detection Test ===")
    
    # Initialize settings
    settings = Settings()
    logger.info(f"Maximum number of faces: {settings.MP_MAX_NUM_FACES}")
    
    # Initialize detector
    detector = EntityDetector(settings)
    
    # Initialize camera
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    
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
            
            # Display number of detected humans
            human_count = sum(1 for entity in entities if entity.type == "human")
            logger.info(f"Detected {human_count} humans and {len(entities) - human_count} animals")
            
            # Add count to frame
            cv2.putText(
                annotated_frame,
                f"Humans: {human_count}, Animals: {len(entities) - human_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display frame
            cv2.imshow("Multi-Person Detection Test", annotated_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
                
    except Exception as e:
        logger.error(f"Error during multi-person detection test: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        logger.info("=== Multi-Person Detection Test Complete ===")

if __name__ == "__main__":
    test_multi_person_detection()