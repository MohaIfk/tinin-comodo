import cv2
import time
from core.detector import EntityDetector
from config.settings import Settings
from utils.logger import app_logger as logger

def test_mediapipe_implementation():
    """Test the enhanced MediaPipe implementation for human detection (face + pose)"""
    logger.info("Starting test for enhanced MediaPipe implementation")
    
    # Initialize settings and detector
    settings = Settings()
    detector = EntityDetector(settings)
    
    # Initialize webcam
    logger.info("Initializing webcam...")
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    
    logger.info(f"Webcam initialized with resolution: {settings.CAMERA_WIDTH}x{settings.CAMERA_HEIGHT}")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame
            process_start = time.time()
            entities = detector.process_frame(frame)
            process_time = time.time() - process_start
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Create annotated frame
            debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
            
            # Add performance info
            cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Process time: {process_time*1000:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Entities: {len(entities)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display info about detected entities
            y_pos = 120
            for i, entity in enumerate(entities):
                if entity.type == "human":
                    # Display emotion info
                    emotions_str = ", ".join([f"{k}: {v:.2f}" for k, v in entity.emotions.items() if v > 0.1])
                    cv2.putText(debug_frame, f"Human {i+1} emotions: {emotions_str}", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
                    
                    # Display keypoints count
                    body_kp_count = len(entity.keypoints.get('body', {}))
                    face_kp_count = sum(len(feature) for feature in entity.keypoints.get('face', {}).values())
                    cv2.putText(debug_frame, f"Human {i+1} keypoints: {body_kp_count} body, {face_kp_count} face", 
                                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 30
            
            # Show frame
            cv2.imshow('MediaPipe Test', debug_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        logger.exception("Error details")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("MediaPipe test completed")

if __name__ == "__main__":
    test_mediapipe_implementation()