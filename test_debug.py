import cv2
import numpy as np
import time
from core.detector import EntityDetector
from config.settings import Settings
from utils.logger import app_logger as logger

class DebugEntityDetector(EntityDetector):
    """Extended EntityDetector with debug logging"""
    
    def process_frame(self, frame):
        """Override process_frame to add debug logging"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        entities = []
        
        # Human detection
        pose_results = self.pose.process(rgb_frame)
        face_results = self.face.process(rgb_frame)
        
        # Log pose detection results
        if pose_results.pose_landmarks:
            logger.info("MediaPipe detected a person")
            human_entity = self._process_human(pose_results, face_results)
            
            # Log bounding box
            x1, y1, x2, y2 = human_entity.bbox
            width = x2 - x1
            height = y2 - y1
            logger.info(f"Human bounding box: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
            logger.info(f"Human dimensions: width={width:.3f}, height={height:.3f}")
            
            # Log validation checks
            if width < self.settings.MIN_ENTITY_SIZE or height < self.settings.MIN_ENTITY_SIZE:
                logger.warning(f"Human entity failed size validation: too small (min={self.settings.MIN_ENTITY_SIZE})")
            elif width > self.settings.MAX_ENTITY_SIZE or height > self.settings.MAX_ENTITY_SIZE:
                logger.warning(f"Human entity failed size validation: too large (max={self.settings.MAX_ENTITY_SIZE})")
            elif (x1 < self.settings.VALID_POSITION_MARGIN or 
                  y1 < self.settings.VALID_POSITION_MARGIN or 
                  x2 > (1 - self.settings.VALID_POSITION_MARGIN) or 
                  y2 > (1 - self.settings.VALID_POSITION_MARGIN)):
                logger.warning(f"Human entity failed position validation: too close to edge (margin={self.settings.VALID_POSITION_MARGIN})")
            else:
                logger.info("Human entity passed validation")
            
            # Validate human entity
            if self._validate_entity(human_entity):
                entities.append(human_entity)
                logger.info("Human entity added to entities list")
            else:
                logger.warning("Human entity failed validation, not added to entities list")
        else:
            logger.info("No persons detected by MediaPipe")
        
        # Animal detection (simplified for brevity)
        animal_results = self.animal_detector(rgb_frame)
        animal_entities = self._process_animals(animal_results)
        
        # Log animal detection results
        if animal_entities:
            logger.info(f"Detected {len(animal_entities)} animals")
            
            # Validate animal entities
            for animal_entity in animal_entities:
                if self._validate_entity(animal_entity):
                    entities.append(animal_entity)
                    logger.info(f"Animal entity ({animal_entity.species}) added to entities list")
                else:
                    logger.warning(f"Animal entity ({animal_entity.species}) failed validation, not added to entities list")
        
        logger.info(f"Total entities after validation: {len(entities)}")
        return entities

def test_debug_detector():
    """Test the detector with debug logging"""
    logger.info("Starting debug detector test")
    
    # Initialize components
    settings = Settings()
    
    # Adjust validation parameters if needed for testing
    # Uncomment and modify these lines to test different validation thresholds
    # settings.MIN_ENTITY_SIZE = 0.01  # More permissive minimum size
    # settings.MAX_ENTITY_SIZE = 0.95  # More permissive maximum size
    # settings.VALID_POSITION_MARGIN = 0.05  # More permissive edge margin
    
    detector = DebugEntityDetector(settings)
    
    # Initialize camera
    logger.info("Initializing camera...")
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    
    logger.info(f"Camera initialized with resolution: {settings.CAMERA_WIDTH}x{settings.CAMERA_HEIGHT}")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            # Process frame with debug logging
            logger.info("Processing frame...")
            entities = detector.process_frame(frame)
            
            # Create debug frame with annotations
            debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
            
            # Add text with entity count
            cv2.putText(debug_frame, f"Entities: {len(entities)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Debug Detector Test', debug_frame)
            
            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
            
            # Slow down the loop for readability of logs
            time.sleep(0.5)
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Debug detector test completed")

if __name__ == "__main__":
    test_debug_detector()