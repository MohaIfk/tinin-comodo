import cv2
import numpy as np
import os
from core.detector import EntityDetector
from core.avatar_generator import AvatarGenerator
from config.settings import Settings
from utils.logger import app_logger as logger

def test_detector_and_avatar_generator():
    """Test the entity validation and model retrieval features"""
    logger.info("Starting test for entity validation and model retrieval")

    # Initialize components
    settings = Settings()
    detector = EntityDetector(settings)
    avatar_gen = AvatarGenerator(settings)

    # Load a sample image (you can replace this with any image path)
    # If no image is available, we'll create a blank one
    try:
        frame = cv2.imread("assets/test_image.jpg")
        if frame is None:
            # Create a blank image if no file is found
            logger.info("Creating blank image for testing")
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

            # Draw a simple human-like shape for testing
            # Head
            cv2.circle(frame, (640, 300), 50, (0, 0, 0), -1)
            # Body
            cv2.rectangle(frame, (615, 350), (665, 500), (0, 0, 0), -1)
            # Arms
            cv2.rectangle(frame, (565, 375), (615, 400), (0, 0, 0), -1)
            cv2.rectangle(frame, (665, 375), (715, 400), (0, 0, 0), -1)
            # Legs
            cv2.rectangle(frame, (615, 500), (635, 600), (0, 0, 0), -1)
            cv2.rectangle(frame, (645, 500), (665, 600), (0, 0, 0), -1)

            # Save the test image
            os.makedirs("assets", exist_ok=True)
            cv2.imwrite("assets/test_image.jpg", frame)
            logger.info("Created and saved test image to assets/test_image.jpg")
    except Exception as e:
        logger.error(f"Error loading/creating image: {e}")
        # Create a blank image
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

    # Test the detector with entity validation
    try:
        logger.info("Testing detector.process_frame with entity validation...")
        entities = detector.process_frame(frame)
        logger.info(f"Detected {len(entities)} entities")

        if len(entities) > 0:
            logger.info("Entity validation test passed!")

            # Test avatar generation with model retrieval
            logger.info("Testing avatar generation with model retrieval...")
            for i, entity in enumerate(entities):
                logger.info(f"Generating avatar for entity {i+1}: {entity.type}")

                # Generate avatar
                avatar = avatar_gen.get_avatar(entity, frame)

                # Check if avatar contains a URL to a 3D model file
                if avatar and "avatar" in avatar and "url" in avatar["avatar"]:
                    url = avatar["avatar"]["url"]
                    logger.info(f"Avatar URL: {url}")

                    # Check if URL points to a .glb or .fbx file
                    if url.endswith(".glb") or url.endswith(".fbx"):
                        logger.info(f"Model retrieval test passed for entity {i+1}!")
                    else:
                        logger.error(f"Model URL does not have a valid extension: {url}")
                else:
                    logger.error("Avatar does not contain a URL to a 3D model file")
        else:
            logger.warning("No entities detected, cannot test avatar generation")

        # Display the frame with detections
        debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
        cv2.imshow('Entity Detection Test', debug_frame)
        logger.info("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_detector_and_avatar_generator()
