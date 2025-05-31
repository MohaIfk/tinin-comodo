import cv2
import time
from core.camera import CameraManager
from core.detector import EntityDetector
from core.avatar_generator import AvatarGenerator
from core.communicator import TDCommunicator
from core.analyser import MotionAnalyser
from config.settings import Settings
from utils.logger import app_logger as logger

def test_integration():
    """Test the integration of all components"""
    logger.info("Starting integration test")

    # Initialize settings
    settings = Settings()

    # Initialize components
    try:
        logger.info("Initializing components...")
        camera = CameraManager(settings)
        detector = EntityDetector(settings)
        avatar_gen = AvatarGenerator(settings)
        td_comm = TDCommunicator(settings)
        motion_analyser = MotionAnalyser(settings)

        logger.info("All components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {str(e)}")
        return False

    # Process a few frames to test the pipeline
    try:
        logger.info("Processing test frames...")
        for i in range(5):  # Process 5 frames
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame, skipping")
                continue

            # Detect entities
            entities = detector.process_frame(frame)
            logger.info(f"Frame {i+1}: Detected {len(entities)} entities")

            # Generate/update avatars and analyze motion
            avatars = []
            for entity in entities:
                # Analyze motion for each entity
                motion_data = motion_analyser.analyze_motion(entity)

                # Get avatar with reference image
                avatar = avatar_gen.get_avatar(entity, frame)

                # Add motion data to avatar
                if "avatar" in avatar and "features" in avatar["avatar"]:
                    avatar["avatar"]["features"]["motion"] = motion_data

                # Add entity type and motion data directly to avatar for OSC
                avatar["entity_type"] = entity.type
                avatar["species"] = entity.species
                avatar["motion_detected"] = motion_data.get("motion_detected", False)
                avatar["joint_rotations"] = motion_data.get("joint_rotations", {})
                avatar["facial_expressions"] = motion_data.get("facial_expressions", {})
                avatar["body_motion"] = motion_data.get("body_motion", {})

                avatars.append(avatar)

            # Send data to TouchDesigner
            td_comm.send_avatar_data(avatars)

            # Display preview if in debug mode
            if settings.DEBUG_MODE:
                debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
                cv2.imshow('Integration Test', debug_frame)
                if cv2.waitKey(100) == 27:  # ESC key
                    break

            time.sleep(0.1)  # Short delay between frames

        logger.info("Test frames processed successfully")
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
    success = test_integration()
    if success:
        logger.info("Integration test completed successfully")
    else:
        logger.error("Integration test failed")
