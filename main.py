from core.camera import CameraManager
from core.detector import EntityDetector
from core.avatar_generator import AvatarGenerator
from core.communicator import TDCommunicator
from core.analyser import MotionAnalyser
from config.settings import Settings
import cv2
import time
from utils.logger import app_logger as logger
from utils.performance import performance_monitor, track_performance
from utils.helpers import draw_text


@track_performance
def main():
    logger.info("Starting 3DMesh application")

    # Initialize components
    settings = Settings()
    logger.info(f"Debug mode: {settings.DEBUG_MODE}")

    # Check for API key
    if not settings.AVATAR_API_KEY:
        logger.warning("No AVATAR_API_KEY found in environment variables. Avatar generation will use fallback mode.")

    # Initialize components with proper error handling
    try:
        camera = CameraManager(settings)
        detector = EntityDetector(settings)
        avatar_gen = AvatarGenerator(settings)
        td_comm = TDCommunicator(settings)
        motion_analyser = MotionAnalyser(settings)

        logger.info("All components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {str(e)}")
        logger.exception("Initialization error details")
        return

    frame_count = 0
    start_time = time.time()

    try:
        logger.info("Entering main processing loop")
        while True:
            loop_start = time.time()

            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame, skipping")
                continue

            # Detect entities
            entities = detector.process_frame(frame)
            logger.debug(f"Detected {len(entities)} entities")
            # Generate/update avatars
            avatars = []
            for entity in entities:
                # Analyze motion for each entity
                motion_data = motion_analyser.analyze_motion(entity)

                # Get avatar with reference image
                avatar = avatar_gen.get_avatar(entity, frame)

                # Add motion data to avatar
                if "avatar" in avatar and "features" in avatar["avatar"]:
                    avatar["avatar"]["features"]["motion"] = motion_data

                avatars.append(avatar)

            # Send data to TouchDesigner
            td_comm.send_avatar_data(avatars)

            # Performance monitoring
            frame_count += 1
            elapsed = time.time() - loop_start
            performance_monitor.track_frame_time(elapsed)

            # Display preview (optional)
            if settings.DEBUG_MODE:
                # Get performance stats
                stats = performance_monitor.get_frame_stats()
                fps = stats.get("fps", 0)

                # Create debug frame with annotations
                debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)

                # Add performance info
                draw_text(debug_frame, f"FPS: {fps:.1f}", (10, 30), 0.7, (0, 255, 0), 2)
                draw_text(debug_frame, f"Entities: {len(entities)}", (10, 60), 0.7, (0, 255, 0), 2)

                # Show frame
                cv2.imshow('3DMesh Debug View', debug_frame)
                if cv2.waitKey(1) == 27:  # ESC key
                    logger.info("ESC key pressed, exiting")
                    break

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {str(e)}")
        logger.exception("Error details")
    finally:
        # Calculate overall stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        logger.info(f"Processed {frame_count} frames in {total_time:.2f} seconds ({avg_fps:.2f} FPS average)")

        # Clean up
        logger.info("Releasing resources")
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    main()
