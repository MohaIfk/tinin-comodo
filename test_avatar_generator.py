import cv2
import time
import os
from core.camera import CameraManager
from core.detector import EntityDetector
from core.avatar_generator import AvatarGenerator
from config.settings import Settings
from utils.logger import app_logger as logger

def test_avatar_generator():
    """Test the avatar generator with reference image capture and API integration"""
    logger.info("Starting avatar generator test")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize components
    try:
        logger.info("Initializing components...")
        camera = CameraManager(settings)
        detector = EntityDetector(settings)
        avatar_gen = AvatarGenerator(settings)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {str(e)}")
        return False
    
    # Process frames to test reference image capture and avatar generation
    try:
        logger.info("Processing frames for reference image capture and avatar generation...")
        
        # Process a few frames to find entities
        max_attempts = 30  # Try for about 30 frames
        attempts = 0
        entities_found = False
        
        while attempts < max_attempts and not entities_found:
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame, skipping")
                attempts += 1
                continue
                
            # Detect entities
            entities = detector.process_frame(frame)
            logger.info(f"Frame {attempts+1}: Detected {len(entities)} entities")
            
            if len(entities) > 0:
                entities_found = True
                logger.info(f"Found {len(entities)} entities to process")
                
                # Process each entity
                for i, entity in enumerate(entities):
                    logger.info(f"Processing entity {i+1}: {entity.type} (ID: {entity.id})")
                    
                    # Test reference image capture
                    ref_image_path = avatar_gen.capture_reference_image(frame, entity)
                    if ref_image_path:
                        logger.info(f"Reference image captured: {ref_image_path}")
                        
                        # Verify the file exists
                        if os.path.exists(ref_image_path):
                            logger.info(f"Reference image file exists: {os.path.getsize(ref_image_path)} bytes")
                        else:
                            logger.error(f"Reference image file does not exist: {ref_image_path}")
                    else:
                        logger.warning("Failed to capture reference image")
                    
                    # Test avatar generation with reference image
                    logger.info("Generating avatar with reference image...")
                    avatar = avatar_gen.get_avatar(entity, frame)
                    
                    # Check avatar data
                    if avatar and "status" in avatar:
                        logger.info(f"Avatar generated with status: {avatar['status']}")
                        
                        # Check if using fallback
                        if avatar["status"] == "fallback":
                            logger.warning("Using fallback avatar - API key may be invalid or API call failed")
                        
                        # Test caching
                        logger.info("Testing avatar caching...")
                        cache_key = avatar_gen._generate_cache_key(entity, ref_image_path)
                        cached_avatar = avatar_gen._check_cache(cache_key)
                        
                        if cached_avatar:
                            logger.info("Avatar successfully cached")
                        else:
                            logger.warning("Avatar not found in cache")
                    else:
                        logger.error("Invalid avatar data returned")
                
                # Display the frame with detections
                debug_frame = detector.get_annotated_frame(frame, entities, True, True, True)
                cv2.imshow('Avatar Generator Test', debug_frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                
                break  # Exit the loop once we've processed entities
            
            attempts += 1
            time.sleep(0.1)  # Short delay between frames
            
        if not entities_found:
            logger.warning(f"No entities detected after {max_attempts} attempts")
            return False
            
        logger.info("Avatar generator test completed")
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
    success = test_avatar_generator()
    if success:
        logger.info("Avatar generator test completed successfully")
    else:
        logger.error("Avatar generator test failed")