import cv2
import time
import json
import numpy as np
from core.camera import CameraManager
from core.detector import EntityDetector
from core.analyser import MotionAnalyser
from config.settings import Settings
from utils.logger import app_logger as logger

def test_pose_retargeting():
    """Test the pose retargeting functionality for human entities"""
    logger.info("Starting pose retargeting test")
    
    # Initialize settings
    settings = Settings()
    
    # Initialize components
    try:
        logger.info("Initializing components...")
        camera = CameraManager(settings)
        detector = EntityDetector(settings)
        motion_analyser = MotionAnalyser(settings)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {str(e)}")
        return False
    
    # Process frames to test pose retargeting
    try:
        logger.info("Processing frames for pose retargeting...")
        frame_count = 0
        start_time = time.time()
        
        # Create a window for visualization
        cv2.namedWindow('Pose Retargeting Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Retargeting Test', 1280, 720)
        
        while True:
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("Failed to capture frame, skipping")
                continue
                
            # Detect entities
            entities = detector.process_frame(frame)
            logger.info(f"Frame {frame_count+1}: Detected {len(entities)} entities")
            
            # Create a copy of the frame for visualization
            display_frame = frame.copy()
            
            # Process each entity
            for entity in entities:
                if entity.type == "human":
                    # Analyze motion and get retargeted pose data
                    motion_data = motion_analyser.analyze_motion(entity)
                    
                    # Extract joint rotations
                    joint_rotations = motion_data.get("joint_rotations", {})
                    
                    # Display joint rotation data on the frame
                    h, w = display_frame.shape[:2]
                    text_y = 30
                    cv2.putText(display_frame, f"Entity {entity.id} Joint Rotations:", 
                                (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    text_y += 30
                    
                    # Display rotation data for each joint
                    for joint_name, rotation in joint_rotations.items():
                        rotation_text = f"{joint_name}: X={rotation['x']:.2f}, Y={rotation['y']:.2f}, Z={rotation['z']:.2f}"
                        cv2.putText(display_frame, rotation_text, 
                                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
                        text_y += 25
                    
                    # Visualize the skeleton with retargeted pose
                    visualize_retargeted_pose(display_frame, entity, joint_rotations)
            
            # Add performance info
            fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Entities: {len(entities)}", (10, h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Pose Retargeting Test', display_frame)
            
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

def visualize_retargeted_pose(frame, entity, joint_rotations):
    """
    Visualize the retargeted pose on the frame
    This draws lines representing the joint rotations
    """
    h, w = frame.shape[:2]
    
    # Get keypoints from entity
    if 'body' not in entity.keypoints:
        return
    
    keypoints = entity.keypoints['body']
    
    # Define colors for different joint types
    colors = {
        'head': (0, 0, 255),      # Red
        'neck': (0, 128, 255),    # Orange
        'torso': (0, 255, 255),   # Yellow
        'left_arm': (0, 255, 0),  # Green
        'right_arm': (255, 0, 0), # Blue
        'left_leg': (255, 0, 255),# Magenta
        'right_leg': (128, 0, 255)# Purple
    }
    
    # Draw lines representing joint rotations
    for joint_name, rotation in joint_rotations.items():
        # Skip joints with no rotation
        if abs(rotation['x']) < 0.01 and abs(rotation['y']) < 0.01 and abs(rotation['z']) < 0.01:
            continue
        
        # Get the joint position
        joint_pos = None
        if joint_name == 'head' and 'nose' in keypoints:
            joint_pos = keypoints['nose']
        elif joint_name == 'neck' and 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            # Calculate neck position as midpoint between shoulders
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            joint_pos = [(left_shoulder[0] + right_shoulder[0]) / 2, 
                         (left_shoulder[1] + right_shoulder[1]) / 2]
        elif joint_name == 'torso' and 'left_hip' in keypoints and 'right_hip' in keypoints:
            # Calculate torso position as midpoint between hips
            left_hip = keypoints['left_hip']
            right_hip = keypoints['right_hip']
            joint_pos = [(left_hip[0] + right_hip[0]) / 2, 
                         (left_hip[1] + right_hip[1]) / 2]
        elif joint_name == 'left_arm' and 'left_shoulder' in keypoints:
            joint_pos = keypoints['left_shoulder']
        elif joint_name == 'right_arm' and 'right_shoulder' in keypoints:
            joint_pos = keypoints['right_shoulder']
        elif joint_name == 'left_forearm' and 'left_elbow' in keypoints:
            joint_pos = keypoints['left_elbow']
        elif joint_name == 'right_forearm' and 'right_elbow' in keypoints:
            joint_pos = keypoints['right_elbow']
        elif joint_name == 'left_leg' and 'left_hip' in keypoints:
            joint_pos = keypoints['left_hip']
        elif joint_name == 'right_leg' and 'right_hip' in keypoints:
            joint_pos = keypoints['right_hip']
        elif joint_name == 'left_foot' and 'left_knee' in keypoints:
            joint_pos = keypoints['left_knee']
        elif joint_name == 'right_foot' and 'right_knee' in keypoints:
            joint_pos = keypoints['right_knee']
        
        if joint_pos is not None:
            # Convert normalized coordinates to pixel coordinates
            x, y = int(joint_pos[0] * w), int(joint_pos[1] * h)
            
            # Calculate line length based on rotation magnitude
            magnitude = max(abs(rotation['x']), abs(rotation['y']), abs(rotation['z'])) * 50
            length = max(20, min(100, magnitude))
            
            # Draw lines representing each rotation axis
            color = colors.get(joint_name, (255, 255, 255))
            
            # X-axis rotation (pitch) - red component
            end_x = x
            end_y = int(y - length * rotation['x'])
            cv2.line(frame, (x, y), (end_x, end_y), (0, 0, 255), 2)
            
            # Y-axis rotation (yaw) - green component
            end_x = int(x + length * rotation['y'])
            end_y = y
            cv2.line(frame, (x, y), (end_x, end_y), (0, 255, 0), 2)
            
            # Z-axis rotation (roll) - blue component
            # For visualization, draw a circle with radius proportional to Z rotation
            radius = int(abs(rotation['z']) * 20)
            if radius > 2:
                cv2.circle(frame, (x, y), radius, (255, 0, 0), 2)
            
            # Label the joint
            cv2.putText(frame, joint_name, (x + 5, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

if __name__ == "__main__":
    success = test_pose_retargeting()
    if success:
        logger.info("Pose retargeting test completed successfully")
    else:
        logger.error("Pose retargeting test failed")