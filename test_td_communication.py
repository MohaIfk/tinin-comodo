import cv2
import time
import os
import json
from core.communicator import TDCommunicator
from config.settings import Settings
from utils.logger import app_logger as logger

def test_td_communication():
    """Test the OSC communication with TouchDesigner and avatar loading"""
    logger.info("Starting TouchDesigner communication test")

    # Initialize settings
    settings = Settings()

    # Initialize communicator
    try:
        logger.info("Initializing TouchDesigner communicator...")
        td_comm = TDCommunicator(settings)
        logger.info("TouchDesigner communicator initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize TouchDesigner communicator: {str(e)}")
        return False

    # Test sending a test avatar
    try:
        logger.info("Sending test avatar data to TouchDesigner...")

        # Create a test avatar
        test_avatar = create_test_avatar()

        # Send the test avatar to TouchDesigner
        td_comm.send_avatar_data([test_avatar])

        logger.info("Test avatar data sent successfully")

        # Wait a moment to allow TouchDesigner to process the data
        time.sleep(1)

        # Send a second test avatar with different properties
        test_avatar2 = create_test_avatar(entity_type="animal", species="dog")
        td_comm.send_avatar_data([test_avatar, test_avatar2])

        logger.info("Multiple test avatars sent successfully")

        return True
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        return False

def create_test_avatar(entity_type="human", species="human", with_retargeting=True):
    """Create a test avatar with sample data including retargeted pose data"""

    # Define a default avatar URL based on entity type
    if entity_type == "human":
        avatar_url = "https://models.readyplayer.me/683716c2b8857cbb3dbbcf7e.glb"
    else:
        avatar_url = "https://models.readyplayer.me/default_animal.glb"

    # Create a test avatar with all required fields
    avatar = {
        "status": "active",
        "entity_type": entity_type,
        "species": species,
        "avatar": {
            "url": avatar_url,
            "features": {
                "body": {
                    "height": 1.75,
                    "build": "average"
                },
                "face": {
                    "eyes": "blue",
                    "hair": "brown"
                }
            }
        },
        # Add animation data
        "motion_detected": True
    }

    # Add basic joint rotations
    basic_rotations = {
        "head": {"x": 0.1, "y": 0.2, "z": 0.0},
        "torso": {"x": 0.0, "y": 0.0, "z": 0.0},
        "left_arm": {"x": 0.3, "y": 0.1, "z": 0.2},
        "right_arm": {"x": -0.3, "y": 0.1, "z": -0.2}
    }

    # Add enhanced retargeted pose data if requested
    if with_retargeting and entity_type == "human":
        retargeted_rotations = {
            "head": {"x": 0.15, "y": 0.25, "z": 0.05},
            "neck": {"x": 0.05, "y": 0.1, "z": 0.0},
            "torso": {"x": 0.0, "y": 0.05, "z": 0.0},
            "left_arm": {"x": 0.35, "y": 0.15, "z": 0.25},
            "right_arm": {"x": -0.35, "y": 0.15, "z": -0.25},
            "left_forearm": {"x": 0.4, "y": 0.0, "z": 0.2},
            "right_forearm": {"x": -0.4, "y": 0.0, "z": -0.2},
            "left_leg": {"x": 0.1, "y": 0.05, "z": 0.0},
            "right_leg": {"x": -0.1, "y": 0.05, "z": 0.0},
            "left_foot": {"x": 0.05, "y": 0.0, "z": 0.0},
            "right_foot": {"x": -0.05, "y": 0.0, "z": 0.0}
        }
        avatar["joint_rotations"] = retargeted_rotations
    else:
        avatar["joint_rotations"] = basic_rotations

    # Add facial expressions
    avatar["facial_expressions"] = {
        "smile": 0.8,
        "eyebrow_raise": 0.2,
        "eye_open": 0.9,
        "mouth_open": 0.3
    }

    # Add body motion data
    avatar["body_motion"] = {
        "velocity": 0.5,
        "acceleration": 0.1
    }

    return avatar

def test_websocket_fallback():
    """Test WebSocket communication as a fallback if OSC fails"""
    logger.info("Testing WebSocket fallback communication...")

    # This is a placeholder for WebSocket implementation
    # In a real implementation, this would establish a WebSocket connection
    # and send data if OSC communication fails

    logger.info("WebSocket fallback test completed (placeholder)")
    return True

if __name__ == "__main__":
    success = test_td_communication()
    if success:
        logger.info("TouchDesigner communication test completed successfully")

        # Test WebSocket fallback (placeholder)
        websocket_success = test_websocket_fallback()
        if websocket_success:
            logger.info("WebSocket fallback test completed successfully")
        else:
            logger.error("WebSocket fallback test failed")
    else:
        logger.error("TouchDesigner communication test failed")
