from pythonosc import udp_client
import json
import os
from utils.logger import app_logger as logger


class TDCommunicator:
    def __init__(self, settings):
        self.osc_client = udp_client.SimpleUDPClient(
            settings.TD_IP, settings.TD_OSC_PORT)
        self.mapping = self._load_mapping()
        logger.info(f"TDCommunicator initialized with OSC client at {settings.TD_IP}:{settings.TD_OSC_PORT}")

    def send_avatar_data(self, avatars):
        """Send avatar data to TouchDesigner using the configured mapping"""
        if not self.mapping:
            # Fallback to hardcoded addresses if no mapping is available
            logger.warning("No OSC mapping found, using fallback method")
            self._send_avatar_data_fallback(avatars)
            return

        # Send entity count to TouchDesigner
        if "scene" in self.mapping and "entities_count" in self.mapping["scene"]:
            self._send_osc(self.mapping["scene"]["entities_count"], len(avatars))
            logger.debug(f"Sent entity count: {len(avatars)}")

        for i, avatar in enumerate(avatars):
            try:
                # Check if avatar has the expected structure
                if avatar.get("status") == "fallback":
                    # Use default values for fallback avatars
                    avatar_mapping = self.mapping.get("avatar", {})
                    status_address = avatar_mapping.get("status", f"/avatar/{i}/status")
                    self._send_osc(status_address.replace("{index}", str(i)), "fallback")
                    logger.debug(f"Sent fallback status for avatar {i}")
                    continue

                # Extract data from avatar structure
                avatar_data = avatar.get("avatar", {})

                # Prepare a complete data structure for sending
                complete_data = {
                    "status": avatar.get("status", "active"),
                    "type": avatar.get("entity_type", "human"),
                    "species": avatar.get("species", "human"),
                    "url": avatar_data.get("url", ""),
                    "features": avatar_data.get("features", {}),
                    "animation": self._prepare_animation_data(avatar)
                }

                # Use mapping to determine OSC addresses
                self._send_mapped_data("avatar", complete_data, i)
                logger.debug(f"Sent complete data for avatar {i}")
            except Exception as e:
                logger.error(f"Error sending avatar data for avatar {i}: {str(e)}")

    def _prepare_animation_data(self, avatar):
        """Extract and prepare animation data from avatar"""
        # Default animation data structure
        animation_data = {
            "motion_detected": False,
            "joint_rotations": {},
            "facial_expressions": {},
            "body_motion": {
                "velocity": 0.0,
                "acceleration": 0.0
            }
        }

        # Extract animation data if available
        if "body_motion" in avatar:
            animation_data["motion_detected"] = avatar.get("motion_detected", False)
            animation_data["body_motion"] = avatar.get("body_motion", animation_data["body_motion"])

        if "joint_rotations" in avatar:
            animation_data["joint_rotations"] = avatar.get("joint_rotations", {})

        if "facial_expressions" in avatar:
            animation_data["facial_expressions"] = avatar.get("facial_expressions", {})

        return animation_data

    def _send_avatar_data_fallback(self, avatars):
        """Fallback method when no mapping is available"""
        for i, avatar in enumerate(avatars):
            try:
                # Check if avatar has the expected structure
                if avatar.get("status") == "fallback":
                    # Use default values for fallback avatars
                    self._send_osc(f"/avatar/{i}/status", "fallback")
                    continue

                # Extract data from avatar structure
                avatar_data = avatar.get("avatar", {})
                features = avatar_data.get("features", {})

                # Send available data
                self._send_osc(f"/avatar/{i}/status", avatar.get("status", "unknown"))
                self._send_osc(f"/avatar/{i}/type", avatar.get("entity_type", "human"))
                self._send_osc(f"/avatar/{i}/species", avatar.get("species", "human"))

                if "url" in avatar_data:
                    self._send_osc(f"/avatar/{i}/url", avatar_data["url"])

                if features:
                    self._send_osc(f"/avatar/{i}/features", json.dumps(features))

                # Send animation data if available
                if "joint_rotations" in avatar:
                    self._send_osc(f"/avatar/{i}/animation/joints", json.dumps(avatar["joint_rotations"]))

                if "facial_expressions" in avatar:
                    self._send_osc(f"/avatar/{i}/animation/face", json.dumps(avatar["facial_expressions"]))

                if "body_motion" in avatar:
                    self._send_osc(f"/avatar/{i}/animation/motion", json.dumps(avatar["body_motion"]))

                if "motion_detected" in avatar:
                    self._send_osc(f"/avatar/{i}/animation/motion_detected", avatar["motion_detected"])
            except Exception as e:
                logger.error(f"Error sending avatar data for avatar {i}: {str(e)}")

    def _send_mapped_data(self, section, data, index=None):
        """Send data according to the mapping"""
        if section not in self.mapping:
            logger.warning(f"Section '{section}' not found in mapping")
            return

        mapping = self.mapping[section]
        self._send_data_recursive(mapping, data, "", index)

    def _send_data_recursive(self, mapping, data, path_prefix, index=None):
        """Recursively traverse mapping and data to send OSC messages"""
        if isinstance(mapping, str):
            # This is a leaf node with an OSC address
            address = mapping
            if index is not None:
                address = address.replace("{index}", str(index))

            # Extract the value from data using the path
            keys = path_prefix.strip('/').split('/')
            value = data
            for key in keys:
                if key and isinstance(value, dict):
                    value = value.get(key, {})

            # Send the OSC message
            if value is not None and not isinstance(value, dict):
                self._send_osc(address, value)
            return

        # This is a branch node, recurse into children
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                new_prefix = f"{path_prefix}/{key}" if path_prefix else key
                self._send_data_recursive(value, data, new_prefix, index)

    def _send_osc(self, address, data):
        """Send OSC message with appropriate data type conversion"""
        try:
            # Convert complex data types to JSON strings
            if isinstance(data, (dict, list)):
                data = json.dumps(data)

            # Send the message
            self.osc_client.send_message(address, data)
        except Exception as e:
            logger.error(f"Error sending OSC message to {address}: {str(e)}")

    def _load_mapping(self):
        """Load OSC mapping from JSON file"""
        try:
            mapping_path = os.path.join("config", "td_osc_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    logger.info(f"Loaded OSC mapping from {mapping_path}")
                    return mapping
            else:
                logger.warning(f"Mapping file {mapping_path} not found. Using default mapping.")
                return {}
        except Exception as e:
            logger.error(f"Error loading mapping: {str(e)}")
            return {}
