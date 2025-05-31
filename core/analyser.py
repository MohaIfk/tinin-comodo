import numpy as np
import math
from typing import Dict, List, Any, Optional
from utils.logger import app_logger as logger


class MotionAnalyser:
    """Analyzes motion patterns from detected entities for real-time animation"""

    def __init__(self, settings):
        self.history_length = getattr(settings, 'MOTION_HISTORY_LENGTH', 30)
        self.entity_history = {}  # Stores motion history for each entity
        self.emotion_history = {}  # Stores emotion history for each entity
        self.smoothing_factor = 0.3  # For smoothing animations (0-1, lower = smoother)
        logger.info("MotionAnalyser initialized")

    def analyze_motion(self, entity) -> Dict[str, Any]:
        """
        Analyze motion patterns and emotions for an entity to drive avatar animation
        Returns animation data suitable for TouchDesigner
        """
        # Initialize history for new entities
        if entity.id not in self.entity_history:
            self.entity_history[entity.id] = []
            self.emotion_history[entity.id] = []

        # Add current keypoints to history
        keypoints_history = self.entity_history[entity.id]
        keypoints_history.append(entity.keypoints)

        # Add current emotions to history
        emotion_history = self.emotion_history[entity.id]
        emotion_history.append(entity.emotions)

        # Limit history length
        if len(keypoints_history) > self.history_length:
            keypoints_history.pop(0)
        if len(emotion_history) > self.history_length:
            emotion_history.pop(0)

        # Not enough history to analyze motion
        if len(keypoints_history) < 2:
            return self._create_default_animation_data(entity)

        # Process based on entity type
        if entity.type == "human":
            animation_data = self._analyze_human_motion(entity, keypoints_history, emotion_history)
        elif entity.type == "animal":
            animation_data = self._analyze_animal_motion(entity, keypoints_history)
        else:
            animation_data = self._create_default_animation_data(entity)

        return animation_data

    def _analyze_human_motion(self, entity, keypoints_history, emotion_history) -> Dict[str, Any]:
        """Analyze human motion and emotions for avatar animation"""
        try:
            # Get the latest and previous keypoints
            current_keypoints = keypoints_history[-1]
            previous_keypoints = keypoints_history[-2] if len(keypoints_history) > 1 else current_keypoints

            # Extract body keypoints
            body_keypoints = current_keypoints.get('body', {})
            prev_body_keypoints = previous_keypoints.get('body', {})

            # Calculate joint rotations for major body parts
            joint_rotations = self._calculate_human_joint_rotations(body_keypoints, prev_body_keypoints)

            # Process facial expressions from emotions
            facial_expressions = self._process_facial_expressions(entity.emotions, self.emotion_history[entity.id])

            # Calculate overall body motion
            body_motion = self._calculate_body_motion(keypoints_history)

            # Combine all animation data
            animation_data = {
                "motion_detected": body_motion["motion_detected"],
                "joint_rotations": joint_rotations,
                "facial_expressions": facial_expressions,
                "body_motion": body_motion,
                "entity_type": "human"
            }

            return animation_data
        except Exception as e:
            logger.error(f"Error analyzing human motion: {str(e)}")
            return self._create_default_animation_data(entity)

    def _analyze_animal_motion(self, entity, keypoints_history) -> Dict[str, Any]:
        """Analyze animal motion for avatar animation"""
        try:
            # Get the latest and previous keypoints
            current_keypoints = keypoints_history[-1]
            previous_keypoints = keypoints_history[-2] if len(keypoints_history) > 1 else current_keypoints

            # Extract body keypoints
            body_keypoints = current_keypoints.get('body', {})
            prev_body_keypoints = previous_keypoints.get('body', {})

            # Calculate joint rotations based on animal type
            joint_rotations = self._calculate_animal_joint_rotations(
                body_keypoints, 
                prev_body_keypoints, 
                entity.species
            )

            # Calculate overall body motion
            body_motion = self._calculate_body_motion(keypoints_history)

            # Combine all animation data
            animation_data = {
                "motion_detected": body_motion["motion_detected"],
                "joint_rotations": joint_rotations,
                "body_motion": body_motion,
                "entity_type": "animal",
                "species": entity.species
            }

            return animation_data
        except Exception as e:
            logger.error(f"Error analyzing animal motion: {str(e)}")
            return self._create_default_animation_data(entity)

    def _calculate_human_joint_rotations(self, keypoints, prev_keypoints) -> Dict[str, Any]:
        """
        Calculate rotation angles for human joints and retarget to avatar skeleton
        This implements pose retargeting for human entities
        """
        rotations = {}

        # Define key joints to track with their corresponding avatar bone chains
        joints = [
            # Joint name, keypoints to use, retargeting multipliers [x, y, z]
            ("head", ["nose", "left_eye", "right_eye"], [1.2, 1.0, 1.5]),
            ("neck", ["left_shoulder", "right_shoulder", "nose"], [1.0, 1.0, 0.8]),
            ("torso", ["left_shoulder", "right_shoulder", "left_hip", "right_hip"], [0.7, 0.7, 1.0]),
            ("left_arm", ["left_shoulder", "left_elbow", "left_wrist"], [1.0, 1.2, 1.0]),
            ("right_arm", ["right_shoulder", "right_elbow", "right_wrist"], [1.0, 1.2, 1.0]),
            ("left_forearm", ["left_elbow", "left_wrist"], [1.3, 1.0, 1.0]),
            ("right_forearm", ["right_elbow", "right_wrist"], [1.3, 1.0, 1.0]),
            ("left_leg", ["left_hip", "left_knee", "left_ankle"], [1.0, 0.8, 1.0]),
            ("right_leg", ["right_hip", "right_knee", "right_ankle"], [1.0, 0.8, 1.0]),
            ("left_foot", ["left_knee", "left_ankle"], [0.5, 0.5, 0.5]),
            ("right_foot", ["right_knee", "right_ankle"], [0.5, 0.5, 0.5])
        ]

        # Calculate rotation for each joint
        for joint_name, joint_points, multipliers in joints:
            try:
                # Check if enough required keypoints exist
                valid_points = [point for point in joint_points if point in keypoints]
                valid_prev_points = [point for point in joint_points if point in prev_keypoints]

                if len(valid_points) >= 2 and len(valid_prev_points) >= 2:
                    # Calculate angle between current and previous position
                    current_angle = self._calculate_angle(keypoints, valid_points)
                    prev_angle = self._calculate_angle(prev_keypoints, valid_prev_points)

                    # Calculate raw rotation
                    raw_rotation = [
                        current_angle[0] - prev_angle[0],
                        current_angle[1] - prev_angle[1],
                        current_angle[2] - prev_angle[2]
                    ]

                    # Apply retargeting multipliers to adapt to avatar skeleton proportions
                    retargeted_rotation = [
                        raw_rotation[0] * multipliers[0],
                        raw_rotation[1] * multipliers[1],
                        raw_rotation[2] * multipliers[2]
                    ]

                    # Apply smoothing to prevent jitter
                    smoothed_rotation = self._smooth_rotation(retargeted_rotation)

                    # Store the rotation
                    rotations[joint_name] = {
                        "x": smoothed_rotation[0],
                        "y": smoothed_rotation[1],
                        "z": smoothed_rotation[2]
                    }
                else:
                    # Default rotation if not enough keypoints are available
                    rotations[joint_name] = {"x": 0, "y": 0, "z": 0}
            except Exception as e:
                logger.error(f"Error calculating rotation for {joint_name}: {str(e)}")
                rotations[joint_name] = {"x": 0, "y": 0, "z": 0}

        # Add special handling for facial rotations if facial landmarks are available
        if "nose" in keypoints and "left_eye" in keypoints and "right_eye" in keypoints:
            try:
                # Calculate head tilt based on eye positions
                left_eye = keypoints["left_eye"]
                right_eye = keypoints["right_eye"]
                nose = keypoints["nose"]

                # Calculate eye line angle
                dx_eyes = right_eye[0] - left_eye[0]
                dy_eyes = right_eye[1] - left_eye[1]
                eye_angle = math.atan2(dy_eyes, dx_eyes)

                # Add head tilt to rotations (roll around z-axis)
                if "head" in rotations:
                    rotations["head"]["z"] += eye_angle / math.pi * 0.5  # Scale down for subtlety
            except Exception as e:
                logger.error(f"Error calculating facial rotation: {str(e)}")

        return rotations

    def _calculate_animal_joint_rotations(self, keypoints, prev_keypoints, species) -> Dict[str, Any]:
        """Calculate rotation angles for animal joints based on species"""
        rotations = {}

        # Define joints based on animal type
        if species in ["dog", "cat", "horse", "cow"]:  # Quadrupeds
            joints = [
                ("head", ["neck", "front_shoulders"]),
                ("body", ["front_shoulders", "back_hips"]),
                ("tail", ["back_hips", "tail"]),
                ("front_legs", ["front_shoulders", "front_left_leg", "front_right_leg"]),
                ("back_legs", ["back_hips", "back_left_leg", "back_right_leg"])
            ]
        elif species == "bird":  # Birds
            joints = [
                ("head", ["beak", "neck"]),
                ("body", ["neck", "wings"]),
                ("tail", ["body", "tail"]),
                ("wings", ["wings", "body"])
            ]
        else:  # Default for other animals
            joints = [
                ("head", ["head", "neck"]),
                ("body", ["neck", "tail"])
            ]

        # Calculate rotation for each joint
        for joint_name, joint_points in joints:
            # Check if all required keypoints exist
            if all(point in keypoints for point in joint_points) and all(point in prev_keypoints for point in joint_points):
                # Calculate angle between current and previous position
                current_angle = self._calculate_angle(keypoints, joint_points)
                prev_angle = self._calculate_angle(prev_keypoints, joint_points)

                # Calculate rotation with smoothing
                rotation = self._smooth_rotation(current_angle - prev_angle)

                rotations[joint_name] = {
                    "x": rotation[0],
                    "y": rotation[1],
                    "z": rotation[2]
                }
            else:
                # Default rotation if keypoints are missing
                rotations[joint_name] = {"x": 0, "y": 0, "z": 0}

        return rotations

    def _calculate_angle(self, keypoints, joint_points) -> List[float]:
        """Calculate 3D angle between joint points"""
        try:
            # Check if we have enough joint points
            if len(joint_points) < 2:
                return [0.0, 0.0, 0.0]

            # Get the coordinates of the joint points
            points = []
            for point_name in joint_points:
                if point_name in keypoints:
                    # Handle nested keypoints
                    if isinstance(keypoints[point_name], dict):
                        # Use the first sub-point for simplicity
                        sub_point = list(keypoints[point_name].values())[0]
                        points.append(sub_point)
                    else:
                        points.append(keypoints[point_name])

            # If we don't have enough valid points, return default
            if len(points) < 2:
                return [0.0, 0.0, 0.0]

            # Calculate angles between consecutive points
            angles = [0.0, 0.0, 0.0]  # [x, y, z] rotations

            # For each pair of consecutive points
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]

                # Calculate direction vector between points
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]

                # Calculate angle in the XY plane (yaw - around Z axis)
                # atan2 returns angle in radians in range [-π, π]
                yaw = math.atan2(dy, dx)
                angles[2] += yaw

                # For pitch (around X axis) and roll (around Y axis)
                # In 2D we can only estimate these based on relative positions
                # In a real 3D system, we would use the Z coordinate as well

                # Estimate pitch based on vertical displacement
                pitch = dy * math.pi / 2  # Simplified approximation
                angles[0] += pitch

                # Estimate roll based on horizontal displacement
                roll = dx * math.pi / 2  # Simplified approximation
                angles[1] += roll

            # Normalize angles to be in range [-π, π]
            for i in range(3):
                angles[i] = (angles[i] + math.pi) % (2 * math.pi) - math.pi

                # Convert to a normalized range for animation (-1 to 1)
                angles[i] = angles[i] / math.pi

            return angles

        except Exception as e:
            logger.error(f"Error calculating angle: {str(e)}")
            return [0.0, 0.0, 0.0]

    def _smooth_rotation(self, rotation) -> List[float]:
        """Apply smoothing to rotation values to prevent jitter"""
        return [
            rotation[0] * self.smoothing_factor,
            rotation[1] * self.smoothing_factor,
            rotation[2] * self.smoothing_factor
        ]

    def _process_facial_expressions(self, emotions, emotion_history) -> Dict[str, float]:
        """Process emotions into facial expression controls for animation"""
        # Default expressions
        expressions = {
            "smile": 0.0,
            "eyebrow_raise": 0.0,
            "eye_open": 0.5,
            "mouth_open": 0.0,
            "jaw_clench": 0.0
        }

        # Map emotions to expressions
        if emotions:
            # Smile based on happiness
            expressions["smile"] = emotions.get("happy", 0.0)

            # Eyebrow raise based on surprise
            expressions["eyebrow_raise"] = emotions.get("surprised", 0.0)

            # Mouth open based on surprise and happy
            expressions["mouth_open"] = max(
                emotions.get("surprised", 0.0) * 0.7,
                emotions.get("happy", 0.0) * 0.3
            )

            # Jaw clench based on anger
            expressions["jaw_clench"] = emotions.get("angry", 0.0)

            # Eye openness (default to normal, widen with surprise, narrow with anger)
            expressions["eye_open"] = 0.5 + (
                emotions.get("surprised", 0.0) * 0.3 - 
                emotions.get("angry", 0.0) * 0.2
            )

        # Apply smoothing if we have history
        if emotion_history and len(emotion_history) > 1:
            prev_emotions = emotion_history[-2]
            for key in expressions:
                # Smooth transition between expressions
                if key in prev_emotions:
                    expressions[key] = (
                        expressions[key] * self.smoothing_factor + 
                        prev_emotions[key] * (1 - self.smoothing_factor)
                    )

        return expressions

    def _calculate_body_motion(self, keypoints_history) -> Dict[str, Any]:
        """Calculate overall body motion metrics"""
        try:
            # Extract body keypoints from history
            body_keypoints = []
            for kp in keypoints_history:
                if 'body' in kp:
                    # Flatten the body keypoints dictionary into a list of coordinates
                    coords = []
                    for part, point in kp['body'].items():
                        if isinstance(point, dict):  # Handle nested keypoints
                            for sub_part, sub_point in point.items():
                                coords.extend(sub_point)
                        else:
                            coords.extend(point)
                    body_keypoints.append(coords)

            # Convert to numpy array for calculations
            if not body_keypoints:
                return {"motion_detected": False, "velocity": 0.0, "acceleration": 0.0}

            keypoints_array = np.array(body_keypoints)

            # Calculate velocity (change between frames)
            if len(keypoints_array) > 1:
                velocities = np.diff(keypoints_array, axis=0)
                avg_velocity = float(np.mean(np.abs(velocities)))
            else:
                avg_velocity = 0.0

            # Calculate acceleration (change in velocity)
            if len(keypoints_array) > 2:
                accelerations = np.diff(velocities, axis=0)
                avg_acceleration = float(np.mean(np.abs(accelerations)))
            else:
                avg_acceleration = 0.0

            # Determine if significant motion is detected
            motion_detected = avg_velocity > 0.01

            return {
                "motion_detected": motion_detected,
                "velocity": avg_velocity,
                "acceleration": avg_acceleration
            }
        except Exception as e:
            logger.error(f"Error calculating body motion: {str(e)}")
            return {"motion_detected": False, "velocity": 0.0, "acceleration": 0.0}

    def _create_default_animation_data(self, entity) -> Dict[str, Any]:
        """Create default animation data when analysis fails"""
        return {
            "motion_detected": False,
            "joint_rotations": {},
            "facial_expressions": {} if entity.type == "human" else None,
            "body_motion": {
                "velocity": 0.0,
                "acceleration": 0.0
            },
            "entity_type": entity.type,
            "species": getattr(entity, "species", None)
        }

    def get_motion_prediction(self, entity_id) -> Optional[Dict[str, Any]]:
        """Predict future position based on motion history"""
        if entity_id not in self.entity_history or len(self.entity_history[entity_id]) < 3:
            return None

        try:
            # Extract body keypoints from history
            body_keypoints = []
            for kp in self.entity_history[entity_id]:
                if 'body' in kp:
                    # Flatten the body keypoints dictionary into a list of coordinates
                    coords = []
                    for part, point in kp['body'].items():
                        if isinstance(point, dict):  # Handle nested keypoints
                            for sub_part, sub_point in point.items():
                                coords.extend(sub_point)
                        else:
                            coords.extend(point)
                    body_keypoints.append(coords)

            if not body_keypoints or len(body_keypoints) < 3:
                return None

            history = np.array(body_keypoints)

            # Simple linear prediction based on recent velocity
            last_position = history[-1]
            velocity = history[-1] - history[-2]

            # Predict next position
            predicted_position = last_position + velocity

            return {
                "predicted_position": predicted_position.tolist(),
                "confidence": 0.7  # Placeholder confidence value
            }
        except Exception as e:
            logger.error(f"Error predicting motion: {str(e)}")
            return None
