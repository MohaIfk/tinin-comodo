import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from dataclasses import dataclass
from typing import List
import cv2

# Complete YOLOv8 animal classes (COCO dataset)
ANIMAL_CLASSES = {
    # Pets
    15: {"name": "bird", "category": "avian", "keypoints": "avian"},
    16: {"name": "cat", "category": "quadruped", "keypoints": "feline"},
    17: {"name": "dog", "category": "quadruped", "keypoints": "canine"},

    # Livestock
    18: {"name": "horse", "category": "quadruped", "keypoints": "equine"},
    19: {"name": "sheep", "category": "quadruped", "keypoints": "bovine"},
    20: {"name": "cow", "category": "quadruped", "keypoints": "bovine"},

    # Wildlife
    21: {"name": "elephant", "category": "quadruped", "keypoints": "pachyderm"},
    22: {"name": "bear", "category": "quadruped", "keypoints": "ursine"},
    23: {"name": "zebra", "category": "quadruped", "keypoints": "equine"},
    24: {"name": "giraffe", "category": "quadruped", "keypoints": "giraffid"},

    # Additional animals that might be detected
    25: {"name": "backpack", "category": "object"},  # Not an animal
    26: {"name": "umbrella", "category": "object"},
    # ... other COCO classes ...

    # Aquatic animals
    1: {"name": "person", "category": "human"},  # Skipped in animal processing
    2: {"name": "fish", "category": "aquatic", "keypoints": "fish"},
    3: {"name": "jellyfish", "category": "aquatic"},
    4: {"name": "crab", "category": "arthropod"}
}

# Animal categories grouped by type
ANIMAL_CATEGORIES = {
    "quadruped": [16, 17, 18, 19, 20, 21, 22, 23, 24],
    "avian": [15],
    "aquatic": [2, 3],
    "arthropod": [4],
    "object": [25, 26]  # Non-animal objects
}

@dataclass
class Entity:
    type: str  # "human" or "animal"
    bbox: tuple  # (x1, y1, x2, y2)
    keypoints: dict
    emotions: dict
    id: int
    species: str


class EntityDetector:
    def __init__(self, settings):
        self.settings = settings
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=settings.MP_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5)
        self.face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=settings.MP_MAX_NUM_FACES,
            min_detection_confidence=settings.MP_FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=settings.MP_FACE_TRACKING_CONFIDENCE,
            refine_landmarks=True)
        # Initialize YOLO with the device setting
        self.animal_detector = YOLO(settings.YOLO_MODEL_PATH)
        # Set the model to use the configured device (GPU or CPU)
        self.animal_detector.to(settings.DEVICE)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame) -> List[Entity]:
        """
        Process a frame to detect and extract entities (humans and animals).

        This method uses YOLO for initial detection of humans and animals,
        then processes each detection with MediaPipe for detailed pose and face analysis.
        The implementation addresses several issues:
        1. False positives are reduced by using a higher confidence threshold
        2. Multiple faces on a single person are handled by selecting the best face
        3. Face detection is performed only within the cropped region of each detected person

        Args:
            frame: Input BGR frame

        Returns:
            List[Entity]: List of detected entities
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        entities = []

        # Use YOLO to detect all entities (humans and animals)
        yolo_results = self.animal_detector(rgb_frame)

        # Process humans
        human_entities = self._process_humans(rgb_frame, yolo_results)
        for human_entity in human_entities:
            if self._validate_entity(human_entity):
                entities.append(human_entity)

        # Process animals
        animal_entities = self._process_animals(yolo_results)
        for animal_entity in animal_entities:
            if self._validate_entity(animal_entity):
                entities.append(animal_entity)

        return entities

    def get_annotated_frame(self, frame, entities=None, show_keypoints=True, show_bbox=True, show_ids=True):
        """
        Returns a frame with visualization of all detected entities
        Args:
            frame: Input BGR frame
            entities: List of Entity objects (if None, will process frame)
            show_keypoints: Whether to draw keypoints
            show_bbox: Whether to draw bounding boxes
            show_ids: Whether to show entity IDs
        Returns:
            Annotated BGR frame
        """
        if entities is None:
            entities = self.process_frame(frame)

        annotated_frame = frame.copy()
        h, w = annotated_frame.shape[:2]

        # Color palette for different entity types
        COLORS = {
            "human": (0, 255, 0),  # Green
            "cat": (255, 0, 0),  # Blue
            "dog": (255, 165, 0),  # Orange
            "bird": (0, 255, 255),  # Yellow
            "default": (255, 0, 255)  # Magenta
        }

        for entity in entities:
            # Get color based on entity type
            color = COLORS.get(entity.type if entity.type != "animal"
                               else entity.species, COLORS["default"])

            # Draw bounding box
            if show_bbox and entity.bbox:
                x1, y1, x2, y2 = entity.bbox
                cv2.rectangle(annotated_frame,
                              (int(x1 * w), int(y1 * h)),
                              (int(x2 * w), int(y2 * h)),
                              color, 2)

            # Draw keypoints
            if show_keypoints and entity.keypoints:
                self._draw_keypoints(annotated_frame, entity, color)

            # Draw ID
            if show_ids:
                text_pos = (int(entity.bbox[0] * w) + 10,
                            int(entity.bbox[1] * h) + 30)
                cv2.putText(annotated_frame,
                            f"ID:{entity.id} {entity.type}",
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

        # Add FPS counter if available
        if hasattr(self, 'fps'):
            cv2.putText(annotated_frame,
                        f"FPS: {self.fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        return annotated_frame

    def _draw_keypoints(self, frame, entity, color):
        """Helper method to draw keypoints on frame"""
        h, w = frame.shape[:2]

        # Draw body keypoints
        if 'body' in entity.keypoints:
            for kp_name, kp in entity.keypoints['body'].items():
                if isinstance(kp, dict):  # Nested keypoints
                    for sub_kp in kp.values():
                        if len(sub_kp) >= 2:  # Ensure we have at least x, y coordinates
                            x, y = sub_kp[0] * w, sub_kp[1] * h
                            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                elif isinstance(kp, list) and len(kp) >= 2:  # Simple keypoint with at least x, y coordinates
                    # Check visibility if available (4th element)
                    visibility = kp[3] if len(kp) > 3 else 1.0
                    if visibility > 0.5:  # Only draw if visibility is above threshold
                        x, y = kp[0] * w, kp[1] * h
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

        # Draw face keypoints if available
        if 'face' in entity.keypoints and entity.keypoints['face']:
            # For each facial feature (eyes, eyebrows, etc.)
            for feature_name, feature_points in entity.keypoints['face'].items():
                if isinstance(feature_points, dict):
                    # Draw each point in the feature
                    for point_name, point in feature_points.items():
                        if isinstance(point, list) and len(point) >= 2:
                            x, y = point[0] * w, point[1] * h
                            # Use smaller circles for face landmarks
                            cv2.circle(frame, (int(x), int(y)), 2, color, -1)

        # Connect keypoints with lines for humans
        if entity.type == "human":
            self._draw_human_skeleton(frame, entity, color)

    def _draw_human_skeleton(self, frame, entity, color):
        """Draw skeleton connections for humans"""
        h, w = frame.shape[:2]
        kps = entity.keypoints['body']

        # Define connections between keypoints (comprehensive MediaPipe pose)
        CONNECTIONS = [
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),

            # Left arm
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_pinky'),
            ('left_wrist', 'left_index'),
            ('left_wrist', 'left_thumb'),
            ('left_pinky', 'left_index'),

            # Right arm
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_pinky'),
            ('right_wrist', 'right_index'),
            ('right_wrist', 'right_thumb'),
            ('right_pinky', 'right_index'),

            # Left leg
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot_index'),
            ('left_heel', 'left_foot_index'),

            # Right leg
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot_index'),
            ('right_heel', 'right_foot_index'),

            # Face
            ('nose', 'left_eye_inner'),
            ('left_eye_inner', 'left_eye'),
            ('left_eye', 'left_eye_outer'),
            ('nose', 'right_eye_inner'),
            ('right_eye_inner', 'right_eye'),
            ('right_eye', 'right_eye_outer'),
            ('left_eye_outer', 'left_ear'),
            ('right_eye_outer', 'right_ear'),
            ('mouth_left', 'mouth_right'),
            ('nose', 'mouth_left'),
            ('nose', 'mouth_right')
        ]

        # Draw connections
        for (start, end) in CONNECTIONS:
            if start in kps and end in kps:
                # Check if the keypoints have sufficient visibility
                if kps[start][3] > 0.5 and kps[end][3] > 0.5:
                    start_pt = (int(kps[start][0] * w), int(kps[start][1] * h))
                    end_pt = (int(kps[end][0] * w), int(kps[end][1] * h))
                    cv2.line(frame, start_pt, end_pt, color, 2)

        # Use MediaPipe's built-in drawing utilities for pose
        if hasattr(self, 'mp_drawing') and hasattr(entity, '_pose_results'):
            pose_results = getattr(entity, '_pose_results')
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

        # Draw face landmarks if available
        if hasattr(entity, '_face_results'):
            self._draw_face_landmarks(frame, entity)

    def _process_human(self, pose_results, face_results, roi_coords=None) -> Entity:
        """
        Process human detection results and extract key features

        Args:
            pose_results: MediaPipe pose detection results
            face_results: MediaPipe face detection results
            roi_coords: Optional tuple (x1, y1, x2, y2) of the region of interest in normalized coordinates
                        If provided, landmarks will be adjusted to the original frame coordinates
        """
        # Extract pose keypoints
        pose_landmarks = pose_results.pose_landmarks.landmark

        # Create a dictionary to store all 33 pose landmarks
        body_keypoints = {}

        # Map all pose landmarks to the keypoints dictionary
        for landmark_name, landmark_value in mp.solutions.pose.PoseLandmark.__members__.items():
            # Get the landmark coordinates
            x = pose_landmarks[landmark_value].x
            y = pose_landmarks[landmark_value].y
            z = pose_landmarks[landmark_value].z
            visibility = pose_landmarks[landmark_value].visibility

            # Adjust coordinates if ROI is provided
            if roi_coords:
                x1, y1, x2, y2 = roi_coords
                roi_width = x2 - x1
                roi_height = y2 - y1

                # Scale and translate coordinates to original frame
                x = x1 + (x * roi_width)
                y = y1 + (y * roi_height)

            body_keypoints[landmark_name.lower()] = [x, y, z, visibility]

        keypoints = {
            'body': body_keypoints,
            'face': {}
        }

        # Extract facial landmarks if available
        if face_results.multi_face_landmarks:
            # Select the best face if multiple faces are detected
            best_face_idx = self._select_best_face(face_results.multi_face_landmarks)
            face_landmarks = face_results.multi_face_landmarks[best_face_idx].landmark
            face_keypoints = {}

            # Extract key facial features
            # Eyes
            for i, eye_set in enumerate([mp.solutions.face_mesh.FACEMESH_LEFT_EYE, 
                                        mp.solutions.face_mesh.FACEMESH_RIGHT_EYE]):
                eye_name = 'left_eye' if i == 0 else 'right_eye'
                eye_points = {}
                for j, point in enumerate(eye_set):
                    idx = point[0]  # Get the landmark index

                    # Get the landmark coordinates
                    x = face_landmarks[idx].x
                    y = face_landmarks[idx].y
                    z = face_landmarks[idx].z

                    # Adjust coordinates if ROI is provided
                    if roi_coords:
                        x1, y1, x2, y2 = roi_coords
                        roi_width = x2 - x1
                        roi_height = y2 - y1

                        # Scale and translate coordinates to original frame
                        x = x1 + (x * roi_width)
                        y = y1 + (y * roi_height)

                    eye_points[f'point_{j}'] = [x, y, z]
                face_keypoints[eye_name] = eye_points

            # Eyebrows
            for i, eyebrow_set in enumerate([mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW, 
                                            mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW]):
                eyebrow_name = 'left_eyebrow' if i == 0 else 'right_eyebrow'
                eyebrow_points = {}
                for j, point in enumerate(eyebrow_set):
                    idx = point[0]  # Get the landmark index

                    # Get the landmark coordinates
                    x = face_landmarks[idx].x
                    y = face_landmarks[idx].y
                    z = face_landmarks[idx].z

                    # Adjust coordinates if ROI is provided
                    if roi_coords:
                        x1, y1, x2, y2 = roi_coords
                        roi_width = x2 - x1
                        roi_height = y2 - y1

                        # Scale and translate coordinates to original frame
                        x = x1 + (x * roi_width)
                        y = y1 + (y * roi_height)

                    eyebrow_points[f'point_{j}'] = [x, y, z]
                face_keypoints[eyebrow_name] = eyebrow_points

            # Lips
            lips_points = {}
            for j, point in enumerate(mp.solutions.face_mesh.FACEMESH_LIPS):
                idx = point[0]  # Get the landmark index

                # Get the landmark coordinates
                x = face_landmarks[idx].x
                y = face_landmarks[idx].y
                z = face_landmarks[idx].z

                # Adjust coordinates if ROI is provided
                if roi_coords:
                    x1, y1, x2, y2 = roi_coords
                    roi_width = x2 - x1
                    roi_height = y2 - y1

                    # Scale and translate coordinates to original frame
                    x = x1 + (x * roi_width)
                    y = y1 + (y * roi_height)

                lips_points[f'point_{j}'] = [x, y, z]
            face_keypoints['lips'] = lips_points

            # Nose
            nose_points = {}
            for j, point in enumerate(mp.solutions.face_mesh.FACEMESH_NOSE):
                idx = point[0]  # Get the landmark index

                # Get the landmark coordinates
                x = face_landmarks[idx].x
                y = face_landmarks[idx].y
                z = face_landmarks[idx].z

                # Adjust coordinates if ROI is provided
                if roi_coords:
                    x1, y1, x2, y2 = roi_coords
                    roi_width = x2 - x1
                    roi_height = y2 - y1

                    # Scale and translate coordinates to original frame
                    x = x1 + (x * roi_width)
                    y = y1 + (y * roi_height)

                nose_points[f'point_{j}'] = [x, y, z]
            face_keypoints['nose'] = nose_points

            keypoints['face'] = face_keypoints

        # Estimate emotions (simplified example)
        emotions = self._estimate_emotions(face_landmarks if face_results.multi_face_landmarks else None)

        # Use provided ROI coordinates as bounding box if available, otherwise calculate from landmarks
        if roi_coords:
            bbox = roi_coords
        else:
            bbox = self._calculate_bbox(pose_landmarks)

        # Create the entity
        entity = Entity(
            type="human",
            bbox=bbox,
            keypoints=keypoints,
            emotions=emotions,
            id=self._generate_entity_id(),
            species="human",
        )

        # Store the original MediaPipe results for drawing
        setattr(entity, '_pose_results', pose_results)

        # Store only the selected face for drawing
        if face_results.multi_face_landmarks:
            # Create a copy of face_results with only the selected face
            import copy
            selected_face_results = copy.deepcopy(face_results)
            selected_face_results.multi_face_landmarks = [face_results.multi_face_landmarks[best_face_idx]]
            setattr(entity, '_face_results', selected_face_results)

        return entity

    def _estimate_emotions(self, face_landmarks) -> dict:
        """Enhanced emotion estimation based on facial landmarks"""
        if not face_landmarks:
            return {"neutral": 1.0}

        # Initialize emotions with base values
        emotions = {
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "surprised": 0.0,
            "neutral": 0.5  # base value
        }

        try:
            # Calculate mouth openness (vertical distance between upper and lower lips)
            upper_lip_idx = 13  # Upper lip center
            lower_lip_idx = 14  # Lower lip center
            mouth_open = face_landmarks[lower_lip_idx].y - face_landmarks[upper_lip_idx].y

            # Calculate mouth width (horizontal distance between mouth corners)
            left_mouth_idx = 78  # Left mouth corner
            right_mouth_idx = 308  # Right mouth corner
            mouth_width = face_landmarks[right_mouth_idx].x - face_landmarks[left_mouth_idx].x

            # Calculate eyebrow positions
            left_eyebrow_idx = 65  # Left eyebrow
            right_eyebrow_idx = 295  # Right eyebrow
            left_eye_idx = 159  # Left eye center
            right_eye_idx = 386  # Right eye center

            # Eyebrow raise (distance from eyebrow to eye)
            left_eyebrow_raise = face_landmarks[left_eye_idx].y - face_landmarks[left_eyebrow_idx].y
            right_eyebrow_raise = face_landmarks[right_eye_idx].y - face_landmarks[right_eyebrow_idx].y
            avg_eyebrow_raise = (left_eyebrow_raise + right_eyebrow_raise) / 2

            # Eye openness
            left_eye_top_idx = 159  # Top of left eye
            left_eye_bottom_idx = 145  # Bottom of left eye
            right_eye_top_idx = 386  # Top of right eye
            right_eye_bottom_idx = 374  # Bottom of right eye

            left_eye_open = face_landmarks[left_eye_bottom_idx].y - face_landmarks[left_eye_top_idx].y
            right_eye_open = face_landmarks[right_eye_bottom_idx].y - face_landmarks[right_eye_top_idx].y
            avg_eye_open = (left_eye_open + right_eye_open) / 2

            # Enhanced emotion detection based on facial features

            # Surprised: raised eyebrows, open mouth, wide eyes
            if mouth_open > 0.05 and avg_eyebrow_raise > 0.03 and avg_eye_open > 0.02:
                emotions["surprised"] = 0.8
                emotions["neutral"] = 0.2

            # Happy: raised cheeks, mouth corners up, slight mouth open
            elif mouth_width > 0.3 and mouth_open > 0.02:
                emotions["happy"] = 0.8
                emotions["neutral"] = 0.2

            # Sad: drooping mouth corners, lowered eyebrows
            elif mouth_width < 0.25 and avg_eyebrow_raise < 0.02:
                emotions["sad"] = 0.7
                emotions["neutral"] = 0.3

            # Angry: lowered eyebrows, tight mouth
            elif avg_eyebrow_raise < 0.01 and mouth_width < 0.2:
                emotions["angry"] = 0.7
                emotions["neutral"] = 0.3

            # If no strong emotion is detected, increase neutral
            else:
                emotions["neutral"] = 0.9

        except (IndexError, AttributeError) as e:
            # If there's an error in emotion estimation, return neutral
            return {"neutral": 1.0}

        # Normalize to sum to 1.0
        total = sum(emotions.values())
        return {k: v / total for k, v in emotions.items()}

    def _draw_face_landmarks(self, frame, entity):
        """Draw facial landmarks using MediaPipe's built-in drawing utilities"""
        if not hasattr(self, 'mp_drawing') or not hasattr(entity, '_face_results'):
            return

        face_results = getattr(entity, '_face_results')
        if not face_results.multi_face_landmarks:
            return

        # Draw face mesh
        for face_landmarks in face_results.multi_face_landmarks:
            # Draw the face mesh tesselation (triangles)
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw the face mesh contours
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            # Draw the irises if available
            if hasattr(mp.solutions.face_mesh, 'FACEMESH_IRISES'):
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

    def _calculate_bbox(self, landmarks) -> tuple:
        """Calculate bounding box from pose landmarks"""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        # Add 10% padding
        width = max_x - min_x
        height = max_y - min_y
        min_x = max(0, min_x - width * 0.1)
        max_x = min(1, max_x + width * 0.1)
        min_y = max(0, min_y - height * 0.1)
        max_y = min(1, max_y + height * 0.1)

        return (min_x, min_y, max_x, max_y)

    def _generate_entity_id(self) -> int:
        """Generate unique ID for each entity"""
        if not hasattr(self, '_id_counter'):
            self._id_counter = 0
        self._id_counter += 1
        return self._id_counter

    def _select_best_face(self, face_landmarks_list):
        """
        Select the best face from multiple detected faces.

        Args:
            face_landmarks_list: List of face landmarks from MediaPipe

        Returns:
            int: Index of the best face in the list
        """
        # If only one face is detected, return its index
        if len(face_landmarks_list) == 1:
            return 0

        # Calculate the center and size of each face
        face_centers = []
        face_sizes = []
        for face_landmarks in face_landmarks_list:
            # Get all x and y coordinates
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]

            # Calculate center
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            face_centers.append((center_x, center_y))

            # Calculate face size (width and height)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            width = max_x - min_x
            height = max_y - min_y
            face_sizes.append((width, height))

        # The center of the ROI is (0.5, 0.5) in normalized coordinates
        roi_center = (0.5, 0.5)

        # Define the valid region (slightly smaller than the full ROI to ensure face is fully inside)
        valid_region = (0.1, 0.1, 0.9, 0.9)  # (x1, y1, x2, y2)

        # Find faces that are within the valid region
        valid_faces = []
        for i, (center_x, center_y) in enumerate(face_centers):
            if (valid_region[0] <= center_x <= valid_region[2] and 
                valid_region[1] <= center_y <= valid_region[3]):
                valid_faces.append(i)

        # If there are valid faces, select the one closest to the center
        if valid_faces:
            # Calculate distance from each valid face center to ROI center
            distances = []
            for i in valid_faces:
                center_x, center_y = face_centers[i]
                distance = ((center_x - roi_center[0]) ** 2 + (center_y - roi_center[1]) ** 2) ** 0.5
                distances.append((i, distance))

            # Sort by distance and return the index of the closest face
            distances.sort(key=lambda x: x[1])
            return distances[0][0]
        else:
            # If no faces are within the valid region, select the one closest to the center
            distances = []
            for i, (center_x, center_y) in enumerate(face_centers):
                distance = ((center_x - roi_center[0]) ** 2 + (center_y - roi_center[1]) ** 2) ** 0.5
                distances.append((i, distance))

            # Sort by distance and return the index of the closest face
            distances.sort(key=lambda x: x[1])
            return distances[0][0]

    def _validate_entity(self, entity) -> bool:
        """
        Validate entity based on size and position

        Args:
            entity: The entity to validate

        Returns:
            bool: True if entity is valid, False otherwise
        """
        if not hasattr(entity, 'bbox'):
            return False

        x1, y1, x2, y2 = entity.bbox

        # Calculate width and height
        width = x2 - x1
        height = y2 - y1

        # Validate size
        if width < self.settings.MIN_ENTITY_SIZE or height < self.settings.MIN_ENTITY_SIZE:
            return False

        if width > self.settings.MAX_ENTITY_SIZE or height > self.settings.MAX_ENTITY_SIZE:
            return False

        # Validate position (not too close to edges)
        margin = self.settings.VALID_POSITION_MARGIN
        if x1 < margin or y1 < margin or x2 > (1 - margin) or y2 > (1 - margin):
            return False

        return True

    def _process_humans(self, rgb_frame, yolo_results) -> list[Entity]:
        """
        Process YOLOv8 results to extract and process human entities.

        This method:
        1. Detects humans using YOLOv8
        2. Crops the region of interest for each human
        3. Processes each cropped region with MediaPipe for pose and face detection
        4. Calculates the normalized coordinates of the cropped region
        5. Passes the MediaPipe results and normalized coordinates to _process_human

        The number of humans detected is limited by MP_MAX_NUM_FACES in settings.py.
        The face coordinates are correctly transformed from the cropped region to the original frame.
        """
        entities = []

        # Human class ID in YOLOv8 (COCO dataset)
        HUMAN_CLASS_ID = 0

        # Get original frame dimensions
        h, w = rgb_frame.shape[:2]

        # Process each detection
        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls)

                # Skip if not human
                if class_id != HUMAN_CLASS_ID:
                    continue

                # Get bounding box coordinates (normalized to 0-1)
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                conf = float(box.conf)

                # Only process high-confidence detections (increased threshold to reduce false positives)
                if conf < 0.65:
                    continue

                # Convert normalized coordinates to pixel coordinates
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)

                # Calculate padding proportional to the size of the detection
                width_px = x2_px - x1_px
                height_px = y2_px - y1_px
                padding_x = int(width_px * 0.1)  # 10% of width
                padding_y = int(height_px * 0.1)  # 10% of height

                # Add padding to ensure the whole person is captured
                x1_px = max(0, x1_px - padding_x)
                y1_px = max(0, y1_px - padding_y)
                x2_px = min(w, x2_px + padding_x)
                y2_px = min(h, y2_px + padding_y)

                # Crop the region of interest
                human_roi = rgb_frame[y1_px:y2_px, x1_px:x2_px]

                # Skip if ROI is empty
                if human_roi.size == 0:
                    continue

                # Process with MediaPipe
                pose_results = self.pose.process(human_roi)
                face_results = self.face.process(human_roi)

                # Skip if no pose landmarks detected
                if not pose_results.pose_landmarks:
                    continue

                # Calculate normalized coordinates of the cropped region relative to the original frame
                x1_norm = x1_px / w
                y1_norm = y1_px / h
                x2_norm = x2_px / w
                y2_norm = y2_px / h

                # Process human with MediaPipe results and normalized coordinates of the cropped region
                human_entity = self._process_human(pose_results, face_results, (x1_norm, y1_norm, x2_norm, y2_norm))

                # Add to entities list
                entities.append(human_entity)

        return entities

    def _process_animals(self, yolo_results) -> list[Entity]:
        """Process YOLOv8 results to extract animal entities"""
        entities = []

        # Animal class IDs in YOLOv8 (COCO dataset classes)
        ANIMAL_CLASSES = {
            15: "bird", 16: "cat", 17: "dog",
            18: "horse", 19: "sheep", 20: "cow",
            21: "elephant", 22: "bear", 23: "zebra",
            24: "giraffe"
        }

        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id not in ANIMAL_CLASSES:
                    continue

                # Get bounding box coordinates (normalized to 0-1)
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                conf = float(box.conf)

                # Only process high-confidence detections
                if conf < 0.5:
                    continue

                # Estimate keypoints (simplified for animals)
                keypoints = self._estimate_animal_keypoints(x1, y1, x2, y2, class_id)

                entities.append(Entity(
                    type="animal",
                    species=ANIMAL_CLASSES[class_id],
                    bbox=(x1, y1, x2, y2),
                    keypoints=keypoints,
                    emotions={"neutral": 1.0},  # Animals get neutral emotion by default
                    id=self._generate_entity_id()
                ))

        return entities

    def _estimate_animal_keypoints(self, x1, y1, x2, y2, class_id) -> dict:
        """Estimate approximate keypoints for animals based on bounding box"""
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        # Generic quadruped keypoint estimation (adjust per animal type)
        keypoints = {
            "head": [x1 + width * 0.8, y1 + height * 0.2],
            "neck": [x1 + width * 0.7, y1 + height * 0.3],
            "front_shoulders": [x1 + width * 0.6, y1 + height * 0.4],
            "back_hips": [x1 + width * 0.4, y1 + height * 0.6],
            "tail": [x1 + width * 0.2, y1 + height * 0.7],
            # Legs (approximate positions)
            "front_left_leg": [x1 + width * 0.55, y1 + height * 0.8],
            "front_right_leg": [x1 + width * 0.65, y1 + height * 0.8],
            "back_left_leg": [x1 + width * 0.35, y1 + height * 0.9],
            "back_right_leg": [x1 + width * 0.45, y1 + height * 0.9]
        }

        # Special cases for different animal types
        if ANIMAL_CLASSES[class_id] == "bird":
            keypoints.update({
                "beak": [x1 + width * 0.9, y1 + height * 0.15],
                "wings": [x1 + width * 0.5, y1 + height * 0.3],
                "tail": [x1 + width * 0.1, y1 + height * 0.5]
            })
        elif ANIMAL_CLASSES[class_id] == "fish":
            keypoints = {
                "head": [x1 + width * 0.8, center_y],
                "tail": [x1 + width * 0.2, center_y],
                "dorsal_fin": [center_x, y1 + height * 0.3]
            }

        return {
            "body": keypoints,
            "face": {}  # Animals don't get facial landmarks in this basic version
        }
