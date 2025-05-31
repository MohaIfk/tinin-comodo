import requests
import json
import os
import time
import cv2
import numpy as np
import hashlib
from typing import Dict, Any, Optional
from utils.logger import app_logger as logger


class AvatarGenerator:
    def __init__(self, settings):
        self._settings = settings  # Store settings object for later use
        self.api_url = settings.AVATAR_API_URL
        self.api_key = settings.AVATAR_API_KEY
        self.avatars = {}  # Stores generated avatars
        self.timeout = settings.AVATAR_API_TIMEOUT
        self.cache_expiration = 86400  # 24 hours in seconds

        # Default avatars for different entity types
        self.default_avatars = {
            "human": {
                "status": "fallback",
                "avatar": {
                    "url": "https://models.readyplayer.me/683716c2b8857cbb3dbbcf7e.glb",
                    "features": {}
                }
            },
            "animal": {
                "status": "fallback",
                "avatar": {
                    "url": "https://models.readyplayer.me/default_animal.glb",
                    "features": {}
                }
            }
        }

        # Create directories if they don't exist
        self.cache_dir = os.path.join("assets", "avatars")
        self.ref_images_dir = os.path.join("assets", "reference_images")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.ref_images_dir, exist_ok=True)

        # Track captured reference images
        self.reference_images = {}

        logger.info("AvatarGenerator initialized")

    def get_avatar(self, entity, frame=None) -> Dict[str, Any]:
        """
        Get or create avatar for entity

        Args:
            entity: The detected entity
            frame: Optional frame for capturing reference image
        """
        try:
            # Check if we already have an avatar for this entity
            if entity.id in self.avatars:
                # Check if we need to update the avatar (e.g., if emotions changed significantly)
                current_avatar = self.avatars[entity.id]
                if self._should_update_avatar(entity, current_avatar):
                    logger.debug(f"Updating avatar for entity {entity.id}")

                    # Capture reference image if frame is provided
                    ref_image_path = None
                    if frame is not None:
                        ref_image_path = self.capture_reference_image(frame, entity)

                    avatar_data = self._generate_avatar(entity, ref_image_path)
                    self.avatars[entity.id] = avatar_data
                return self.avatars[entity.id]
            else:
                # Generate a new avatar
                logger.debug(f"Generating new avatar for entity {entity.id} of type {entity.type}")

                # Capture reference image if frame is provided
                ref_image_path = None
                if frame is not None:
                    ref_image_path = self.capture_reference_image(frame, entity)

                avatar_data = self._generate_avatar(entity, ref_image_path)
                self.avatars[entity.id] = avatar_data
                return avatar_data
        except Exception as e:
            logger.error(f"Error getting avatar for entity {entity.id}: {str(e)}")
            # Return appropriate default avatar based on entity type
            return self._get_default_avatar(entity.type)

    def _should_update_avatar(self, entity, current_avatar) -> bool:
        """Determine if avatar should be updated based on changes in entity"""
        # For now, we don't update avatars once created
        # In a more advanced implementation, we could check for significant changes
        # in emotions or other features
        return False

    def _get_default_avatar(self, entity_type) -> Dict[str, Any]:
        """Get the appropriate default avatar based on entity type"""
        if entity_type in self.default_avatars:
            return self.default_avatars[entity_type]
        return self.default_avatars["human"]  # Fallback to human if type not recognized

    def _generate_avatar(self, entity, ref_image_path=None):
        """
        Call external API to generate avatar based on entity type

        Args:
            entity: The detected entity
            ref_image_path: Optional path to reference image
        """
        try:
            # Check if we have a valid API key before making the request
            if not hasattr(self, '_settings') or not self._settings.has_valid_api_key:
                logger.warning(f"No valid API key available for entity {entity.id}")
                return self._get_default_avatar(entity.type)

            # Generate cache key including reference image info
            cache_key = self._generate_cache_key(entity, ref_image_path)

            # Check if we have a cached avatar for this entity type and features
            cached_avatar = self._check_cache(cache_key)
            if cached_avatar:
                logger.debug(f"Using cached avatar for entity {entity.id}")
                return cached_avatar

            # Prepare payload based on entity type
            if entity.type == "human":
                payload = self._prepare_human_payload(entity)
            elif entity.type == "animal":
                payload = self._prepare_animal_payload(entity)
            else:
                logger.warning(f"Unknown entity type: {entity.type}")
                return self._get_default_avatar(entity.type)

            # Make API request
            headers = {"Authorization": f"Bearer {self.api_key}"}
            logger.debug(f"Sending request to {self.api_url} for entity {entity.id}")

            # Prepare for multipart request if reference image is provided
            if ref_image_path and os.path.exists(ref_image_path):
                logger.debug(f"Including reference image in request: {ref_image_path}")

                # Convert payload to form data
                files = {
                    'reference_image': (os.path.basename(ref_image_path), open(ref_image_path, 'rb'), 'image/jpeg'),
                    'payload': ('payload.json', json.dumps(payload), 'application/json')
                }

                # Make multipart request
                response = requests.post(
                    self.api_url,
                    files=files,
                    headers=headers,
                    timeout=self.timeout
                )
            else:
                # Make JSON request without reference image
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )

            response.raise_for_status()  # Raise exception for bad status codes

            # Process and cache the response
            avatar_data = response.json()

            # Ensure the response contains a URL to a 3D model file (.glb or .fbx)
            if "avatar" not in avatar_data:
                avatar_data["avatar"] = {}

            if "url" not in avatar_data["avatar"]:
                # Check if the API response contains a model URL directly
                if "model_url" in avatar_data:
                    avatar_data["avatar"]["url"] = avatar_data["model_url"]
                elif "url" in avatar_data:
                    avatar_data["avatar"]["url"] = avatar_data["url"]
                else:
                    # Fallback to a default URL based on entity type
                    if entity.type == "human":
                        avatar_data["avatar"]["url"] = "https://models.readyplayer.me/default_human.glb"
                    else:
                        avatar_data["avatar"]["url"] = f"https://models.readyplayer.me/default_{entity.species}.glb"
                    logger.warning(f"No model URL found in API response for entity {entity.id}, using fallback URL")

            # Ensure the URL points to a .glb or .fbx file
            url = avatar_data["avatar"]["url"]
            if not url.endswith(".glb") and not url.endswith(".fbx"):
                # Append .glb extension if missing
                avatar_data["avatar"]["url"] = f"{url}.glb"
                logger.warning(f"Model URL does not have a valid extension, appending .glb: {url}")

            # Add reference to the reference image if used
            if ref_image_path:
                avatar_data["reference_image"] = ref_image_path

            self._cache_avatar(cache_key, avatar_data)

            return avatar_data
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out while generating avatar for entity {entity.id}")
            return self._get_default_avatar(entity.type)
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while generating avatar: {str(e)}")
            return self._get_default_avatar(entity.type)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response received for entity {entity.id}")
            return self._get_default_avatar(entity.type)
        except Exception as e:
            logger.error(f"Unexpected error while generating avatar: {str(e)}")
            return self._get_default_avatar(entity.type)

    def _prepare_human_payload(self, entity) -> Dict[str, Any]:
        """Prepare payload for human avatar generation"""
        # Extract facial features for more accurate avatar
        face_features = {}
        if 'face' in entity.keypoints and entity.keypoints['face']:
            face_features = entity.keypoints['face']

        # Extract body measurements
        body_features = {}
        if 'body' in entity.keypoints and entity.keypoints['body']:
            # Calculate height based on keypoints
            body_features = {
                "height": self._estimate_height(entity.keypoints['body']),
                "build": self._estimate_build(entity.keypoints['body'])
            }

        return {
            "type": "human",
            "gender": "neutral",  # Default to neutral, could be inferred from body shape
            "features": {
                "body": body_features,
                "face": face_features,
                "emotions": entity.emotions
            },
            "style": "stylized"  # As per requirements, generate stylized avatars
        }

    def _prepare_animal_payload(self, entity) -> Dict[str, Any]:
        """Prepare payload for animal avatar generation"""
        return {
            "type": "animal",
            "species": entity.species,
            "features": {
                "body": entity.keypoints['body'] if 'body' in entity.keypoints else {},
            },
            "style": "stylized"  # As per requirements, generate stylized avatars
        }

    def _estimate_height(self, body_keypoints) -> float:
        """Estimate height from body keypoints"""
        # This is a placeholder. In a real implementation, we would use
        # the distance between key points to estimate height
        return 1.75  # Default height in meters

    def _estimate_build(self, body_keypoints) -> str:
        """Estimate body build from keypoints"""
        # This is a placeholder. In a real implementation, we would analyze
        # the proportions of the body to determine build
        return "average"  # Default build

    def _generate_cache_key(self, entity, ref_image_path=None) -> str:
        """
        Generate a unique cache key for an entity

        Args:
            entity: The detected entity
            ref_image_path: Optional path to reference image
        """
        # Start with basic entity information
        entity_type = entity.type
        species = getattr(entity, 'species', 'human')

        # Generate a feature hash for more uniqueness
        feature_hash = self._hash_entity_features(entity)

        # Include reference image information if available
        ref_image_part = ""
        if ref_image_path:
            # Extract just the filename without path
            ref_image_filename = os.path.basename(ref_image_path)
            # Use first 10 chars of filename (which includes the feature hash)
            ref_image_part = f"_ref_{ref_image_filename[:10]}"

        # Include emotion information for humans
        emotion_part = ""
        if entity_type == "human" and hasattr(entity, 'emotions') and entity.emotions:
            # Get the dominant emotion
            dominant_emotion = max(entity.emotions.items(), key=lambda x: x[1])[0]
            emotion_part = f"_emo_{dominant_emotion}"

        # Combine all parts into a unique key
        return f"{entity_type}_{species}_{feature_hash}{ref_image_part}{emotion_part}"

    def _check_cache(self, cache_key) -> Dict[str, Any] or None:
        """Check if we have a cached avatar for this cache key"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                # Check if cache has expired
                if "cache_timestamp" in cached_data:
                    cache_time = cached_data["cache_timestamp"]
                    current_time = time.time()
                    if current_time - cache_time > self.cache_expiration:
                        logger.debug(f"Cache expired for {cache_key}")
                        return None

                return cached_data
            except Exception as e:
                logger.error(f"Error loading cached avatar: {str(e)}")
        return None

    def _cache_avatar(self, cache_key, avatar_data) -> None:
        """Cache avatar data to disk"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            # Add timestamp to cache data for expiration
            avatar_data["cache_timestamp"] = time.time()
            with open(cache_file, 'w') as f:
                json.dump(avatar_data, f)
        except Exception as e:
            logger.error(f"Error caching avatar: {str(e)}")

    def capture_reference_image(self, frame, entity) -> Optional[str]:
        """
        Capture a reference image of the entity from the frame

        Args:
            frame: The full camera frame
            entity: The detected entity

        Returns:
            Path to the saved reference image or None if failed
        """
        try:
            if frame is None or entity is None or not hasattr(entity, 'bbox'):
                logger.warning("Cannot capture reference image: invalid frame or entity")
                return None

            # Get frame dimensions
            h, w = frame.shape[:2]

            # Extract bounding box coordinates
            x1, y1, x2, y2 = entity.bbox

            # Convert normalized coordinates to pixel coordinates
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)

            # Ensure coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Check if bounding box is valid
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bounding box for entity {entity.id}: {entity.bbox}")
                return None

            # Crop the image to the bounding box
            ref_image = frame[y1:y2, x1:x2]

            # Generate a unique filename based on entity properties
            entity_type = entity.type
            species = getattr(entity, 'species', 'unknown')
            timestamp = int(time.time())

            # Create a hash of entity features for uniqueness
            features_hash = self._hash_entity_features(entity)

            # Create filename
            filename = f"{entity_type}_{species}_{features_hash}_{timestamp}.jpg"
            filepath = os.path.join(self.ref_images_dir, filename)

            # Save the image
            cv2.imwrite(filepath, ref_image)

            # Store reference to the image
            self.reference_images[entity.id] = filepath

            logger.debug(f"Captured reference image for entity {entity.id}: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error capturing reference image: {str(e)}")
            return None

    def _hash_entity_features(self, entity) -> str:
        """Generate a hash based on entity features for unique identification"""
        # Create a string representation of key entity features
        feature_str = f"{entity.type}_{getattr(entity, 'species', 'unknown')}"

        # Add keypoints if available
        if hasattr(entity, 'keypoints') and entity.keypoints:
            # Add a few key points to the feature string
            if 'body' in entity.keypoints:
                body = entity.keypoints['body']
                # Take a sample of keypoints to keep the hash stable but unique
                for key in sorted(list(body.keys())[:3]):  # Use first 3 keypoints
                    if isinstance(body[key], dict):
                        continue  # Skip nested keypoints
                    if body[key]:
                        feature_str += f"_{key}_{body[key][0]:.2f}_{body[key][1]:.2f}"

        # Generate MD5 hash of the feature string
        return hashlib.md5(feature_str.encode()).hexdigest()[:10]
