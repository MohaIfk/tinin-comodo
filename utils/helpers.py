import os
import json
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


def ensure_directory(directory_path: str) -> bool:
    """Ensure a directory exists, creating it if necessary"""
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False


def load_json_file(file_path: str, default: Any = None) -> Any:
    """Load JSON data from a file with error handling"""
    try:
        if not os.path.exists(file_path):
            return default
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {str(e)}")
        return default


def save_json_file(file_path: str, data: Any) -> bool:
    """Save data to a JSON file with error handling"""
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {str(e)}")
        return False


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize an image while maintaining aspect ratio"""
    if width is None and height is None:
        return image
        
    h, w = image.shape[:2]
    if width is None:
        aspect = height / float(h)
        dim = (int(w * aspect), height)
    else:
        aspect = width / float(w)
        dim = (width, int(h * aspect))
        
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def draw_text(image: np.ndarray, text: str, position: Tuple[int, int], 
              font_scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255),
              thickness: int = 1) -> np.ndarray:
    """Draw text on an image with a background for better visibility"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw background rectangle
    bg_color = (0, 0, 0)
    cv2.rectangle(
        image, 
        (position[0], position[1] - text_size[1] - 5),
        (position[0] + text_size[0], position[1] + 5),
        bg_color, 
        -1
    )
    
    # Draw text
    cv2.putText(
        image, 
        text, 
        position, 
        font, 
        font_scale, 
        color, 
        thickness
    )
    
    return image


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points"""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def normalize_points(points: List[Tuple[float, float]], image_width: int, image_height: int) -> List[Tuple[float, float]]:
    """Normalize points to 0-1 range based on image dimensions"""
    return [(x / image_width, y / image_height) for x, y in points]
