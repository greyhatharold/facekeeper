"""
Face detection module optimized for Raspberry Pi 4.
Provides functionality to detect faces in images using various detection models
with configuration-based optimization options.
"""

import cv2
import face_recognition
import numpy as np
import yaml
import logging
from typing import List, Tuple, Optional
from time import time
import dlib

class FaceDetector:
    """Handles face detection with configurable parameters and optimization."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize face detector with configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.last_detection_time = 0
        self.frame_count = 0
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['storage']['log_level']),
            filename=self.config['storage']['log_file']
        )
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load detection configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration parameters
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
            
    def _should_process_frame(self) -> bool:
        """
        Determine if the current frame should be processed based on performance settings.
        
        Returns:
            bool: True if frame should be processed, False otherwise
        """
        self.frame_count += 1
        return self.frame_count % self.config['performance']['process_every_n_frames'] == 0
        
    def _resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame based on scale factor for performance optimization.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Resized frame
        """
        scale_factor = self.config['detection']['scale_factor']
        if scale_factor < 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return cv2.resize(frame, (new_width, new_height))
        return frame
        
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given frame using dlib.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            
        Returns:
            List[Tuple[int, int, int, int]]: List of face locations in (top, right, bottom, left) format
        """
        if not self._should_process_frame():
            return []
            
        try:
            # Resize for performance if needed
            detection_frame = self._resize_frame_if_needed(frame)
            
            # Use dlib's face detector
            if self.config['detection']['model'] == 'dlib_cnn':
                # Use CNN detector if specified
                detector = dlib.cnn_face_detection_model_v1(
                    'models/mmod_human_face_detector.dat'
                )
                dets = detector(detection_frame)
                face_locations = [
                    (d.rect.top(), d.rect.right(), 
                     d.rect.bottom(), d.rect.left()) for d in dets
                ]
            else:
                # Use HOG detector by default
                detector = dlib.get_frontal_face_detector()
                dets = detector(detection_frame)
                face_locations = [
                    (d.top(), d.right(), d.bottom(), d.left()) 
                    for d in dets
                ]
            
            # If frame was resized, scale coordinates back to original size
            if self.config['detection']['scale_factor'] < 1.0:
                scale = 1.0 / self.config['detection']['scale_factor']
                face_locations = [
                    (int(top * scale), int(right * scale),
                     int(bottom * scale), int(left * scale))
                    for top, right, bottom, left in face_locations
                ]
            
            # Limit number of faces if specified
            max_faces = self.config['detection']['max_faces']
            if max_faces > 0:
                face_locations = face_locations[:max_faces]
                
            self.last_detection_time = time()
            return face_locations
            
        except Exception as e:
            logging.error(f"Error during face detection: {e}")
            return []
            
    def draw_faces(self, frame: np.ndarray, 
                  face_locations: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            frame (np.ndarray): Input frame
            face_locations (List[Tuple[int, int, int, int]]): List of face locations
            
        Returns:
            np.ndarray: Frame with drawn bounding boxes
        """
        annotated_frame = frame.copy()
        box_color = tuple(self.config['display']['box_color'])  # Convert RGB to BGR
        
        for top, right, bottom, left in face_locations:
            # Draw bounding box
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), box_color, 2)
            
            # Optionally add additional annotations here
            if self.config['display']['show_names']:
                cv2.putText(
                    annotated_frame,
                    "Face",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config['display']['font_scale'],
                    box_color,
                    self.config['display']['font_thickness']
                )
                
        return annotated_frame

def create_detector() -> Optional[FaceDetector]:
    """
    Factory function to create and initialize a FaceDetector instance.
    
    Returns:
        Optional[FaceDetector]: Initialized face detector or None if initialization fails
    """
    try:
        return FaceDetector()
    except Exception as e:
        logging.error(f"Failed to create face detector: {e}")
        return None