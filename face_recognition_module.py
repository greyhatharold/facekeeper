"""
Face recognition module optimized for Raspberry Pi 4.
Provides functionality for face encoding and matching using dlib and face_recognition libraries.
"""

import face_recognition
import numpy as np
import yaml
import logging
import dlib
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path
import json

class FaceRecognizer:
    """Handles face encoding and recognition with configurable parameters."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize face recognizer with configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        
        # Initialize dlib's face detector and shape predictor
        self.face_detector = dlib.get_frontal_face_detector()
        model_path = self.config['recognition'].get('shape_predictor_path', 
                                                  'shape_predictor_68_face_landmarks.dat')
        self.shape_predictor = dlib.shape_predictor(model_path)
        self.face_encoder = dlib.face_recognition_model_v1(
            self.config['recognition'].get('face_recognition_model_path',
                                         'dlib_face_recognition_resnet_model_v1.dat')
        )
        
        self._load_known_faces()
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config['storage']['log_level']),
            filename=self.config['storage']['log_file']
        )
        
    def _load_config(self, config_path: str) -> dict:
        """Load recognition configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise
            
    def _load_known_faces(self):
        """Load known face encodings from storage."""
        try:
            known_faces_path = Path(self.config['recognition']['known_faces_path'])
            if known_faces_path.exists():
                with open(known_faces_path, 'r') as file:
                    data = json.load(file)
                    self.known_face_names = data['names']
                    # Convert string representations back to numpy arrays
                    self.known_face_encodings = [
                        np.array(encoding) for encoding in data['encodings']
                    ]
                logging.info(f"Loaded {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")
            
    def save_known_faces(self):
        """Save known face encodings to storage."""
        try:
            known_faces_path = Path(self.config['recognition']['known_faces_path'])
            known_faces_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'names': self.known_face_names,
                'encodings': [encoding.tolist() for encoding in self.known_face_encodings]
            }
            
            with open(known_faces_path, 'w') as file:
                json.dump(data, file)
                
            logging.info(f"Saved {len(self.known_face_names)} known faces")
        except Exception as e:
            logging.error(f"Error saving known faces: {e}")
            
    def encode_faces(self, frame: np.ndarray, 
                    face_locations: List[Tuple[int, int, int, int]]
                    ) -> List[np.ndarray]:
        """
        Generate face encodings for detected faces in the frame using dlib.
        
        Args:
            frame (np.ndarray): Input frame in BGR format
            face_locations (List[Tuple[int, int, int, int]]): List of face locations
            
        Returns:
            List[np.ndarray]: List of 128-dimensional face encodings
        """
        try:
            encodings = []
            
            # Convert face_locations to dlib rectangles
            dlib_rectangles = [
                dlib.rectangle(left=loc[3], top=loc[0], right=loc[1], bottom=loc[2])
                for loc in face_locations
            ]
            
            for rect in dlib_rectangles:
                # Get facial landmarks
                shape = self.shape_predictor(frame, rect)
                # Compute face encoding
                face_encoding = np.array(
                    self.face_encoder.compute_face_descriptor(frame, shape)
                )
                encodings.append(face_encoding)
            
            return encodings
            
        except Exception as e:
            logging.error(f"Error encoding faces: {e}")
            return []
            
    def match_face(self, encoding: np.ndarray, 
                  threshold: Optional[float] = None) -> Tuple[str, float]:
        """Match a face encoding against known face encodings."""
        if not self.known_face_encodings:
            return "unknown", 1.0
            
        try:
            # Use a more lenient threshold by default
            if threshold is None:
                threshold = 0.6  # Increased from default (typically 0.4-0.5)
                
            # Calculate Euclidean distances to all known faces
            face_distances = [np.linalg.norm(encoding - known_enc) 
                            for known_enc in self.known_face_encodings]
            
            # Find best match
            best_match_index = np.argmin(face_distances)
            min_distance = face_distances[best_match_index]
            
            # Return match with confidence score
            if min_distance <= threshold:
                confidence = min_distance / threshold  # Normalize confidence score
                return self.known_face_names[best_match_index], confidence
                
            return "unknown", 1.0
            
        except Exception as e:
            logging.error(f"Error matching face: {e}")
            return "unknown", 1.0
            
    def add_known_face(self, name: str, encoding: np.ndarray) -> bool:
        """
        Add a new known face to the database.
        
        Args:
            name (str): Name of the person
            encoding (np.ndarray): 128-dimensional face encoding
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            self.save_known_faces()
            return True
        except Exception as e:
            logging.error(f"Error adding known face: {e}")
            return False

def create_recognizer() -> Optional[FaceRecognizer]:
    """
    Factory function to create and initialize a FaceRecognizer instance.
    
    Returns:
        Optional[FaceRecognizer]: Initialized face recognizer or None if initialization fails
    """
    try:
        return FaceRecognizer()
    except Exception as e:
        logging.error(f"Failed to create face recognizer: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from face_detection import create_detector
    from camera_capture import create_camera
    
    recognizer = create_recognizer()
    detector = create_detector()
    camera = create_camera()
    
    if all((recognizer, detector, camera)):
        try:
            frame = camera.get_frame()
            if frame is not None:
                # Detect faces
                face_locations = detector.detect_faces(frame)
                
                # Generate encodings
                encodings = recognizer.encode_faces(frame, face_locations)
                
                # Match each face
                for encoding in encodings:
                    name, confidence = recognizer.match_face(encoding)
                    print(f"Detected: {name} (confidence: {1-confidence:.2f})")
        finally:
            camera.release() 