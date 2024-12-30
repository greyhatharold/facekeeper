"""
Face enrollment module for Raspberry Pi face recognition system.
Handles the process of capturing, encoding, and storing new face data.
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import face_recognition
import dlib
import logging

from camera_capture import create_camera
from face_detection import create_detector
from face_recognition_module import create_recognizer
from data_storage import FaceDataStorage


class FaceEnrollment:
    """Handles the enrollment of new faces into the recognition system."""
    
    def __init__(self):
        """Initialize components needed for face enrollment."""
        self.camera = create_camera()
        self.detector = create_detector()
        self.recognizer = create_recognizer()
        self.storage = FaceDataStorage()
        
        # Enhanced enrollment configuration
        self.num_samples = 10
        self.capture_interval = 0.3
        self.min_required_samples = 7
        self.quality_threshold = 0.8
        
        # Adjusted quality check parameters for better balance
        self.min_face_size = 100  # Reduced from 150 for easier face detection
        self.max_angle_deviation = 20  # Increased from 15 for more flexibility
        self.brightness_threshold = 30  # Reduced from 40 for lower light conditions
        self.blur_threshold = 50  # Reduced from 100 for more lenient blur detection
        
        # Add adaptive blur threshold based on image size
        self.min_blur_threshold = 30
        self.max_blur_threshold = 150

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
        
    def enroll_new_face(self, name: str, status_callback=None) -> bool:
        """
        Enroll a new face by capturing multiple samples and storing the encoding.
        
        Args:
            name: Name or ID of the person being enrolled
            status_callback: Optional callback function to report status messages
            
        Returns:
            bool: True if enrollment was successful, False otherwise
        """
        def update_status(msg):
            if status_callback:
                status_callback(msg)
            print(msg)
        
        update_status(f"\nStarting enrollment for {name}")
        update_status("Please look at the camera and keep your face centered...")
        
        face_encodings = self._capture_face_samples(status_callback=update_status)
        
        if not face_encodings:
            update_status("Failed to capture any valid face samples.")
            return False
            
        if len(face_encodings) < self.min_required_samples:
            print(f"Only captured {len(face_encodings)} valid samples. "
                  f"Need at least {self.min_required_samples}.")
            return False
        
        # Average the encodings for more robust recognition
        final_encoding = np.mean(face_encodings, axis=0)
        
        # Load existing data
        known_names, known_encodings = self.storage.load_known_faces()
        
        # Check if name already exists
        if name in known_names:
            response = input(f"Name '{name}' already exists. Update? (y/n): ")
            if response.lower() != 'y':
                return False
            # Replace existing encoding
            idx = known_names.index(name)
            known_encodings[idx] = final_encoding
        else:
            # Add new encoding
            known_names.append(name)
            known_encodings.append(final_encoding)
        
        # Save updated data
        if self.storage.save_known_faces(known_names, known_encodings):
            print(f"\nSuccessfully enrolled {name}!")
            return True
        else:
            print("Error saving face data.")
            return False
            
    def _check_face_quality(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> bool:
        """
        Check if the detected face meets quality standards.
        
        Args:
            frame: Input frame containing the face
            face_location: Tuple of (top, right, bottom, left) coordinates
            
        Returns:
            bool: True if face meets quality standards, False otherwise
        """
        try:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Dynamic blur threshold based on face size
            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height
            dynamic_blur_threshold = max(
                self.min_blur_threshold,
                min(self.blur_threshold * (face_area / (640 * 480)), self.max_blur_threshold)
            )
            
            # Check blur with dynamic threshold
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            
            if laplacian_var < dynamic_blur_threshold:
                print(f"Image too blurry ({laplacian_var:.1f} < {dynamic_blur_threshold:.1f})")
                print("Tip: Hold the camera steady and ensure good lighting")
                return False
                
            # Check face size
            face_width = right - left
            face_height = bottom - top
            if min(face_width, face_height) < self.min_face_size:
                print("Face too small, please move closer")
                return False
                
            # Check brightness
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_face)
            if brightness < self.brightness_threshold:
                print("Image too dark, please improve lighting")
                return False
                
            # Check face angle using facial landmarks
            shape_predictor = self.recognizer.shape_predictor
            rect = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
            landmarks = shape_predictor(frame, rect)
            
            # Calculate rough head pose using eye positions
            left_eye = np.mean([(landmarks.part(36+i).x, landmarks.part(36+i).y) for i in range(6)], axis=0)
            right_eye = np.mean([(landmarks.part(42+i).x, landmarks.part(42+i).y) for i in range(6)], axis=0)
            
            eye_angle = np.degrees(np.arctan2(
                right_eye[1] - left_eye[1],
                right_eye[0] - left_eye[0]
            ))
            
            if abs(eye_angle) > self.max_angle_deviation:
                print("Please face the camera directly")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Error in face quality check: {e}")
            return False

    def _capture_face_samples(self, status_callback=None) -> List[np.ndarray]:
        """Capture multiple samples of a face with enhanced quality checks."""
        def update_status(msg):
            if status_callback:
                status_callback(msg)
            print(msg)
        
        if not self.camera:
            update_status("Error: Camera not initialized")
            return []
        
        face_encodings = []
        attempts = 0
        max_attempts = self.num_samples * 3
        
        try:
            while len(face_encodings) < self.num_samples and attempts < max_attempts:
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if len(face_locations) == 1:
                    if self._check_face_quality(frame, face_locations[0]):
                        encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                        
                        # Only check similarity if we have previous samples
                        if face_encodings:
                            similarities = [1 - np.linalg.norm(encoding - e) 
                                         for e in face_encodings]
                            if min(similarities) < self.quality_threshold:
                                update_status("Please maintain consistent face position")
                                attempts += 1
                                continue
                        
                        face_encodings.append(encoding)
                        update_status(f"Captured sample {len(face_encodings)}/{self.num_samples}")
                        
                        # Visual feedback
                        self._draw_feedback(frame, face_locations[0])
                    else:
                        attempts += 1
                else:
                    update_status("Please ensure exactly one face is visible")
                    attempts += 1
                
                time.sleep(self.capture_interval)
                
            if len(face_encodings) >= self.min_required_samples:
                update_status("Successfully captured face samples")
                return face_encodings
            else:
                update_status("Failed to capture enough quality samples")
                return []
            
        except Exception as e:
            update_status(f"Error during face capture: {e}")
            return []
            
        finally:
            cv2.destroyAllWindows()

    def _draw_feedback(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]):
        """Draw visual feedback during enrollment."""
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Draw guide overlay
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        target_size = self.min_face_size
        
        cv2.rectangle(
            frame,
            (center_x - target_size, center_y - target_size),
            (center_x + target_size, center_y + target_size),
            (255, 255, 255),
            1
        )


def main():
    """CLI interface for face enrollment."""
    enrollor = FaceEnrollment()
    
    while True:
        print("\nFace Enrollment System")
        print("1. Enroll new face")
        print("2. Exit")
        
        choice = input("\nSelect an option (1-2): ")
        
        if choice == '1':
            name = input("Enter name for the new face: ").strip()
            if not name:
                print("Name cannot be empty")
                continue
                
            enrollor.enroll_new_face(name)
            
        elif choice == '2':
            print("Exiting enrollment system")
            break
            
        else:
            print("Invalid choice. Please select 1 or 2.")


if __name__ == "__main__":
    main() 