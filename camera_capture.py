"""
Camera capture module for video stream handling.
Provides functionality to capture frames from various camera sources
with configuration management and error handling.
"""

import cv2
import yaml
import logging
from typing import Optional, Tuple, Union
import numpy as np

class CameraCapture:
    """Handles camera initialization and frame capture with configurable parameters."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize camera capture with configuration from YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.camera = None
        self.config = self._load_config(config_path)
        self.initialize_camera()
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load camera configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration parameters
            
        Raises:
            FileNotFoundError: If config file is not found
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config['camera']
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise
            
    def initialize_camera(self):
        """
        Initialize camera device with configured parameters.
        
        Raises:
            RuntimeError: If camera initialization fails
        """
        try:
            self.camera = cv2.VideoCapture(self.config.get('device_id', 0))
            
            # Reset any existing camera settings
            self.camera.set(cv2.CAP_PROP_ZOOM, 0)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.camera.set(cv2.CAP_PROP_FPS, self.config.get('fps', 30))
            
            # Apply zoom factor if specified
            zoom = self.config.get('zoom_factor', 1.0)
            if zoom != 1.0:
                self.camera.set(cv2.CAP_PROP_ZOOM, zoom)
            
            if not self.camera.isOpened():
                raise RuntimeError("Failed to open camera")
                
            # Verify settings were applied
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logging.info(f"Camera initialized with resolution: {actual_width}x{actual_height}")
            
        except Exception as e:
            logging.error(f"Camera initialization failed: {e}")
            raise RuntimeError(f"Camera initialization failed: {e}")
            
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture and return the latest frame from the camera.
        
        Returns:
            Optional[np.ndarray]: Captured frame as NumPy array, or None if capture fails
        """
        if self.camera is None:
            logging.error("Camera not initialized")
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                logging.warning("Failed to capture frame")
                return None
                
            # Apply center offset if specified
            offset_x = self.config.get('center_offset_x', 0)
            offset_y = self.config.get('center_offset_y', 0)
            if offset_x != 0 or offset_y != 0:
                height, width = frame.shape[:2]
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                frame = cv2.warpAffine(frame, M, (width, height))
                
            return frame
            
        except Exception as e:
            logging.error(f"Error capturing frame: {e}")
            return None
            
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Get the current frame dimensions.
        
        Returns:
            Tuple[int, int]: Width and height of the frame
        """
        return (
            int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        
    def __del__(self):
        """Ensure camera resources are properly released."""
        self.release()
        
    def release(self):
        """Release camera resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

def create_camera() -> Optional[CameraCapture]:
    """
    Factory function to create and initialize a CameraCapture instance.
    
    Returns:
        Optional[CameraCapture]: Initialized camera capture object or None if initialization fails
    """
    try:
        return CameraCapture()
    except Exception as e:
        logging.error(f"Failed to create camera capture: {e}")
        return None