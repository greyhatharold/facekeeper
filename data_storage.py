"""
Data storage module for face recognition system.
Handles loading and saving of face encodings and associated names.
Optimized for Raspberry Pi 4 with efficient disk operations.
"""

import os
import yaml
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging


class FaceDataStorage:
    def __init__(self):
        """Initialize the data storage with configuration settings."""
        self.config = self._load_config()
        self.data_path = Path(self.config.get('data_path', 'face_data.yaml'))
        # Cache for storing face data in memory
        self._cached_data: Optional[Dict] = None
        
    def _load_config(self) -> dict:
        """Load configuration from config.yaml file."""
        try:
            config_path = Path('config.yaml')
            with open(config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def load_known_faces(self):
        """Load known faces from YAML file."""
        try:
            with open(self.data_path, 'r') as file:
                data = yaml.safe_load(file)
                if not data or 'faces' not in data:
                    return [], []
                
                names = []
                encodings = []
                for face in data['faces']:
                    if 'name' in face and 'encoding' in face:
                        names.append(face['name'])
                        encodings.append(face['encoding'])
                
                return names, encodings
        except Exception as e:
            logging.error(f"Error loading face data: {e}")
            return [], []

    def save_known_faces(self, names: List[str], encodings: List[np.ndarray]) -> bool:
        """
        Save face data to storage.
        
        Args:
            names: List of person names
            encodings: List of face encodings (128-d numpy arrays)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for YAML serialization
            data = {
                'faces': [
                    {
                        'name': name,
                        'encoding': encoding.tolist()
                    }
                    for name, encoding in zip(names, encodings)
                ]
            }
            
            # Create directory if it doesn't exist
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temporary file first
            temp_path = self.data_path.with_suffix('.tmp')
            with open(temp_path, 'w') as file:
                yaml.dump(data, file, default_flow_style=False)
            
            # Atomic rename for data integrity
            temp_path.replace(self.data_path)
            
            # Update cache
            self._cached_data = data
            return True
            
        except Exception as e:
            print(f"Error saving face data: {e}")
            return False

    def _convert_to_lists(self, data: Dict) -> Tuple[List[str], List[np.ndarray]]:
        """Convert stored data format to parallel lists of names and encodings."""
        if not data or 'faces' not in data:
            return [], []
            
        names = []
        encodings = []
        
        for face in data['faces']:
            names.append(face['name'])
            encodings.append(np.array(face['encoding']))
            
        return names, encodings

    def clear_cache(self):
        """Clear the in-memory cache of face data."""
        self._cached_data = None 