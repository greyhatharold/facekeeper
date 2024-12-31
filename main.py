"""
Main entry point for the Raspberry Pi 4 face recognition system.
Provides both GUI and command-line interfaces with face detection,
recognition, and enrollment capabilities.
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from camera_capture import create_camera
from face_detection import create_detector
from face_recognition_module import create_recognizer
from face_enrollment import FaceEnrollment
from data_storage import FaceDataStorage
from gui import FaceRecognitionGUI
import tkinter as tk


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('face_recognition.log'),
            logging.StreamHandler()
        ]
    )

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logging.info("Shutdown signal received")
    sys.exit(0)

def run_cli_recognition(args) -> None:
    """
    Run face recognition in command-line mode.
    
    Args:
        args: Command line arguments
    """
    camera = create_camera()
    detector = create_detector()
    recognizer = create_recognizer()
    storage = FaceDataStorage()
    
    if not all([camera, detector, recognizer]):
        logging.error("Failed to initialize components")
        return
        
    try:
        frame_count = 0
        last_time = time.time()
        fps_interval = 30  # Calculate FPS every 30 frames
        
        logging.info("Starting face recognition loop...")
        while True:
            frame = camera.get_frame()
            if frame is None:
                continue
                
            frame_count += 1
            
            # Performance monitoring
            if frame_count % fps_interval == 0:
                current_time = time.time()
                fps = fps_interval / (current_time - last_time)
                last_time = current_time
                logging.info(f"FPS: {fps:.2f}")
            
            # Detect faces
            face_locations = detector.detect_faces(frame)
            
            if face_locations:
                # Generate encodings
                encodings = recognizer.encode_faces(frame, face_locations)
                
                # Match each face
                for encoding in encodings:
                    name, confidence = recognizer.match_face(encoding)
                    if name != "unknown":
                        logging.info(f"Detected: {name} (confidence: {1-confidence:.2f})")
                    
            # Optional delay for CPU management
            if args.delay > 0:
                time.sleep(args.delay)
                
    except KeyboardInterrupt:
        logging.info("Recognition stopped by user")
    finally:
        camera.release()

def run_enrollment(args) -> None:
    """
    Run face enrollment process.
    
    Args:
        args: Command line arguments containing enrollment name
    """
    enrollment = FaceEnrollment()
    if enrollment.enroll_new_face(args.enroll):
        logging.info(f"Successfully enrolled {args.enroll}")
    else:
        logging.error("Enrollment failed")

def run_gui() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

def main():
    """Main entry point for the application."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--cli", action="store_true",
                       help="Run in command-line mode instead of GUI")
    parser.add_argument("--enroll", type=str, metavar="NAME",
                       help="Enroll a new face with the given name")
    parser.add_argument("--delay", type=float, default=0.0,
                       help="Delay between frames in seconds (default: 0.0)")
    
    args = parser.parse_args()
    
    # Set up logging and signal handling
    setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.enroll:
            run_enrollment(args)
        elif args.cli:
            run_cli_recognition(args)
        else:
            run_gui()
            
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 