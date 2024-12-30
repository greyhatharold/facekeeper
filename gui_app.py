"""
GUI application for face recognition system.
Provides a unified interface for face detection, recognition, and enrollment.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
from typing import Optional
import logging
import face_recognition
import numpy as np
import math

from camera_capture import create_camera
from face_detection import create_detector
from face_recognition_module import create_recognizer
from face_enrollment import FaceEnrollment
from data_storage import FaceDataStorage


class FaceRecognitionGUI:
    def __init__(self, root: tk.Tk):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.camera = create_camera()
        self.detector = create_detector()
        self.recognizer = create_recognizer()
        self.storage = FaceDataStorage()
        self.enrollment = FaceEnrollment()
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.processing_thread = None
        
        self._setup_gui()
        self._load_known_faces()
        
    def _setup_gui(self):
        """Set up the GUI layout and components."""
        # Configure style
        style = ttk.Style()
        style.configure('TButton', padding=5, font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10))
        style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 11, 'bold'))
        
        # Main container with improved padding
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel with better spacing
        left_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=2)
        
        # Title for video feed
        ttk.Label(left_panel, text="Live Camera Feed", style='Title.TLabel').pack(pady=(0, 5))
        
        # Video feed with border
        video_frame = ttk.Frame(left_panel, relief='solid', borderwidth=1)
        video_frame.pack(padx=10, pady=5)
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(padx=2, pady=2)
        
        # Control buttons in a better layout
        controls = ttk.Frame(left_panel)
        controls.pack(fill=tk.X, padx=10, pady=10)
        
        # Buttons with consistent spacing and improved style
        ttk.Button(controls, text="▶ Start", command=self.start_recognition, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="⏹ Stop", command=self.stop_recognition, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls, text="➕ Enroll New Face", command=self.show_enrollment_dialog, width=15).pack(side=tk.LEFT, padx=5)
        
        # Right panel with better organization
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=1)
        
        # Recognition results section
        ttk.Label(right_panel, text="Recognition Results", style='Header.TLabel').pack(pady=5)
        results_frame = ttk.Frame(right_panel, relief='solid', borderwidth=1)
        results_frame.pack(padx=10, pady=(0, 10), fill=tk.BOTH)
        
        self.results_text = tk.Text(results_frame, height=10, width=40, font=('Helvetica', 10),
                                   wrap=tk.WORD, padx=5, pady=5)
        self.results_text.pack(padx=2, pady=2, fill=tk.BOTH)
        
        # Known faces section
        ttk.Label(right_panel, text="Known Faces", style='Header.TLabel').pack(pady=5)
        faces_frame = ttk.Frame(right_panel, relief='solid', borderwidth=1)
        faces_frame.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)
        
        self.known_faces_listbox = tk.Listbox(faces_frame, font=('Helvetica', 10),
                                             selectmode=tk.SINGLE, activestyle='dotbox')
        self.known_faces_listbox.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)
        
    def _load_known_faces(self):
        """Load and display known faces in the listbox."""
        names, _ = self.storage.load_known_faces()
        self.known_faces_listbox.delete(0, tk.END)
        for name in names:
            self.known_faces_listbox.insert(tk.END, name)
            
    def start_recognition(self):
        """Start the face recognition process."""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            self.update_gui()
            
    def stop_recognition(self):
        """Stop the face recognition process."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def _process_frames(self):
        """Process frames from camera in a separate thread."""
        while self.running:
            if self.camera:
                frame = self.camera.get_frame()
                if frame is not None:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                    
                    # Generate encodings for detected faces
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    frame_with_boxes = frame.copy()
                    results = []
                    
                    # Load known faces for comparison
                    known_names, known_encodings = self.storage.load_known_faces()
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Compare with known faces
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        
                        name = "Unknown"
                        confidence = 0.0
                        
                        if len(face_distances) > 0:
                            best_match_index = face_distances.argmin()
                            if matches[best_match_index]:
                                name = known_names[best_match_index]
                                confidence = 1 - face_distances[best_match_index]
                        
                        # Draw box with gradient color based on confidence
                        if name != "Unknown":
                            # Green gradient based on confidence
                            color = (0, int(255 * confidence), 0)
                        else:
                            # Yellow for unknown faces
                            color = (0, 255, 255)
                        
                        # Draw box with thicker lines
                        cv2.rectangle(frame_with_boxes, (left, top), (right, bottom), 
                                    color, 3)
                        
                        # Add background for text for better visibility
                        label = f"{name} ({confidence:.2f})"
                        font = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.8
                        thickness = 2
                        
                        # Calculate text size and position
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                        text_width = text_size[0]
                        text_height = text_size[1]
                        
                        # Position text above face box with padding
                        text_x = left
                        text_y = top - 15 if top > 30 else bottom + 30
                        
                        # Draw background rectangle for text with rounded corners
                        padding = 5
                        box_coords = ((text_x - padding, text_y - text_height - padding),
                                    (text_x + text_width + padding, text_y + padding))
                        
                        # Draw semi-transparent background
                        overlay = frame_with_boxes.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color, cv2.FILLED)
                        cv2.addWeighted(overlay, 0.7, frame_with_boxes, 0.3, 0, frame_with_boxes)
                        
                        # Draw text with shadow for better visibility
                        cv2.putText(frame_with_boxes, label,
                                  (text_x, text_y),
                                  font, font_scale, (0, 0, 0), thickness + 1)  # shadow
                        cv2.putText(frame_with_boxes, label,
                                  (text_x, text_y),
                                  font, font_scale, (255, 255, 255), thickness)  # main text
                        
                        if name != "Unknown":
                            results.append(f"{name} (Confidence: {confidence:.2f})")
                    
                    # Update results text
                    if results:
                        self.results_text.delete(1.0, tk.END)
                        self.results_text.insert(tk.END, "\n".join(results))
                    
                    try:
                        self.frame_queue.put_nowait(frame_with_boxes)
                    except queue.Full:
                        continue
                        
    def update_gui(self):
        """Update the GUI with the latest frame."""
        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.video_label.image = photo
            except queue.Empty:
                pass
            self.root.after(30, self.update_gui)
            
    def show_enrollment_dialog(self):
        """Show dialog for enrolling a new face."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Face Enrollment")
        dialog.geometry("1000x800")
        
        # Add flag to track if dialog is still open
        dialog.is_open = True
        
        # Add handler for dialog closing
        def on_dialog_close():
            dialog.is_open = False
            dialog.destroy()
        
        dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Enrollment.TLabelframe', padding=10)
        style.configure('Enrollment.TLabel', font=('Helvetica', 10))
        style.configure('EnrollmentTitle.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Guide.TLabel', font=('Helvetica', 10), padding=5)
        
        # Main container with improved spacing
        container = ttk.Frame(dialog, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video and guide
        left_frame = ttk.Frame(container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        ttk.Label(left_frame, text="Face Enrollment Camera", style='EnrollmentTitle.TLabel').pack(pady=(0, 5))
        
        # Video preview with border
        preview_frame = ttk.Frame(left_frame, relief='solid', borderwidth=1, width=800, height=600)
        preview_frame.pack_propagate(False)
        preview_frame.pack(pady=5)
        
        preview_label = ttk.Label(preview_frame)
        preview_label.pack(expand=True)
        
        # Guide text with better formatting
        guide_text = ttk.Label(left_frame, 
                              text="📋 Enrollment Guidelines:\n\n" +
                                   "• Position your face within the white box\n" +
                                   "• Keep your face straight and well-lit\n" +
                                   "• Stay still during capture",
                              style='Guide.TLabel',
                              justify=tk.LEFT)
        guide_text.pack(pady=10)
        
        # Right side controls with consistent styling
        right_frame = ttk.Frame(container, padding=(10, 0, 0, 0))
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        
        # Name input section
        input_frame = ttk.LabelFrame(right_frame, text="Enter Name", style='Enrollment.TLabelframe')
        input_frame.pack(fill=tk.X, pady=5)
        
        name_var = tk.StringVar()
        name_entry = ttk.Entry(input_frame, textvariable=name_var, font=('Helvetica', 10))
        name_entry.pack(padx=5, pady=5, fill=tk.X)
        
        # Status display
        status_frame = ttk.LabelFrame(right_frame, text="Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        status_var = tk.StringVar(value="Ready to enroll...")
        status_label = ttk.Label(status_frame, textvariable=status_var, 
                                wraplength=200, justify=tk.LEFT)
        status_label.pack(padx=5, pady=5, fill=tk.X)
        
        # Quality indicators
        quality_frame = ttk.LabelFrame(right_frame, text="Quality Checks")
        quality_frame.pack(fill=tk.X, pady=5)
        
        # Quality indicator variables
        face_detected_var = tk.StringVar(value="❌ No Face Detected")
        face_size_var = tk.StringVar(value="❌ Face Size")
        face_angle_var = tk.StringVar(value="❌ Face Angle")
        brightness_var = tk.StringVar(value="❌ Brightness")
        blur_var = tk.StringVar(value="❌ Sharpness")
        
        # Quality indicator labels
        ttk.Label(quality_frame, textvariable=face_detected_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(quality_frame, textvariable=face_size_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(quality_frame, textvariable=face_angle_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(quality_frame, textvariable=brightness_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(quality_frame, textvariable=blur_var).pack(anchor=tk.W, padx=5, pady=2)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(right_frame, text="Enrollment Progress")
        progress_frame.pack(fill=tk.X, pady=5)
        
        progress_var = tk.IntVar(value=0)
        progress = ttk.Progressbar(progress_frame, variable=progress_var,
                                 maximum=self.enrollment.num_samples)
        progress.pack(padx=5, pady=5, fill=tk.X)
        
        def update_quality_indicators(frame):
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    face_detected_var.set("✅ Face Detected")
                    face_location = face_locations[0]
                    
                    # Update quality indicators based on checks
                    if self.enrollment._check_face_quality(frame, face_location):
                        face_size_var.set("✅ Face Size")
                        face_angle_var.set("✅ Face Angle")
                        brightness_var.set("✅ Brightness")
                        blur_var.set("✅ Sharpness")
                        return True
                else:
                    face_detected_var.set("❌ No Face Detected")
                    face_size_var.set("❌ Face Size")
                    face_angle_var.set("❌ Face Angle")
                    brightness_var.set("❌ Brightness")
                    blur_var.set("❌ Sharpness")
            
            return False
        
        def update_preview():
            if hasattr(dialog, 'enrolling'):
                frame = self.camera.get_frame()
                if frame is not None:
                    # Update quality indicators
                    quality_ok = update_quality_indicators(frame)
                    
                    # Draw guide overlay
                    height, width = frame.shape[:2]
                    center_x, center_y = width // 2, height // 2
                    target_size = self.enrollment.min_face_size
                    
                    overlay = frame.copy()
                    
                    # Draw semi-transparent overlay outside target area
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, 
                                 (center_x - target_size, center_y - target_size),
                                 (center_x + target_size, center_y + target_size),
                                 255, cv2.FILLED)
                    frame = cv2.addWeighted(frame, 0.7, 
                                          cv2.cvtColor(cv2.bitwise_not(mask), 
                                             cv2.COLOR_GRAY2BGR), 0.3, 0)
                    
                    # Draw target box with animated corners
                    t = time.time() * 2  # Animation speed
                    corner_length = 20
                    alpha = abs(math.sin(t)) * 0.5 + 0.5  # Pulsing effect
                    
                    color = (0, 255, 0) if quality_ok else (0, 165, 255)  # Green if OK, orange if not
                    thickness = 2
                    
                    # Draw corners
                    for x, y in [(center_x - target_size, center_y - target_size),  # Top-left
                                (center_x + target_size, center_y - target_size),  # Top-right
                                (center_x - target_size, center_y + target_size),  # Bottom-left
                                (center_x + target_size, center_y + target_size)]: # Bottom-right
                        
                        # Horizontal lines
                        cv2.line(frame, (x - corner_length, y), (x + corner_length, y),
                                color, thickness)
                        # Vertical lines
                        cv2.line(frame, (x, y - corner_length), (x, y + corner_length),
                                color, thickness)
                    
                    # Draw face detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    
                    for (top, right, bottom, left) in face_locations:
                        # Draw box with dashed lines
                        dash_length = 10
                        for i in range(left, right, dash_length * 2):
                            end = min(i + dash_length, right)
                            cv2.line(frame, (i, top), (end, top), color, 2)
                            cv2.line(frame, (i, bottom), (end, bottom), color, 2)
                        for i in range(top, bottom, dash_length * 2):
                            end = min(i + dash_length, bottom)
                            cv2.line(frame, (left, i), (left, end), color, 2)
                            cv2.line(frame, (right, i), (right, end), color, 2)
                    
                    # Convert and display
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    photo = ImageTk.PhotoImage(image=image)
                    preview_label.configure(image=photo)
                    preview_label.image = photo
                
                if dialog.enrolling:
                    dialog.after(30, update_preview)
        
        def start_enrollment():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name")
                return
            
            dialog.enrolling = True
            start_button.state(['disabled'])
            name_entry.state(['disabled'])
            progress_var.set(0)
            update_preview()
            
            def enrollment_process():
                try:
                    def status_callback(msg):
                        if dialog.is_open:  # Only update if dialog still exists
                            status_var.set(msg)
                            if "Captured sample" in msg:
                                sample_num = int(msg.split('/')[0].split()[-1])
                                progress_var.set(sample_num)
                    
                    if self.enrollment.enroll_new_face(name, status_callback=status_callback):
                        if dialog.is_open:  # Check if dialog still exists
                            messagebox.showinfo("Success", f"Successfully enrolled {name}")
                            self._load_known_faces()
                            dialog.destroy()
                    else:
                        if dialog.is_open:  # Check if dialog still exists
                            messagebox.showerror("Error", "Enrollment failed")
                            start_button.state(['!disabled'])
                            name_entry.state(['!disabled'])
                finally:
                    if dialog.is_open:  # Only cleanup if dialog still exists
                        dialog.enrolling = False
            
            threading.Thread(target=enrollment_process, daemon=True).start()
        
        start_button = ttk.Button(dialog, text="Start Enrollment", command=start_enrollment)
        start_button.pack(pady=10)
        
    def __del__(self):
        """Clean up resources."""
        self.stop_recognition()
        if self.camera:
            self.camera.release()


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 