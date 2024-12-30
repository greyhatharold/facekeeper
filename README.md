# Face Encoding Generator

A simple tool to capture and encode human faces into 128-dimensional numpy arrays for use in facial recognition applications.

## Overview

This program provides a user-friendly way to:

- Capture face images through a camera
- Process and validate face quality 
- Generate standardized 128-D face encodings
- Store encodings with associated names
- Preview and verify captured faces
- Export encodings for use in other applications
- Scare your friends who don't know how to use this program or what it does

The face encodings are generated using dlib's deep learning model and can be used for:

- Face verification/matching
- Identity management systems  
- Access control applications
- Custom facial recognition projects

## Features

- Live camera preview with face detection
- Quality checks for optimal face capture
- Multiple sample collection for accuracy
- Both GUI and CLI interfaces
- Configurable capture parameters
- Secure local storage of encodings
- Export options for encoded data

## Setup

1. Install required dependencies:

```bash
pip install -r requirements.txt
```

2. Download dlib models:

```bash
python download_models.py
```

3. Run the program:

```bash
python main.py
```

The program focuses on generating high-quality, consistent face encodings that can be reliably used for facial recognition tasks in other applications. Totally open source and free, let me know if you have any questions or suggestions.