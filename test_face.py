# test_face.py
import face_recognition
import face_recognition_models
import pkg_resources

def test_installation():
    print("Face Recognition Version:", face_recognition.__version__)
    
    # Check model files
    try:
        model_path = pkg_resources.resource_filename('face_recognition_models', 
            'models/shape_predictor_68_face_landmarks.dat')
        print("Model path exists:", model_path)
        
        # Try to load a test image
        image = face_recognition.load_image_file("test.jpg")  # You'll need any test image
        face_locations = face_recognition.face_locations(image)
        print("Successfully processed test image")
        
    except Exception as e:
        print("Error:", str(e))
        print("Model path:", pkg_resources.resource_filename('face_recognition_models', 'models'))

if __name__ == "__main__":
    test_installation()
