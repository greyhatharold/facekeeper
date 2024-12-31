import os
import requests
from tqdm import tqdm

def download_dlib_models():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Dictionary of model files and their URLs
    models = {
        'shape_predictor_68_face_landmarks.dat': 
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'dlib_face_recognition_resnet_model_v1.dat':
            'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
        'mmod_human_face_detector.dat':
            'http://dlib.net/files/mmod_human_face_detector.dat.bz2'
    }
    
    for model_name, url in models.items():
        output_path = os.path.join('models', model_name)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"{model_name} already exists, skipping...")
            continue
            
        print(f"Downloading {model_name}...")
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path + '.bz2', 'wb') as file, tqdm(
            desc=model_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        # Decompress the .bz2 file
        print(f"Decompressing {model_name}...")
        os.system(f"bzip2 -d {output_path}.bz2")
        
        print(f"Successfully downloaded and decompressed {model_name}")

if __name__ == "__main__":
    download_dlib_models()