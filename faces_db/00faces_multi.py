import os
from PIL import Image
import face_recognition

IMAGE_FOLDER = 'faces_db'
MULTIPLE_FACES_LOG = 'multiple_faces_detected.txt'

with open(MULTIPLE_FACES_LOG, 'w') as log:
    for filename in os.listdir(IMAGE_FOLDER):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(IMAGE_FOLDER, filename)
        try:
            image = face_recognition.load_image_file(path)
            face_locations = face_recognition.face_locations(image)

            if len(face_locations) > 1:
                log.write(f"{filename} has {len(face_locations)} faces\n")
                print(f"{filename}: {len(face_locations)} faces")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
