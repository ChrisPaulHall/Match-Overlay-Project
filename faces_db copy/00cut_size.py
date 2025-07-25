import os
from PIL import Image
import face_recognition

INPUT_FOLDER = 'faces_db_copy'
OUTPUT_FOLDER = 'cropped_resized2'
DEEPFACE_SIZE = (224, 224)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        image = face_recognition.load_image_file(input_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            print(f"No face detected in {filename}")
            continue

        # Use the first detected face
        top, right, bottom, left = face_locations[0]
        face_image = image[top:bottom, left:right]

        pil_image = Image.fromarray(face_image)
        resized_image = pil_image.resize(DEEPFACE_SIZE)
        resized_image.save(output_path)

        print(f"Saved cropped and resized face: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
