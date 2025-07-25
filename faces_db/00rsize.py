import os
import cv2

# Configuration
input_folder = "faces_db"
output_folder = "faces_db_resized"
target_size = (224, 224)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    image = cv2.imread(input_path)
    if image is None:
        print(f"⚠️ Could not read image: {filename}")
        continue

    resized = cv2.resize(image, target_size)
    cv2.imwrite(output_path, resized)
    print(f"✅ Resized and saved: {filename}")
