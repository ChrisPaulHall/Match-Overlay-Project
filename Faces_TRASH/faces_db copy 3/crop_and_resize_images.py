import os
import cv2

faces_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(faces_dir, "cropped_resized")
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for fname in image_files:
    path = os.path.join(faces_dir, fname)
    img = cv2.imread(path)
    if img is None:
        print(f"Unreadable: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        print(f"No face detected: {fname}")
        continue
    # Use the largest detected face
    x, y, w, h = max(faces, key=lambda rect: rect[2]*rect[3])
    face_img = img[y:y+h, x:x+w]
    resized = cv2.resize(face_img, (224, 224))
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, resized)
    print(f"Cropped and resized: {fname} -> {out_path}")