import os
import cv2

faces_dir = os.path.dirname(os.path.abspath(__file__))
image_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def analyze_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return {"status": "Unreadable"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Check image size
    h, w = img.shape[:2]
    # Check blurriness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return {
        "size": f"{w}x{h}",
        "blur": lap_var,
        "faces_detected": len(faces)
    }

for fname in image_files:
    path = os.path.join(faces_dir, fname)
    result = analyze_image(path)
    print(f"{fname}: {result}")