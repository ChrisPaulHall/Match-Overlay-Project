import cv2
import os
import numpy as np

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def has_face(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def review_images(folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"{filename}: Unable to read image.")
                continue
            blurry = is_blurry(image)
            face_found = has_face(image, face_cascade)
            if blurry or not face_found:
                reason = []
                if blurry:
                    reason.append("blurry")
                if not face_found:
                    reason.append("no face detected")
                print(f"{filename}: Unuseful ({', '.join(reason)})")
            else:
                print(f"{filename}: Useful")

if __name__ == "__main__":
    review_images(".")