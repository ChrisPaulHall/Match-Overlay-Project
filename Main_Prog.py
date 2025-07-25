import cv2
import face_recognition
import numpy as np
import os
import csv
from flask import Flask, jsonify

app = Flask(__name__)

def quick_scrape_summary_info(url):
    pass


# Configurable label for unknown faces (for internationalization)
UNKNOWN_LABEL = "Unknown"

# Load known faces
KNOWN_DIR = 'cropped_resized2'
known_encodings = []
known_names = []
known_urls = []

for filename in os.listdir(KNOWN_DIR):
    if filename.endswith('.jpg'):
        image = face_recognition.load_image_file(os.path.join(KNOWN_DIR, filename))
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].strip().lower()
            known_names.append(name)
            known_urls.append(None)  # Placeholder, to be updated with your mapping

# Load slug_url mapping from CSV
slug_map = {}
with open("opensecrets1.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        slug_key = row['first_name'].strip().lower() + row['last_name'].strip().lower()
        slug_map[slug_key] = row['slug_url']

# Real-time webcam feed
video_capture = cv2.VideoCapture(1)

# Cache for summary info to avoid repeated scraping
summary_cache = {}

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        continue
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            slug_url = slug_map.get(name, '')
            if slug_url:
                if slug_url not in summary_cache:
                    summary_cache[slug_url] = quick_scrape_summary_info(slug_url)
                info = summary_cache[slug_url]
            else:
                info = {"name": name, "summary": {}}

            # Label and draw box
            y = top - 10 if top - 10 > 10 else top + 10
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            # Calculate height of one line of text
            (text_width, text_height), baseline = cv2.getTextSize("A", font, font_scale, thickness)
            # Build label string
            label = name
            if info and "summary" in info and isinstance(info["summary"], dict):
                for k, v in info["summary"].items():
                    label += f"\n{k}: {v}"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            for i, line in enumerate(label.splitlines()):
                cv2.putText(frame, line, (left, y + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow('Live Match with Overlay', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

@app.route("/data.json")
def data():
    if os.path.exists("overlay_data.json"):
        with open("overlay_data.json", "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"name": "No speaker", "top_donors": []})

video_capture.release()
cv2.destroyAllWindows()
