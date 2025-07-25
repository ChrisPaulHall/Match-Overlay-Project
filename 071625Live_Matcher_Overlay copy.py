import cv2
import os
import json
from datetime import datetime

# Suppress DeepFace + TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('deepface').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

from deepface import DeepFace
import numpy as np
import csv
import face_recognition

import argparse

#CONFIGURATION
parser = argparse.ArgumentParser(description="Live Face Matcher Overlay")
parser.add_argument("--cam_index", type=int, default=1, help="Camera index for cv2.VideoCapture")
args = parser.parse_args()

cam_index = args.cam_index
frame_interval = 10
face_match_threshold = 0.6
faces_db_path = "cropped_resized3"
overlay_path = "overlay_data.json"
temp_img_path = "temp_frame.jpg"
frames_dir = "matched_frames"
os.makedirs(frames_dir, exist_ok=True)
display_live_feed = False  

name_map = {}
name_map_csv_path = "name_map.csv"
if os.path.exists(name_map_csv_path):
    with open(name_map_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            
            filename = row.get('filename')
            if not filename:
                continue  # Skip malformed rows
            name_map[filename.strip()] = {
                "first_name": str(row["first_name"] or "").strip(),
                "last_name": str(row["last_name"] or "").strip(),
                "title": str(row["title"] or "").strip(),
                "state": str(row["state"] or "").strip(),
                "party": str(row["party"] or "").strip(),
                "terms": str(row["terms"] or "").strip()
            }
else:
    print(f"Warning: {name_map_csv_path} not found. name_map will be empty.")

# Load donor data from merged CSV (once at top of script)
donor_lookup = {}
with open("opensecrets_merged.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        first = row['first_name'].strip().lower()
        last = row['last_name'].strip().lower()
        key = f"{first} {last}"
        donor_lookup[key] = row  # Store the entire row as a dict


# Init video capture
video = cv2.VideoCapture(cam_index)
frame_count = 0
last_match = None
match_cooldown = 90
last_match_frame = -match_cooldown
last_overlay_data = None

while True:
    success, frame = video.read()
    if not success:
        print("Failed to read from the camera source.")
        break

    if frame_count % frame_interval == 0:
        try:
            cv2.imwrite(temp_img_path, frame)
            face_locations = face_recognition.face_locations(frame)
            if not face_locations:
                continue

            result = DeepFace.find(img_path=temp_img_path, db_path=faces_db_path, enforce_detection=False, refresh_database=True, threshold=face_match_threshold)
            if hasattr(result[0], "iloc") and not result[0].empty:
                best = result[0].iloc[0]
                match = best['identity']
                
                if match != last_match or (frame_count - last_match_frame) > match_cooldown:
                    last_match = match
                    last_match_frame = frame_count
                    
                    info = name_map.get(os.path.basename(match), {})
                    title = info.get("title", "")
                    first = info.get("first_name", "")
                    last = info.get("last_name", "")
                    state = info.get("state", "")
                    party = info.get("party", "")
                    speaker_name = f"{first} {last} {title} ({party}-{state})"

                    first = info.get("first_name", "").strip().lower()
                    last = info.get("last_name", "").strip().lower()
                    lookup_key = f"{first} {last}"

                    donor_row = donor_lookup.get(lookup_key, {})
                    top_donors_str = donor_row.get('top_contributors', '') if isinstance(donor_row, dict) else ''
                    donors = []
                    for entry in top_donors_str.split(';')[:3]:
                        if '(' in entry and ')' in entry:
                            name, amount = entry.rsplit('(', 1)
                            donors.append({
                                "name": name.strip(),
                                "amount": f"${amount.strip(')')}"
                            })
                    industry_raw = donor_row.get('top_industries', '') if isinstance(donor_row, dict) else ''
                    industries = []
                    for entry in industry_raw.split(';')[:3]:
                        if '(' in entry and ')' in entry:
                            name, amount = entry.rsplit('(', 1)
                            industries.append({
                                "name": name.strip(),
                                "amount": f"${amount.strip(')')}"
                            })

                    overlay_data = {
                        "name": speaker_name,
                        "image": os.path.basename(match), 
                        "top_donors": donors,
                        "top_industries": industries,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Only write & print if the overlay data truly changed
                    if overlay_data != last_overlay_data:
                        print(json.dumps(overlay_data, indent=2))
                        with open(overlay_path, "w") as f:
                            json.dump(overlay_data, f, indent=2)
                        last_overlay_data = overlay_data
                        
                        print(f"Match found: {os.path.basename(match)} at frame {frame_count} (distance: {best['distance']:.2f}) Overlay updated.")
                        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, frame)
                    else:
                        logging.debug(f"Repeat match. Overlay data unchanged at frame {frame_count}.")
                else:
                    print(f"Repeat match: {match} at frame {frame_count} — skipping overlay update.")
            else:
                print(f"No reliable match at frame {frame_count}.")        
        except Exception as e:
            print(f"Error at frame {frame_count}: {e}")
            
    frame_count += 1

    # Optional: display live feed
    if display_live_feed:
        cv2.imshow("OBS Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()