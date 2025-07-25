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
import pytesseract
import re

import argparse
import difflib
from difflib import SequenceMatcher

#CONFIGURATION
parser = argparse.ArgumentParser(description="Live Face Matcher Overlay")
parser.add_argument("--cam_index", type=int, default=1, help="Camera index for cv2.VideoCapture")
args = parser.parse_args()

cam_index = args.cam_index
frame_interval = 45

#Thresholds
face_match_threshold = 0.65
high_face_threshold    = 0.35

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

# Load donor data from merged CSV 
donor_lookup = {}
with open("opensecrets_merged.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        first = row['first_name'].strip().lower()
        last = row['last_name'].strip().lower()
        key = f"{first} {last}"
        donor_lookup[key] = row  

# Load 2018 net‑worth data from merged CSV
networth_lookup = {}
with open("current_with_networth.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = f"{row['first_name'].strip().lower()} {row['last_name'].strip().lower()}"
        networth_lookup[key] = row.get('networth_2018', 'N/A')


# Party abbreviation mapping
party_map = {
    "democratic": "D",
    "democrat": "D",
    "republican": "R",
    "gop": "R"
}

# State abbreviation mapping
state_abbr = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY"
}

known_last_names = [info['last_name'].lower() for info in name_map.values()]
name_word_re = re.compile(r"[A-Za-z']{3,}")

# Init video capture
video = cv2.VideoCapture(cam_index)
frame_count = 0
last_match = None
match_cooldown = 45
last_match_frame = -match_cooldown
last_overlay_data = None

while True:
    success, frame = video.read()
    if not success:
        print("Failed to read from the camera source.")
        break

    if frame_count % frame_interval == 0:
        try:
            # 1) OCR on bottom half
            h, w = frame.shape[:2]
            bottom = frame[h//2:h, :]
            gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(thresh)

            # 1a) Last‑name matching only

            #words = re.findall(r"[A-Za-z'-]{3,}", text)
            
            words = name_word_re.findall(text)    # now using the precompiled regex
            ocr_lastname_candidate = None
            matched_word = ""
            hit = ""
            for word in words:
                matches = difflib.get_close_matches(word.lower(), known_last_names, n=1, cutoff=0.8)
                if matches:
                    hit = matches[0]            # the normalized last name
                    matched_word = word         # the raw OCR text
                    
                    # reverse-lookup filename for the matched last name
                    for filename, info in name_map.items():
                        if info['last_name'].lower() == hit:
                            ocr_lastname_candidate = filename
                            logging.info("🔍 OCR last‑name match: %s → %s", hit, filename)
                            break
                    break

                if ocr_lastname_candidate:
                    logging.info("🔍 OCR last‑name match: %s → %s", hit, ocr_lastname_candidate)
                    break

            # 2) Face recognition 
            cv2.imwrite(temp_img_path, frame)
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                df_results = DeepFace.find(
                    img_path=temp_img_path,
                    db_path=faces_db_path,
                    enforce_detection=False,
                    refresh_database=False,
                    threshold=face_match_threshold
                )
                if hasattr(df_results[0], "iloc") and not df_results[0].empty:
                    best_row   = df_results[0].iloc[0]
                    face_match = best_row['identity']
                    dist       = best_row['distance']

                    ocr_weight = 0.6  # tune between 0.0 (all face) and 1.0 (all OCR)
                    face_conf = max(0.0, (face_match_threshold - dist) / face_match_threshold)
                    ocr_conf = 0.0
                    if ocr_lastname_candidate:
                        ocr_conf = SequenceMatcher(None,
                                                matched_word.lower(),
                                                hit.lower()
                                                ).ratio()

                    # weighted “scores”
                    score_face = face_conf * (1 - ocr_weight)
                    score_ocr  = ocr_conf   * ocr_weight

                    # choose whichever wins
                    if ocr_lastname_candidate and score_ocr > score_face:
                        final_match = ocr_lastname_candidate
                        logging.info("Chose OCR match %s (ocr_conf=%.2f > face_conf=%.2f)",
                                    ocr_lastname_candidate, ocr_conf, face_conf)
                    else:
                        final_match = face_match
                        logging.info("Chose face match %s (face_conf=%.2f ≥ ocr_conf=%.2f)",
                                    face_match, face_conf, ocr_conf)

                    # 3) Disambiguate: OCR‑lastname + moderate face → OCR wins
                    #if ocr_lastname_candidate and high_face_threshold < dist <= face_match_threshold:
                        #final_match = ocr_lastname_candidate
                        #logging.info("Using OCR last‑name match: %s", ocr_lastname_candidate)
                    #else:
                        #final_match = face_match
                        #logging.info("Using face match: %s (distance: %.2f)", face_match, dist)

                    # only update overlay if new
                    if final_match != last_match or (frame_count - last_match_frame) > match_cooldown:
                        last_match       = final_match
                        last_match_frame = frame_count

                        filename = os.path.basename(final_match)
                        info = name_map.get(filename)
                        if not info:
                            logging.warning("No metadata for %s; skipping", filename)
                            continue

                        # consolidate name fields
                        first = info.get('first_name', '').strip()
                        last  = info.get('last_name', '').strip()
                        lookup_key = f"{first.lower()} {last.lower()}"
                    
                        # party & state abbreviations
                        party_full = info.get('party','').strip().lower()
                        party_abbr = party_map.get(party_full, party_full[:1].upper() if party_full else "")
                        state_full = info.get('state','').strip()
                        state_abbrv = state_abbr.get(state_full, state_full)
                    
                        # build the speaker name
                        speaker_name = (
                            f"{first} {last}\n"
                            f"{info.get('title','')} {party_abbr}-{state_abbrv}"
                        ).strip()
                    
                        # use lookup_key for donors and networth
                        donor_row = donor_lookup.get(lookup_key, {})
                        raw_networth = networth_lookup.get(lookup_key, 'N/A')

                        # parse top 3 contributors
                        top_donors_str = donor_row.get('top_contributors', '')
                        donors = []
                        for entry in top_donors_str.split(';')[:5]:
                            if '(' in entry and ')' in entry:
                                name, amt = entry.rsplit('(', 1)
                                donors.append({
                                    "name":   name.strip(),
                                    "amount": f"${amt.rstrip(')')}"
                                })

                        # parse top 3 industries
                        industry_raw = donor_row.get('top_industries', '')
                        industries = []
                        for entry in industry_raw.split(';')[:5]:
                            if '(' in entry and ')' in entry:
                                name, amt = entry.rsplit('(', 1)
                                industries.append({
                                    "name":   name.strip(),
                                    "amount": f"${amt.rstrip(')')}"
                                })

                        lookup_key = f"{info['first_name'].lower()} {info['last_name'].lower()}"
                        raw_networth = networth_lookup.get(lookup_key, None)
                        if raw_networth and raw_networth not in ("", "N/A"):
                            try:
                                val = float(raw_networth)
                                networth = f"${val:,.0f}"
                            except ValueError:
                                networth = raw_networth
                        else:
                            networth = "N/A"

                        overlay_data = {
                            "name":          speaker_name,
                            "image":         os.path.basename(final_match),
                            "networth_2018": networth,
                            "top_donors":    donors,
                            "top_industries": industries,
                            "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        # skip if identical
                        if overlay_data != last_overlay_data:
                            logging.info("Overlay updated: %s", json.dumps(overlay_data))
                            with open(overlay_path, "w") as f:
                                json.dump(overlay_data, f, indent=2)
                            last_overlay_data = overlay_data

                        #frame_fn = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                        #cv2.imwrite(frame_fn, frame)
                    else:
                        logging.debug(f"Repeat match – skipping update at frame {frame_count}.")
                else:
                    print(f"No reliable face match at frame {frame_count}.")
            # else: no faces → continue
        except Exception as e:
            logging.exception("Error at frame %d", frame_count)

    frame_count += 1

    if display_live_feed:
        cv2.imshow("OBS Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video.release()
cv2.destroyAllWindows()