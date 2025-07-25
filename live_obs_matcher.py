import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import logging
logging.getLogger('deepface').setLevel(logging.ERROR)  # Suppress DeepFace logs

import warnings
warnings.filterwarnings("ignore")  # Suppress Python warnings
from deepface import DeepFace
import numpy as np

cam_index = 1
video = cv2.VideoCapture(cam_index)

frame_count = 0
FRAME_INTERVAL = 30
FACE_MATCH_THRESHOLD = 0.7

frames_dir = "matched_frames"
os.makedirs(frames_dir, exist_ok=True)

while True:
    success, frame = video.read()
    if not success:
        print("Failed to read from OBS virtual camera.")
        break

    if frame_count % FRAME_INTERVAL == 0:
        try:
            result = DeepFace.find(img_path=frame, db_path="faces_db", enforce_detection=True, refresh_database=False, threshold=FACE_MATCH_THRESHOLD)
            if len(result) > 0 and len(result[0]) > 0:
                best = result[0].iloc[0]
                match = best['identity']
                print(f"Match found: {os.path.basename(match)} at frame {frame_count} (distance: {best['distance']:.2f})")
            else:
                print(f"No reliable match at frame {frame_count}.")
        except Exception as e:
            print(f"Exception at frame {frame_count}: {e}")

        # Save frames used for matching 
        frame_filename = f"{frames_dir}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)
    
    frame_count += 1

video.release()
cv2.destroyAllWindows()

                    # Save frames used for matching 
                    frame_filename = f"{frames_dir}/frame_{frame_count}.jpg"
                    cv2.imwrite(frame_filename, frame)