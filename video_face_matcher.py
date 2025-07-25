import cv2
from deepface import DeepFace
import os
import numpy as np

video = cv2.VideoCapture("testvid4.mp4")
frame_count = 0

FACE_MATCH_THRESHOLD = 0.7
FRAME_INTERVAL = 30

frames_dir = "matched_frames"
os.makedirs(frames_dir, exist_ok=True)

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    if frame_count % FRAME_INTERVAL == 0:
        try:
            result = DeepFace.find(img_path=frame, db_path="faces_db", enforce_detection=False, refresh_database=False, threshold=FACE_MATCH_THRESHOLD)
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