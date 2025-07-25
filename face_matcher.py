from deepface import DeepFace

# Compare testx.jpg to all the images in faces_db/
result = DeepFace.find(img_path="test7.jpg", db_path="faces_db", enforce_detection=False)

# If a match is found, print it
if len(result) > 0 and len(result[0]) > 0:
    best_match = result[0].iloc[0]['identity']
    print(f"Best match: {best_match}")
else:
    print("No match found.")
