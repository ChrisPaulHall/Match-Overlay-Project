import os
import csv
import requests
from PIL import Image
from io import BytesIO
import wikipedia
from collections import defaultdict

# Prepare output folder
faces_folder = "faces_db"
os.makedirs(faces_folder, exist_ok=True)

# Load and group name_map.csv entries by full name
name_map_csv_path = "name_map.csv"
grouped_entries = defaultdict(list)

with open(name_map_csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        full_name = f"{row['first_name'].strip()} {row['last_name'].strip()}"
        filename = row["\ufefffilename"].strip()
        grouped_entries[full_name].append(filename)

# Helper to extract best Wikipedia image
def get_best_wikipedia_image_url(name):
    try:
        page = wikipedia.page(name, auto_suggest=False)
        for img_url in page.images:
            if img_url.lower().endswith(".jpg") and not any(x in img_url.lower() for x in ['seal', 'signature', 'logo']):
                return img_url
    except Exception as e:
        print(f"❌ Wikipedia error for {name}: {e}")
    return None

# Download and save one image per politician
for full_name, filenames in grouped_entries.items():
    all_exist = all(os.path.exists(os.path.join(faces_folder, fn)) for fn in filenames)
    if all_exist:
        print(f"🟡 Skipped (all exist): {full_name}")
        continue

    image_url = get_best_wikipedia_image_url(full_name)
    if not image_url:
        print(f"⚠️ No image found for {full_name}")
        continue

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(image_url, headers=headers, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            for filename in filenames:
                save_path = os.path.join(faces_folder, filename)
                if not os.path.exists(save_path):
                    img.save(save_path)
            print(f"✅ Saved image for: {full_name}")
        else:
            print(f"❌ Download failed for {full_name} (status {response.status_code})")
    except Exception as e:
        print(f"❌ Error downloading image for {full_name}: {e}")
