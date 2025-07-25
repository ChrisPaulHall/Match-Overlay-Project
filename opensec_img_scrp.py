import os
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO

INPUT_CSV = 'opensecrets1.csv'
OUTPUT_FOLDER = 'faces_db'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

def get_next_available_filename(base_path):
    # Check FLast.jpg to FLast4.jpg
    for i in range(5):
        suffix = '' if i == 0 else str(i)
        path = f"{base_path}{suffix}.jpg"
        if not os.path.exists(path):
            return path
    return None  # All 5 slots used

def download_photo(name_slug, url):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for profile image
        img_tag = None
        for img in soup.find_all('img'):
            print(img)  # Debug: see all images
            if 'profile' in img.get('src', '') or 'member' in img.get('src', ''):
                img_tag = img
                break

        if img_tag and img_tag.get('src'):
            img_url = urljoin(url, img_tag['src'])
            img_response = requests.get(img_url, headers=headers)
            img_response.raise_for_status()

            base_filename = os.path.join(OUTPUT_FOLDER, name_slug)
            save_path = get_next_available_filename(base_filename)

            if save_path:
                image = Image.open(BytesIO(img_response.content))
                image.save(save_path)
                print(f"Saved: {save_path}")
            else:
                print(f"All 5 image slots already used for {name_slug}")
        else:
            print(f"No image found for {name_slug} at {url}")

    except Exception as e:
        print(f"Error downloading photo for {name_slug}: {e}")

with open(INPUT_CSV, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        first = row['\ufefffirst_name'].strip()
        last = row['last_name'].strip()
        if not first or not last:
            continue
        name_slug = first[0] + last
        url = row.get('slug_url', '').strip()
        if url:
            download_photo(name_slug, url)
