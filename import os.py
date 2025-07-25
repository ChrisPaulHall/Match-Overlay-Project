import os
import csv
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import pandas as pd

# File paths
input_csv = "opensecrets.xlsx"
output_csv = "opensecrets_updated.xlsx"
faces_folder = "faces_db"
os.makedirs(faces_folder, exist_ok=True)

# Load Excel data
df = pd.read_excel(input_csv)

# Loop through each row
top_contributors = []
top_industries = []

for index, row in df.iterrows():
    slug_url = row.get("slug_url", "")
    if not isinstance(slug_url, str) or not slug_url.strip():
        top_contributors.append("")
        top_industries.append("")
        continue

    full_url = f"https://www.opensecrets.org/members-of-congress/{slug_url}/summary"
    print(f"Scraping: {full_url}")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(full_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract top 3 contributors
        contrib_section = soup.find("section", id="contributors")
        contributors = []
        if contrib_section:
            rows = contrib_section.select("table tr")[1:4]  # skip header, get top 3
            for tr in rows:
                cols = tr.find_all("td")
                if len(cols) >= 2:
                    contributors.append(f"{cols[0].text.strip()} ({cols[1].text.strip()})")

        # Extract top 3 industries
        industry_section = soup.find("section", id="industries")
        industries = []
        if industry_section:
            rows = industry_section.select("table tr")[1:4]  # skip header, get top 3
            for tr in rows:
                cols = tr.find_all("td")
                if len(cols) >= 2:
                    industries.append(f"{cols[0].text.strip()} ({cols[1].text.strip()})")

        top_contributors.append("; ".join(contributors))
        top_industries.append("; ".join(industries))

        # Try to download image
        if pd.notna(row['first_name']) and pd.notna(row['last_name']):
            search_name = f"{row['first_name']} {row['last_name']}"
            from wikipedia import page
            from wikipedia.exceptions import PageError
            try:
                p = page(search_name, auto_suggest=False)
                for img_url in p.images:
                    if img_url.lower().endswith(".jpg") and not any(x in img_url.lower() for x in ["seal", "logo", "signature"]):
                        img_data = requests.get(img_url, headers=headers).content
                        img = Image.open(BytesIO(img_data)).convert("RGB")
                        filename = f"{row['first_name'][0]}{row['last_name']}.jpg"
                        img.save(os.path.join(faces_folder, filename))
                        print(f"✅ Saved image for {search_name}")
                        break
            except PageError:
                print(f"⚠️ No Wikipedia page found for {search_name}")

    except Exception as e:
        print(f"❌ Error processing {slug_url}: {e}")
        top_contributors.append("")
        top_industries.append("")

# Add to DataFrame and save

df["top_3_contributors_2023_2024"] = top_contributors
df["top_3_industries_2023_2024"] = top_industries

print("Saving to", output_csv)
df.to_excel(output_csv, index=False)
