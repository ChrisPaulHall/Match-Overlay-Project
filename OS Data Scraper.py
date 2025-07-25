import requests
from bs4 import BeautifulSoup
import csv
import time
import os

INPUT_CSV = "OSsource.csv"
OUTPUT_CSV = "OS_scraped.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

CONTRIB_COLS = [f"contributor_{i+1}" for i in range(5)]
INDUSTRY_COLS = [f"industry_{i+1}" for i in range(5)]

def extract_contributors(soup):
    contributors = []
    contrib_section = soup.find("div", id="contributors")
    if contrib_section:
        table = contrib_section.find_next("table")
        if table:
            rows = table.find_all("tr")[1:]  # Skip header
            for row in rows[:5]:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    name = cols[0].get_text(strip=True)
                    amount = cols[1].get_text(strip=True)
                    contributors.append(f"{name} ({amount})")
    return contributors

def extract_industries(soup):
    industries = []
    industry_section = soup.find("div", id="industries")
    if industry_section:
        table = industry_section.find_next("table")
        if table:
            rows = table.find_all("tr")[1:]  # Skip header
            for row in rows[:5]:
                cols = row.find_all("td")
                if len(cols) >= 2:
                    name = cols[0].get_text(strip=True)
                    amount = cols[1].get_text(strip=True)
                    industries.append(f"{name} ({amount})")
    return industries

def scrape_summary_page(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        contributors = extract_contributors(soup)
        industries = extract_industries(soup)
        return contributors, industries
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return [], []

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Input CSV '{INPUT_CSV}' not found.")
        return

    with open(INPUT_CSV, newline='', encoding='utf-8') as infile, \
         open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = ['first_name', 'last_name', 'slug_url'] + CONTRIB_COLS + INDUSTRY_COLS
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            slug_url = row.get("slug_url", "").strip()
            if not slug_url or not slug_url.startswith("http"):
                print(f"⚠️ Skipping row without valid slug_url: {row}")
                continue

            print(f"🔍 Scraping: {slug_url}")
            contributors, industries = scrape_summary_page(slug_url)

            contributors += [""] * (5 - len(contributors))
            industries += [""] * (5 - len(industries))

            outrow = {
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "slug_url": slug_url,
            }
            for idx in range(5):
                outrow[f"contributor_{idx+1}"] = contributors[idx]
                outrow[f"industry_{idx+1}"] = industries[idx]
            writer.writerow(outrow)

            writer.writerow({
                "first_name": row.get("first_name", ""),
                "last_name": row.get("last_name", ""),
                "slug_url": slug_url,
                "top_contributors": "; ".join(contributors),
                "top_industries": "; ".join(industries)
            })

            time.sleep(1.0)  

    print(f"✅ Done. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
