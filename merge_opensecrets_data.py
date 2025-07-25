import csv

# Files
BASE_FILE = "opensecrets2.csv"                  # Your main file with image filenames etc.
SCRAPED_FILE = "opensecrets_scraped_summary.csv"  # The one we just scraped
OUTPUT_FILE = "opensecrets_merged.csv"

# Load scraped data into a dictionary for fast lookup
scraped_data = {}
with open(SCRAPED_FILE, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['first_name'].strip().lower(), row['last_name'].strip().lower())
        scraped_data[key] = {
            'top_contributors': row['top_contributors'],
            'top_industries': row['top_industries']
        }

# Merge with your base data
with open(BASE_FILE, newline='', encoding='utf-8') as infile, \
     open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:

    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + ['top_contributors', 'top_industries']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        key = (row['first_name'].strip().lower(), row['last_name'].strip().lower())
        extras = scraped_data.get(key, {'top_contributors': '', 'top_industries': ''})
        row.update(extras)
        writer.writerow(row)

print(f"✅ Merged data written to: {OUTPUT_FILE}")
