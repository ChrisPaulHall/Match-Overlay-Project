import csv
import re

input_file = "name_map_updated.csv"
output_file = "name_map_with_slugs.csv"

def clean_name(name):
    # Remove punctuation and normalize spacing
    name = re.sub(r"[^a-zA-Z\s-]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.lower().replace(" ", "-")

updated_rows = []
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames.copy()

    if "slug_url" not in fieldnames:
        fieldnames.append("slug_url")

    for row in reader:
        first_clean = clean_name(row["first_name"])
        last_clean = clean_name(row["last_name"])
        cid = row.get("cid", "").strip()

        # Create and store slug
        slug = f"{first_clean}-{last_clean}"
        row["nananame"] = slug  # Replace column 8 value with the slug

        # Create full URL if CID is present
        row["slug_url"] = (
            f"https://www.opensecrets.org/members-of-congress/{slug}/summary?cid={cid}"
            if cid else ""
        )

        updated_rows.append(row)

# Write updated CSV
with open(output_file, "w", newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print("✅ name_map_with_slugs.csv generated with cleaned slugs and URLs.")


