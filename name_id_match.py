import csv

crp_lookup = {}
with open("crp_ids.csv", newline='', encoding='utf-8') as crp_file:
    reader = csv.DictReader(crp_file)
    for row in reader:
        key = (row["first"].strip().lower(), row["last"].strip().lower())
        crp_lookup[key] = {
            "cid": row.get("cid", "").strip(),
            "feccandid": row.get("feccandid", "").strip()
        }

# Read name_map.csv, match names, and add cid and feccandid
updated_rows = []
with open("name_map.csv", newline='', encoding='utf-8') as name_file:
    reader = csv.DictReader(name_file)
    fieldnames = reader.fieldnames + ["cid", "feccandid"] if "cid" not in reader.fieldnames else reader.fieldnames
    for row in reader:
        key = (row["first_name"].strip().lower(), row["last_name"].strip().lower())
        match = crp_lookup.get(key, {})
        row["cid"] = match.get("cid", "")
        row["feccandid"] = match.get("feccandid", "")
        updated_rows.append(row)

# Write the updated CSV
with open("name_map_updated.csv", "w", newline='', encoding='utf-8') as out_file:
    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print("✅ name_map_updated.csv created with cid and feccandid columns.")
