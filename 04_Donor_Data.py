import csv
import pandas as pd

output_excel_filename = "opensecrets4.xlsx"
data = []

try:
    with open("name_map_with_slugs.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            entry = {
                "first_name": row.get("first_name", "").strip(),
                "last_name": row.get("last_name", "").strip(),
                "nananame": row.get("nananame", "").strip(),
                "slug_url": row.get("slug_url", "").strip(),
                "Top Contributor 1": "",
                "amount 1": "",
                "Top Contributor 2": "",
                "amount 2": "",
                "Top Contributor 3": "",
                "amount 3": "",
                "Top Industry 1": "",
                "industry amount 1": "",
                "Top Industry 2": "",
                "industry amount 2": "",
                "Top Industry 3": "",
                "industry amount 3": "",
            }
            data.append(entry)
except FileNotFoundError:
    print("Error: 'name_map_with_slugs.csv' not found in the working directory.")

df = pd.DataFrame(data)
df.to_excel(output_excel_filename, index=False)