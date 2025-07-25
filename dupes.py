import csv

INPUT_CSV = 'opensecrets.csv'
OUTPUT_CSV = 'opensecrets1.csv'

# Use a set to track seen rows
deduped_rows = set()

with open(INPUT_CSV, newline='', encoding='utf-8') as infile, open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    writer.writerow(header)

    for row in reader:
        row_tuple = tuple(row)
        if row_tuple not in deduped_rows:
            deduped_rows.add(row_tuple)
            writer.writerow(row)