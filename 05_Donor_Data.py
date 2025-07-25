import csv
import requests
from bs4 import BeautifulSoup
import time

INPUT_CSV = 'opensecrets1.csv'
OUTPUT_CSV = 'opensecrets3.csv'

# Function to scrape OpenSecrets summary pages
def scrape_opensecrets_summary(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract top contributors
        contrib_section = soup.find('section', id='contributors')
        contribs = contrib_section.find_all('div', class_='contributor-name')[:3] if contrib_section else []
        amounts = contrib_section.find_all('div', class_='contributor-amount')[:3] if contrib_section else []
        top_contributors = [(c.get_text(strip=True), a.get_text(strip=True)) for c, a in zip(contribs, amounts)]

        # Extract top industries
        industry_section = soup.find('section', id='industries')
        industries = industry_section.find_all('div', class_='industry-name')[:3] if industry_section else []
        ind_amounts = industry_section.find_all('div', class_='industry-amount')[:3] if industry_section else []
        top_industries = [(i.get_text(strip=True), a.get_text(strip=True)) for i, a in zip(industries, ind_amounts)]

        top_contributors += [('', '')] * (3 - len(top_contributors))
        top_industries += [('', '')] * (3 - len(top_industries))

        return top_contributors, top_industries

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return [('', '')] * 3, [('', '')] * 3

# Read opensecrets1.csv and write to opensecrets3.csv with new data
with open(INPUT_CSV, newline='', encoding='utf-8') as infile, open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames + [
        'Top Contributor 1', 'amount 1',
        'Top Contributor 2', 'amount 2',
        'Top Contributor 3', 'amount 3',
        'Top Industry 1', 'industry amount 1',
        'Top Industry 2', 'industry amount 2',
        'Top Industry 3', 'industry amount 3'
    ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        url = row.get('slug_url', '').strip()
        if url:
            top_contributors, top_industries = scrape_opensecrets_summary(url)
            top_contributors += [('', '')] * (3 - len(top_contributors))
            top_industries += [('', '')] * (3 - len(top_industries))
            for i in range(3):
                row[f'Top Contributor {i+1}'] = top_contributors[i][0]
                row[f'amount {i+1}'] = top_contributors[i][1]
                row[f'Top Industry {i+1}'] = top_industries[i][0]
                row[f'industry amount {i+1}'] = top_industries[i][1]
        else:
            for i in range(3):
                row[f'Top Contributor {i+1}'] = ''
                row[f'amount {i+1}'] = ''
                row[f'Top Industry {i+1}'] = ''
                row[f'industry amount {i+1}'] = ''
        writer.writerow(row)
        time.sleep(1)