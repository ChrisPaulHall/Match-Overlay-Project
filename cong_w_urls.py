from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time

BASE_URL = "https://www.opensecrets.org"
LIST_URL = f"{BASE_URL}/members-of-congress/members-list?cong_no=118&cycle=2024"
OUTPUT_CSV = "congress_members_with_urls.csv"

driver = webdriver.Chrome()
driver.get(LIST_URL)
try:
    WebDriverWait(driver, 60).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "table.DataTable"))
    )
except Exception as e:
    print("Timeout waiting for table. Saving page source for debugging.")
    with open("debug_page.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.quit()
    raise
time.sleep(2)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

rows = soup.select('table.DataTable tbody tr')
print(f"Found {len(rows)} rows in table.")

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['first_name', 'last_name', 'slug_name', 'slug_url'])
    for row in rows:
        name_cell = row.select_one('td a')
        if name_cell and name_cell.get('href'):
            full_name = name_cell.text.strip()
            slug_url = BASE_URL + name_cell['href']
            slug_name = name_cell['href'].split('/')[-2]
            if "," in full_name:
                last_name, first_name = [x.strip() for x in full_name.split(",", 1)]
            else:
                first_name, last_name = '', full_name
            writer.writerow([first_name, last_name, slug_name, slug_url])
        else:
            print("No name cell or href found in row:", row)
print("Done. Check congress_members_with_urls.csv for output.")

