import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import shelve
import threading
import os
import json
import csv
import atexit

# --- CONFIGURATION ---
CACHE_TTL = timedelta(hours=1)
USE_PERSISTENT_CACHE = True
SHELVE_PATH        = "donors_cache_shelve.db"
JSON_PATH          = "donors_cache.json"
CSV_PATH           = "donors_cache.csv"

# in-memory cache: slug -> (timestamp, data)
_memory_cache = {}
_cache_lock   = threading.Lock()

# persistent on-disk cache
_persistent_cache = None

def get_persistent_cache():
    global _persistent_cache
    if not USE_PERSISTENT_CACHE:
        return None
    with _cache_lock:
        if _persistent_cache is None:
            try:
                _persistent_cache = shelve.open(SHELVE_PATH)
            except Exception as e:
                logging.warning(f"Could not open shelve database: {e}")
                _persistent_cache = None
        return _persistent_cache

def _load_from_persistent(slug):
    cache = get_persistent_cache()
    if not cache:
        return None
    with _cache_lock:
        if slug not in cache:
            return None
        entry = cache[slug]
    ts = datetime.fromisoformat(entry["ts"])
    if datetime.now() - ts < CACHE_TTL:
        return entry["data"]
    return None

def _save_to_persistent(slug, data):
    cache = get_persistent_cache()
    if not cache:
        return
    with _cache_lock:
        cache[slug] = {"ts": datetime.now().isoformat(), "data": data}
        cache.sync()

def _dump_full_json_cache():
    """
    Write all in-memory cache entries out to JSON_PATH.

    Output Format:
        The JSON file will contain a dictionary where each key is a slug and the value is an object:
        {
            "slug": {
                "ts": "<ISO formatted timestamp>",
                "data": <donor data>
            },
            ...
        }

    Error Handling:
        If an error occurs during writing, a warning will be logged and the function will not raise an exception.
    """
    try:
        serializable = {
            slug: {"ts": ts.isoformat(), "data": data}
            for slug, (ts, data) in _memory_cache.items()
        }
        with open(JSON_PATH, "w", encoding="utf-8") as jf:
            json.dump(serializable, jf, indent=2)
    except Exception as e:
        logging.warning(f"Couldn’t write full JSON cache: {e}")

def _append_to_csv_log(slug, data, ts):
    """Append this fetch to CSV_PATH, one row per donor."""
    try:
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=["slug","donor_name","amount","value","timestamp"])
            if not file_exists:
                writer.writeheader()
            for d in data:
                writer.writerow({
                    "slug":      slug,
                    "donor_name":d["name"],
                    "amount":    d["amount"],
                    "value":     d["value"],
                    "timestamp": ts.isoformat()
                })
    except Exception as e:
        logging.warning(f"Couldn’t append to CSV log: {e}")

def fetch_top_donors(name_slug):
    now = datetime.now()

    # 1) in-memory
    with _cache_lock:
        entry = _memory_cache.get(name_slug)
    if entry:
        ts, data = entry
        if now - ts < CACHE_TTL:
            return data

    # 2) shelve
    if USE_PERSISTENT_CACHE:
        data = _load_from_persistent(name_slug)
        if data is not None:
            with _cache_lock:
                _memory_cache[name_slug] = (now, data)
            return data

    # 3) scrape
    url = f"https://www.opensecrets.org/members-of-congress/{name_slug}/summary"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SpeakerOverlay/1.0)"}
    try:
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.warning(f"Request failed for {name_slug}: {e}")
        return []

    # Parse HTML and extract donor data
    soup = BeautifulSoup(resp.text, "html.parser")
    tbl = soup.find("table", class_="donor-table")
    result = []
    if tbl:
        rows = tbl.select("tr")
        for row in rows[1:6]:
            cols = row.find_all("td")
            if len(cols) >= 2:
                nm = cols[0].get_text(strip=True)
                amt_text = cols[1].get_text(strip=True).replace("$", "").replace(",", "")
                try:
                    val = int(amt_text)
                except ValueError:
                    val = 0
                import locale
                locale.setlocale(locale.LC_ALL, '')  # Set to user's default locale
                try:
                    amount_str = locale.currency(val, grouping=True)
                except Exception:
                    amount_str = f"${val:,}"
                result.append({
                    "name":   nm,
                    "amount": amount_str,
                    "value":  val
                })
    # 4) save to persistent cache
    if USE_PERSISTENT_CACHE:
        _save_to_persistent(name_slug, result)
    # 5) dump out to JSON + CSV

    # Only dump JSON cache every N fetches or after a time interval
    if not hasattr(fetch_top_donors, "_fetch_count"):
        fetch_top_donors._fetch_count = 0
        fetch_top_donors._last_dump = datetime.now()
    fetch_top_donors._fetch_count += 1
    if (
        fetch_top_donors._fetch_count >= 10 or
        (datetime.now() - fetch_top_donors._last_dump).total_seconds() > 300
    ):
        _dump_full_json_cache()
        fetch_top_donors._fetch_count = 0
        fetch_top_donors._last_dump = datetime.now()

    _append_to_csv_log(name_slug, result, now)

    return result

def close_persistent_cache():
    global _persistent_cache
    with _cache_lock:
        if _persistent_cache is not None:
            try:
                _persistent_cache.close()
                _persistent_cache = None
            except Exception as e:
                logging.warning(f"Error closing persistent cache: {e}")
import atexit
atexit.register(close_persistent_cache)

if __name__ == "__main__":
    # Replace 'nancy-pelosi' with the desired member slug
    donors = fetch_top_donors("nancy-pelosi")
    print(donors)
