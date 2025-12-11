#!/usr/bin/env python3
"""Scrape Top Contributors/Industries from OpenSecrets summary pages.

This script reads member URLs from an input CSV and scrapes campaign finance
data from OpenSecrets.org public pages. No API key required.

Output CSV contains: bioguide_id, cid, first_name, last_name, slug_url,
top_contributors, top_industries, and period information.
"""

import argparse
import csv
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "01_data" / "members_donor_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "01_data" / "00members_donor_summary.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache_pages"

# ---------- Year/period parsing ----------
_DASH = r"[\-–—]"
_YEAR = r"(?:19|20)\d{2}"
YEAR_RANGE_RE = re.compile(rf"(?P<start>{_YEAR})\s*{_DASH}\s*(?P<end>{_YEAR})")
YEAR_SINGLE_RE = re.compile(rf"(?P<single>{_YEAR})")

def _extract_period_from_text(text: str) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    m = YEAR_RANGE_RE.search(text)
    if m:
        return m.group("start"), m.group("end"), m.group(0)
    m = YEAR_SINGLE_RE.search(text)
    if m:
        y = m.group("single")
        return y, y, y
    return "", "", ""

# ---------- Table parsing from heading ----------
def _parse_table_name_amount(table: Tag, top_n: int = 5) -> List[str]:
    out: List[str] = []
    if not table:
        return out
    rows = table.find_all("tr")
    if not rows:
        return out
    start_idx = 1 if rows and rows[0].find_all("th") else 0
    for tr in rows[start_idx:]:
        tds = tr.find_all("td")
        if len(tds) >= 2:
            name = tds[0].get_text(strip=True)
            total = tds[1].get_text(strip=True)
            if name:
                out.append(f"{name} ({total})")
        if len(out) >= top_n:
            break
    return out

def _find_heading(soup: BeautifulSoup, label: str) -> Optional[Tag]:
    for h in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
        txt = h.get_text(" ", strip=True).lower()
        if label.lower() in txt:
            return h
    return None

def extract_section_by_heading(soup: BeautifulSoup, label: str, top_n: int = 5):
    """
    Locate heading (e.g., 'Top Contributors'), read period from heading text,
    then parse the next table for name/amount pairs.
    Returns: (items:list[str], start_year:str, end_year:str, raw_period:str)
    """
    h = _find_heading(soup, label)
    if not h:
        return [], "", "", ""
    start, end, raw = _extract_period_from_text(h.get_text(" ", strip=True))
    table = h.find_next("table")
    items = _parse_table_name_amount(table, top_n=top_n)
    return items, start, end, raw

# ---------- HTTP/session/retry ----------
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

def build_session(timeout: int, headers: dict) -> requests.Session:
    sess = requests.Session()
    sess.headers.update(headers)
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    # default timeout wrapper
    original_get = sess.get
    def get_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_get(*args, **kwargs)
    sess.get = get_with_timeout  # type: ignore
    return sess

# ---------- Caching ----------
def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def cache_path_for(url: str, cache_dir: str) -> str:
    parsed = urlparse(url)
    base = (parsed.path.rstrip("/").split("/")[-1] or "index")
    return os.path.join(cache_dir, f"{base}-{_short_hash(url)}.html")

def get_html(url: str, session: requests.Session, cache_dir: Optional[str], referer: Optional[str]) -> str:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cpath = cache_path_for(url, cache_dir)
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()

    headers = {}
    if referer:
        headers["Referer"] = referer
    resp = session.get(url, headers=headers)
    resp.raise_for_status()
    html = resp.text

    if cache_dir:
        with open(cache_path_for(url, cache_dir), "w", encoding="utf-8") as fh:
            fh.write(html)
    return html

# ---------- Utilities ----------
def valid_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)

# ---------- Main extractors ----------
def extract_summary_meta(soup: BeautifulSoup, top_n: int = 5) -> dict:
    contrib_items, c_start, c_end, _ = extract_section_by_heading(soup, "Top Contributors", top_n=top_n)
    industry_items, i_start, i_end, _ = extract_section_by_heading(soup, "Top Industries", top_n=top_n)

    contrib_period_str = f"{c_start} - {c_end}" if c_start and c_end and c_start != c_end else (c_start or "")
    industry_period_str = f"{i_start} - {i_end}" if i_start and i_end and i_start != i_end else (i_start or "")

    return {
        "contributors": contrib_items,
        "industries": industry_items,
        "contributors_period_start": c_start or "",
        "contributors_period_end": c_end or "",
        "contributors_period": contrib_period_str,
        "industries_period_start": i_start or "",
        "industries_period_end": i_end or "",
        "industries_period": industry_period_str,
    }

def scrape_summary_page(
    url: str,
    session: requests.Session,
    parser: str,
    cache_dir: Optional[str],
    referer: Optional[str],
    top_n: int,
) -> dict:
    try:
        html = get_html(url, session=session, cache_dir=cache_dir, referer=referer)
        soup = BeautifulSoup(html, parser)  # 'lxml' if installed, else 'html.parser'
        return extract_summary_meta(soup, top_n=top_n)
    except Exception as e:
        logging.error("Error scraping %s: %s", url, e)
        return {
            "contributors": [],
            "industries": [],
            "contributors_period_start": "",
            "contributors_period_end": "",
            "contributors_period": "",
            "industries_period_start": "",
            "industries_period_end": "",
            "industries_period": "",
        }

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Scrape Top Contributors/Industries + period from OpenSecrets summary pages.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV with columns: bioguide_id,cid,first_name,last_name,slug_url")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    ap.add_argument("--delay", type=float, default=1.5, help="Delay between requests (seconds)")
    ap.add_argument("--top-n", type=int, default=5, help="How many items to keep per section")
    ap.add_argument("--resume", action="store_true", help="Skip rows whose slug_url already exists in output")
    ap.add_argument("--timeout", type=int, default=15, help="HTTP timeout (seconds)")
    ap.add_argument("--parser", default="html.parser", choices=["html.parser", "lxml"], help="BeautifulSoup parser")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory to cache fetched HTML (set '' to disable)")
    ap.add_argument("--referer", default=None, help="Optional Referer header value")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    input_csv = args.input
    output_csv = args.output
    cache_dir = args.cache_dir or None

    # Resolve input CSV relative to script dir if not found in CWD
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(input_csv) and not os.path.exists(input_csv):
        alt_input = os.path.join(script_dir, input_csv)
        if os.path.exists(alt_input):
            input_csv = alt_input
            logging.info("Resolved input CSV to %s", input_csv)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(input_csv):
        logging.error("Input CSV '%s' not found.", input_csv)
        return 1

    session = build_session(timeout=args.timeout, headers=DEFAULT_HEADERS)

    # Resumability
    done = set()
    mode = "w"
    if args.resume and os.path.exists(output_csv):
        with open(output_csv, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                url = (row.get("slug_url") or "").strip()
                if url:
                    done.add(url)
        mode = "a"

    fieldnames = [
        "bioguide_id", "cid",
        "first_name", "last_name", "slug_url",
        "top_contributors", "top_industries",
        "contributors_period_start", "contributors_period_end", "contributors_period",
        "industries_period_start", "industries_period_end", "industries_period",
    ]

    # Count total rows for progress display
    with open(input_csv, newline="", encoding="utf-8") as f:
        total_rows = sum(1 for _ in csv.DictReader(f))

    stats = {"processed": 0, "skipped_done": 0, "skipped_invalid": 0, "errors": 0}

    with open(input_csv, newline="", encoding="utf-8") as infile, \
         open(output_csv, mode, newline="", encoding="utf-8") as outfile:

        reader = csv.DictReader(infile)
        required = {"bioguide_id", "cid", "first_name", "last_name", "slug_url"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            logging.error("Input CSV missing expected columns: %s", ", ".join(sorted(missing)))
            return 1

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        for idx, row in enumerate(reader, 1):
            slug_url = (row.get("slug_url") or "").strip()
            first_name = (row.get("first_name") or "").strip()
            last_name = (row.get("last_name") or "").strip()
            bioguide_id = (row.get("bioguide_id") or "").strip()
            cid = (row.get("cid") or "").strip()

            if not valid_url(slug_url):
                logging.warning("Skipping row with invalid slug_url: %r", row)
                stats["skipped_invalid"] += 1
                continue
            if slug_url in done:
                if args.verbose:
                    logging.info("Skipping already processed: %s", slug_url)
                stats["skipped_done"] += 1
                continue

            logging.info("[%d/%d] Scraping: %s %s", idx, total_rows, first_name, last_name)
            meta = scrape_summary_page(
                slug_url,
                session=session,
                parser=args.parser,
                cache_dir=cache_dir,
                referer=args.referer,
                top_n=args.top_n,
            )

            # Track errors (empty results likely mean scrape failed)
            if not meta["contributors"] and not meta["industries"]:
                stats["errors"] += 1

            writer.writerow({
                "bioguide_id": bioguide_id,
                "cid": cid,
                "first_name": first_name,
                "last_name": last_name,
                "slug_url": slug_url,
                "top_contributors": "; ".join(meta["contributors"]),
                "top_industries": "; ".join(meta["industries"]),
                "contributors_period_start": meta["contributors_period_start"],
                "contributors_period_end": meta["contributors_period_end"],
                "contributors_period": meta["contributors_period"],
                "industries_period_start": meta["industries_period_start"],
                "industries_period_end": meta["industries_period_end"],
                "industries_period": meta["industries_period"],
            })

            stats["processed"] += 1
            time.sleep(max(0.0, args.delay))

    # Summary
    logging.info("=" * 50)
    logging.info("SUMMARY")
    logging.info("  Processed: %d", stats["processed"])
    logging.info("  Skipped (already done): %d", stats["skipped_done"])
    logging.info("  Skipped (invalid URL): %d", stats["skipped_invalid"])
    logging.info("  Empty results (possible errors): %d", stats["errors"])
    logging.info("Output: %s", output_csv)
    logging.info("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
