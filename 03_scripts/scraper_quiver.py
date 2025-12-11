#!/usr/bin/env python3
"""Fetch Quiver net-worth estimates, top holdings, and traded sectors.

This script reads `01_data/members_face_lookup.csv`, de-duplicates entries by
BioGuide ID, fetches the holdings payload used by Quiver's politician detail
page, and writes a summary CSV containing the current net-worth estimate,
the top holdings breakdown, and the "Top Traded Sectors" snippet per member.

The output CSV is stored at `01_data/00members_holdings_and_sectors.csv`.

Rate Limiting Strategy:
- Global RateLimiter ensures max 1 request/second across all threads
- Reduced concurrent workers from 5 to 2 to avoid burst requests
- Added 0.5s delay between the two API calls per member
- All requests use exponential backoff with retry logic
- Jitter added to avoid synchronized request bursts
- Incremental saves every 10 records for better resume capability
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "01_data" / "members_face_lookup.csv"
OUTPUT_CSV = PROJECT_ROOT / "01_data" / "00members_holdings_and_sectors.csv"
DEFAULT_CACHE_DIR = ".cache_quiver"

REQUEST_DELAY_SEC = 1.0  # 1 second minimum between requests (increased from 0.1)
MAX_BACKOFF_SEC = 15.0  # reduced from 30 to fail faster
MAX_RETRIES = 4  # reduced from 8 to avoid excessive retries
TOP_SECTOR_LIMIT = 5
SECTOR_PATTERN = re.compile(r"let\s+sectorData\s*=\s*({.*?});", re.DOTALL)

# Quiver API settings
ENV_TOKEN_KEY = "QUIVER_API_TOKEN"
DEFAULT_TOKEN_FILE = "quiver_token.txt"

OUTPUT_FIELDNAMES = [
    "bioguide_id",
    "first_name",
    "last_name",
    "middle_name",
    "net_worth_estimate",
    "normalized_net_worth",
    "top_holdings",
    "top_traded_sectors",
]


def load_token() -> str:
    """Resolve the API token from env var or a local file."""
    env_token = os.getenv(ENV_TOKEN_KEY)
    if env_token:
        return env_token.strip()

    candidate_paths = (
        Path.cwd() / DEFAULT_TOKEN_FILE,
        PROJECT_ROOT / DEFAULT_TOKEN_FILE,
        PROJECT_ROOT / "01_data" / DEFAULT_TOKEN_FILE,
    )

    for path in candidate_paths:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()

    raise ValueError(
        "No API token found. Set QUIVER_API_TOKEN or create quiver_token.txt."
    )


# Token loaded lazily in main() to allow --help without credentials
TOKEN: Optional[str] = None


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_second: float):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                # Add jitter to avoid synchronized bursts
                jitter = random.uniform(0, 0.2)
                time.sleep(sleep_time + jitter)
            self.last_request_time = time.time()


def build_session() -> requests.Session:
    """Build a session whose retries are handled explicitly by our backoff."""
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (Codex fetch script)",
        "X-Requested-With": "XMLHttpRequest",
    })
    # Keep the adapter for pooling but let _request_with_backoff drive retries.
    adapter = HTTPAdapter(max_retries=0, pool_connections=10, pool_maxsize=10)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    # Set default timeout
    original_get = sess.get
    def get_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", 15)  # reduced from 30 to fail faster
        return original_get(*args, **kwargs)
    sess.get = get_with_timeout  # type: ignore
    return sess


SESSION = build_session()
RATE_LIMITER = RateLimiter(requests_per_second=1.0)  # Max 1 request/second globally

CACHE_DIR: Optional[str] = None
CACHE_LOCK = threading.Lock()


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _cache_path_for(url: str, ext: str) -> Optional[str]:
    if not CACHE_DIR:
        return None
    parsed = urlparse(url)
    base = parsed.path.rstrip("/").split("/")[-1] or "index"
    filename = f"{base}-{_short_hash(url)}.{ext}"
    return os.path.join(CACHE_DIR, filename)


def _read_cache_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _write_cache_text(path: str, data: str) -> None:
    try:
        with CACHE_LOCK:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(data)
            os.replace(tmp_path, path)
    except OSError:
        pass


def _request_with_backoff(
    url: str,
    *,
    headers: Dict[str, str] | None = None,
    timeout: int = 15,
    purpose: str = "",
):
    """GET helper with exponential backoff for rate-limited endpoints."""

    delay = max(REQUEST_DELAY_SEC, 0.5)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, headers=headers, timeout=timeout)
        except requests.Timeout as exc:
            if attempt == MAX_RETRIES:
                print(f"WARN: {purpose or url} timed out after {timeout}s: {exc}", file=sys.stderr)
                return None
            print(f"WARN: {purpose or url} timeout on attempt {attempt}/{MAX_RETRIES}, retrying...", file=sys.stderr)
            time.sleep(min(delay, MAX_BACKOFF_SEC))
            delay = min(delay * 2, MAX_BACKOFF_SEC)
            continue
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"WARN: {purpose or url} request failed: {exc}", file=sys.stderr)
                return None
            print(f"WARN: {purpose or url} error on attempt {attempt}/{MAX_RETRIES}: {exc}, retrying...", file=sys.stderr)
            time.sleep(min(delay, MAX_BACKOFF_SEC))
            delay = min(delay * 2, MAX_BACKOFF_SEC)
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = delay
            else:
                wait = delay

            if attempt == MAX_RETRIES:
                print(
                    f"WARN: rate limited on {purpose or url}; giving up after {attempt} attempts",
                    file=sys.stderr,
                )
                return None

            time.sleep(min(wait, MAX_BACKOFF_SEC))
            delay = min(max(delay, wait) * 1.5, MAX_BACKOFF_SEC)
            continue

        if response.status_code >= 400:
            if attempt == MAX_RETRIES:
                print(
                    f"WARN: {purpose or url} returned HTTP {response.status_code}",
                    file=sys.stderr,
                )
                return None

            time.sleep(min(delay, MAX_BACKOFF_SEC))
            delay = min(delay * 1.5, MAX_BACKOFF_SEC)
            continue

        return response

    return None


def load_member_base_rows(path: Path) -> List[Dict[str, str]]:
    """Return ordered, unique member records with identity fields copied."""

    seen: OrderedDict[str, Dict[str, str]] = OrderedDict()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bioguide = (row.get("bioguide_id") or "").strip()
            if not bioguide or bioguide in seen:
                continue

            first = (
                row.get("first_name")
                or row.get("first")
                or row.get("given_name")
                or row.get("first_name1")
                or row.get("first_name2")
                or ""
            ).strip()
            last = (
                row.get("last_name")
                or row.get("last")
                or row.get("family_name")
                or row.get("last_name1")
                or row.get("last_name2")
                or ""
            ).strip()
            middle = (
                row.get("middle_name")
                or row.get("middle")
                or row.get("middle_name1")
                or ""
            ).strip()

            seen[bioguide] = {
                "bioguide_id": bioguide,
                "first_name": first,
                "last_name": last,
                "middle_name": middle,
                "net_worth_estimate": "",
                "normalized_net_worth": "",
                "top_holdings": "",
                "top_traded_sectors": "",
            }

    return list(seen.values())


def hydrate_existing_rows(path: Path, rows_index: Dict[str, Dict[str, str]]) -> Set[str]:
    """Populate base rows with any existing output data; return completed IDs."""

    done: Set[str] = set()
    if not path.exists():
        return done

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bioguide = (row.get("bioguide_id") or "").strip()
            if not bioguide:
                continue

            entry = rows_index.get(bioguide)
            if not entry:
                entry = {
                    "bioguide_id": bioguide,
                    "first_name": (row.get("first_name") or "").strip(),
                    "last_name": (row.get("last_name") or "").strip(),
                    "middle_name": (row.get("middle_name") or "").strip(),
                    "net_worth_estimate": "",
                    "normalized_net_worth": "",
                    "top_holdings": "",
                    "top_traded_sectors": "",
                }
                rows_index[bioguide] = entry

            for key in OUTPUT_FIELDNAMES:
                if key in row:
                    entry[key] = (row.get(key) or "").strip()

            # Only mark as done if we have actual data (any of the data fields populated)
            has_data = any([
                (row.get("net_worth_estimate") or "").strip(),
                (row.get("normalized_net_worth") or "").strip(),
                (row.get("top_holdings") or "").strip(),
                (row.get("top_traded_sectors") or "").strip(),
            ])
            if has_data:
                done.add(bioguide)

    return done


def build_profile_url(first: str, last: str, bioguide: str) -> str:
    """Base politician URL used for scraping and referers."""

    full_name = f"{first} {last}".strip() or bioguide
    safe_name = quote(full_name.replace(" ", " "))
    return (
        "https://www.quiverquant.com/congresstrading/politician/"
        f"{safe_name}-{bioguide}"
    )


def build_referer(first: str, last: str, bioguide: str) -> str:
    """Construct the Quiver referer URL expected by their endpoint."""

    return build_profile_url(first, last, bioguide) + "/net-worth"


def fetch_quiver_payload(bioguide: str, first: str, last: str) -> Dict:
    """Return the holdings payload for a member; empty dict on failure."""

    url = f"https://www.quiverquant.com/get_politician_page_tab_data/{bioguide}"
    headers = {
        "Referer": build_referer(first, last, bioguide),
        "Authorization": f"Bearer {TOKEN}",
        "accept": "application/json",
    }

    cache_path = _cache_path_for(url, "json")
    if cache_path:
        cached = _read_cache_text(cache_path)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass

    # Use global rate limiter and backoff logic
    RATE_LIMITER.wait()
    response = _request_with_backoff(
        url,
        headers=headers,
        timeout=30,
        purpose=f"payload for {bioguide}"
    )
    if not response:
        return {}

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        print(f"WARN: invalid JSON from payload for {bioguide}: {exc}", file=sys.stderr)
        return {}

    if cache_path and payload:
        try:
            _write_cache_text(cache_path, json.dumps(payload))
        except Exception:
            pass

    return payload


def extract_top_sectors_from_html(html: str, limit: int = TOP_SECTOR_LIMIT) -> str:
    """Parse the Top Traded Sectors list from the raw HTML snippet."""

    match = SECTOR_PATTERN.search(html)
    if not match:
        return ""

    raw_map = match.group(1)
    try:
        data = ast.literal_eval(raw_map)
    except (SyntaxError, ValueError):
        return ""

    if not isinstance(data, dict):
        return ""

    items = [
        (str(name), float(value) if isinstance(value, (int, float)) else 0.0)
        for name, value in data.items()
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)

    formatted = []
    for name, value in items[:max(1, limit)]:
        if value.is_integer():
            display_val = int(value)
        else:
            display_val = round(value, 2)
        formatted.append(f"{name}: {display_val}")

    return "; ".join(formatted)


def fetch_top_traded_sectors(bioguide: str, first: str, last: str) -> str:
    """Retrieve and format the Top Traded Sectors snippet for a member."""

    url = build_profile_url(first, last, bioguide)
    cache_path = _cache_path_for(url, "html")
    html: Optional[str] = None
    if cache_path:
        html = _read_cache_text(cache_path)

    if html is None:
        # Use global rate limiter and backoff logic
        RATE_LIMITER.wait()
        response = _request_with_backoff(
            url,
            timeout=30,
            purpose=f"sectors for {bioguide}"
        )
        if not response:
            return ""

        html = response.text
        if cache_path:
            _write_cache_text(cache_path, html)

    return extract_top_sectors_from_html(html)


def _normalize_money_display(value: Optional[str | float | int]) -> str:
    """Normalize money-like text into a dollar string like "$8,200,000".

    Accepts forms like "8.2M", "$8.2m", "1.5B", "250k", numeric strings, or numbers.
    Returns an empty string when input is blank or not parseable.
    """

    amount: Optional[float] = None
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        amount = float(value)
    else:
        s = str(value).strip()
        if not s:
            return ""
        lowered = s.lower().replace(",", "").replace("$", "").strip()
        if lowered in {"n/a", "na", "none", "unknown", "-"}:
            return ""

        match = re.match(
            r"^([+-]?[0-9]*\.?[0-9]+)\s*([kmmbn]|mm|mn|million|billion|thousand)?$",
            lowered,
        )
        if match:
            num = float(match.group(1))
            unit = (match.group(2) or "").lower()
            mult = 1.0
            if unit in {"k", "thousand"}:
                mult = 1e3
            elif unit in {"m", "mm", "mn", "million"}:
                mult = 1e6
            elif unit in {"b", "bn", "billion"}:
                mult = 1e9
            amount = num * mult
        else:
            try:
                amount = float(lowered)
            except ValueError:
                amount = None

    if amount is None or not math.isfinite(amount):
        return ""

    rounded = round(amount / 1000.0) * 1000.0  # nearest thousand dollars
    return f"${rounded:,.0f}"


def extract_networth_and_holdings(data: Dict) -> Tuple[str, str, str]:
    """Return (net_worth_estimate, normalized_net_worth, top_holdings).

    - net_worth_estimate: raw numeric string from Quiver (when available)
    - normalized_net_worth: human-friendly, normalized like "$8,200,000"
    - top_holdings: semicolon-separated list of top holdings with dollar amounts
    """

    holdings_data = data.get("holdings_data") or {}
    net_worth_estimate = ""

    try:
        live_vals = json.loads(holdings_data.get("politician_net_worth_live") or "[]")
        hist_vals = json.loads(holdings_data.get("politician_net_worth_data") or "[]")
    except json.JSONDecodeError:
        live_vals, hist_vals = [], []

    if live_vals:
        net_worth_estimate = f"{live_vals[0]:.2f}"
    elif hist_vals:
        latest = hist_vals[0][0] if isinstance(hist_vals[0], (list, tuple)) else hist_vals[0]
        try:
            net_worth_estimate = f"{float(latest):.2f}"
        except (TypeError, ValueError):
            net_worth_estimate = ""

    normalized_net_worth = _normalize_money_display(net_worth_estimate)

    holdings_amounts = holdings_data.get("holdings_amounts") or {}
    if isinstance(holdings_amounts, dict):
        top_items = sorted(
            holdings_amounts.items(),
            key=lambda kv: (kv[1] if kv[1] is not None else 0.0),
            reverse=True,
        )[:5]
        formatted = []
        for name, value in top_items:
            if value is None:
                continue
            formatted.append(f"{name}: ${value:,.0f}")
        top_holdings = "; ".join(formatted)
    else:
        top_holdings = ""

    return net_worth_estimate, normalized_net_worth, top_holdings


def process_member(base_row: Dict[str, str]) -> Dict[str, str]:
    """Process a single member: fetch data and return enriched row dict."""

    bioguide = (base_row.get("bioguide_id") or "").strip()
    first = (base_row.get("first_name") or "").strip()
    last = (base_row.get("last_name") or "").strip()
    middle = (base_row.get("middle_name") or "").strip()

    result = {
        "bioguide_id": bioguide,
        "first_name": first,
        "last_name": last,
        "middle_name": middle,
        "net_worth_estimate": "",
        "normalized_net_worth": "",
        "top_holdings": "",
        "top_traded_sectors": "",
    }

    if not bioguide:
        return result

    print(f"Processing {bioguide} ({first} {last})...", file=sys.stderr)

    networth = ""
    normalized_networth = ""
    holdings = ""
    sectors = ""

    try:
        payload = fetch_quiver_payload(bioguide, first, last)
        if payload:
            networth, normalized_networth, holdings = extract_networth_and_holdings(payload)

        # Add delay between API calls for same member to avoid burst requests
        time.sleep(0.5)

        sectors = fetch_top_traded_sectors(bioguide, first, last)
    except Exception as exc:
        print(f"Error processing {bioguide} ({first} {last}): {exc}", file=sys.stderr)

    result.update(
        {
            "net_worth_estimate": networth,
            "normalized_net_worth": normalized_networth,
            "top_holdings": holdings,
            "top_traded_sectors": sectors,
        }
    )

    return result


def ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_output_rows(rows: List[Dict[str, str]], path: Path) -> None:
    """Write the collected rows to disk, ensuring the parent directory exists."""

    ensure_output_dir(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Quiver net-worth, holdings, and sectors for politicians.")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache fetched responses (set '' to disable)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows whose bioguide_id already exists in output")
    parser.add_argument(
        "--test",
        nargs="?",
        const=1,
        type=int,
        metavar="N",
        help="Process only the first N politicians for testing (default 1)",
    )
    args = parser.parse_args()

    # Load token after argparse so --help works without credentials
    global TOKEN
    TOKEN = load_token()

    print("Script starting...")

    global CACHE_DIR
    CACHE_DIR = args.cache_dir or None
    if CACHE_DIR and not os.path.isabs(CACHE_DIR):
        CACHE_DIR = str((PROJECT_ROOT / CACHE_DIR).resolve())

    if not INPUT_CSV.exists():
        print(f"ERROR: missing input CSV at {INPUT_CSV}")
        return 1

    member_rows = load_member_base_rows(INPUT_CSV)
    if not member_rows:
        print(f"ERROR: no members found in {INPUT_CSV}")
        return 1

    print(f"Copied {len(member_rows)} member identities from {INPUT_CSV}")

    rows_index = {row["bioguide_id"]: row for row in member_rows if row.get("bioguide_id")}

    existing_done = hydrate_existing_rows(OUTPUT_CSV, rows_index)
    done = existing_done if args.resume else set()

    to_process = [row for row in member_rows if row.get("bioguide_id") and row["bioguide_id"] not in done]
    test_limit = 0
    if args.test is not None:
        test_limit = max(0, args.test)
    if test_limit:
        to_process = to_process[:test_limit]

    num_to_process = len(to_process)
    print(f"Starting to fetch data for {num_to_process} politicians...")

    if not args.resume or not OUTPUT_CSV.exists():
        write_output_rows(member_rows, OUTPUT_CSV)

    processed = 0
    if num_to_process:
        # Reduced from 5 to 2 workers to avoid overwhelming rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_member, dict(row)): row["bioguide_id"] for row in to_process}
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                bioguide = (row.get("bioguide_id") or "").strip()
                if bioguide:
                    target = rows_index.get(bioguide)
                    if not target:
                        target = {key: "" for key in OUTPUT_FIELDNAMES}
                        target["bioguide_id"] = bioguide
                        member_rows.append(target)
                        rows_index[bioguide] = target
                    target.update(row)
                processed += 1
                if processed % 10 == 0 or processed == num_to_process:
                    print(f"Processed {processed}/{num_to_process} politicians")
                    # Save incrementally every 10 records for better resume capability
                    write_output_rows(member_rows, OUTPUT_CSV)

    write_output_rows(member_rows, OUTPUT_CSV)
    print(f"Wrote {len(member_rows)} rows to {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
