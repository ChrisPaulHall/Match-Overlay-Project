#!/usr/bin/env python3
"""Fetch Quiver politician profile data via HTML + JSON API.

This script reads `01_data/members_face_lookup.csv`, de-duplicates entries by
BioGuide ID, fetches politician data from Quiver, and extracts:
- Net worth estimate (from JSON API)
- Top holdings by asset type (from JSON API)
- Top traded sectors (from HTML page)
- Strategy returns vs SPY (from strategy page)

The output CSV is stored at `01_data/00members_holdings_and_sectors.csv`.

Approach:
- Sequential processing for simplicity and rate limit compliance
- JSON API for net worth + holdings (single call)
- HTML page for sectors (single call)
- Strategy page for portfolio returns vs SPY
- Append mode for resume capability
- Optional API token for authenticated requests
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Logging setup
LOG_FORMAT = "%(asctime)s %(levelname)s [QUIVER]: %(message)s"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "01_data" / "members_face_lookup.csv"
OUTPUT_CSV = PROJECT_ROOT / "01_data" / "00members_holdings_and_sectors.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache_quiver"

# API token settings (optional - works without token but may have lower rate limits)
ENV_TOKEN_KEY = "QUIVER_API_TOKEN"
DEFAULT_TOKEN_FILE = "quiver_token.txt"

# Rate limiting - per-worker delay (not global) for true concurrency
DEFAULT_DELAY_SEC = 0.5  # Delay per worker after each politician (was 1.5 global)
DEFAULT_TIMEOUT = 15
DEFAULT_CONCURRENCY = 6

# Regex for sectorData in JavaScript
SECTOR_PATTERN = re.compile(r"let\s+sectorData\s*=\s*(\{[^}]+\});", re.DOTALL)

# Browser-like headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_FIELDNAMES = [
    "bioguide_id",
    "first_name",
    "last_name",
    "middle_name",
    "net_worth_estimate",
    "normalized_net_worth",
    "top_holdings",
    "top_traded_sectors",
    "strategy_return",
    "spy_return",
    "strategy_start_date",
]


def load_token() -> Optional[str]:
    """Resolve the API token from env var or a local file. Returns None if not found."""
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

    return None


def build_session(
    timeout: int = DEFAULT_TIMEOUT,
    pool_maxsize: int = 10,
) -> requests.Session:
    """Build a session with automatic retry handling."""
    sess = requests.Session()
    sess.headers.update(DEFAULT_HEADERS)

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
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=pool_maxsize,
        pool_maxsize=pool_maxsize,
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    # Default timeout wrapper
    original_get = sess.get
    def get_with_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_get(*args, **kwargs)
    sess.get = get_with_timeout  # type: ignore

    return sess


# ---------- Caching ----------
def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def cache_path_for(url: str, cache_dir: Path) -> Path:
    parsed = urlparse(url)
    base = parsed.path.rstrip("/").split("/")[-1] or "index"
    return cache_dir / f"{base}-{_short_hash(url)}.html"


def get_html(
    url: str,
    session: requests.Session,
    cache_dir: Optional[Path],
) -> Optional[str]:
    """Fetch HTML, using cache if available."""
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_path_for(url, cache_dir)
        if cpath.exists():
            return cpath.read_text(encoding="utf-8", errors="ignore")

    try:
        resp = session.get(url)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "30")
            wait_time = min(float(retry_after), 60.0)
            logging.warning("Rate limited, waiting %.1fs...", wait_time)
            time.sleep(wait_time)
            resp = session.get(url)

        if resp.status_code >= 400:
            logging.warning("HTTP %d for %s", resp.status_code, url)
            return None

        html = resp.text

        if cache_dir:
            cpath = cache_path_for(url, cache_dir)
            cpath.write_text(html, encoding="utf-8")

        return html

    except requests.RequestException as exc:
        logging.warning("Request failed for %s: %s", url, exc)
        return None


# ---------- URL building ----------
def build_profile_url(first: str, last: str, bioguide: str) -> str:
    """Build the Quiver politician profile URL."""
    full_name = f"{first} {last}".strip() or bioguide
    safe_name = quote(full_name.replace(" ", " "))
    return f"https://www.quiverquant.com/congresstrading/politician/{safe_name}-{bioguide}"


# ---------- Money normalization ----------
def normalize_money_display(value: Optional[str | float | int]) -> str:
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


# ---------- Parsing functions ----------
def parse_sector_data(html: str, limit: int = 5) -> str:
    """Extract top traded sectors from sectorData JavaScript variable."""
    match = SECTOR_PATTERN.search(html)
    if not match:
        return ""

    raw_map = match.group(1)
    try:
        # Parse JavaScript object literal (simple case)
        # Convert to valid Python dict syntax
        cleaned = raw_map.replace("'", '"')
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Fallback: try ast.literal_eval
        try:
            import ast
            data = ast.literal_eval(raw_map)
        except (SyntaxError, ValueError):
            return ""

    if not isinstance(data, dict):
        return ""

    # Sort by value descending
    items = [
        (str(name), float(value) if isinstance(value, (int, float)) else 0.0)
        for name, value in data.items()
    ]
    items.sort(key=lambda kv: kv[1], reverse=True)

    formatted = []
    for name, value in items[:limit]:
        if value.is_integer():
            display_val = int(value)
        else:
            display_val = round(value, 2)
        formatted.append(f"{name}: {display_val}")

    return "; ".join(formatted)


def parse_strategy_returns_from_graphdata(html: str) -> Dict[str, str]:
    """Extract strategy return data by parsing the graphDataStrategy and graphDataSPY arrays.

    The strategy page contains two JavaScript arrays:
    - graphDataStrategy: [{date, close}, ...] - normalized strategy portfolio values starting at 100M
    - graphDataSPY: [{date, close}, ...] - normalized SPY values starting at 100M

    Returns are calculated as: (last_value - 100000000) / 100000000 * 100
    """
    result = {
        "strategy_return": "",
        "spy_return": "",
        "strategy_start_date": "",
    }

    # Extract graphDataStrategy array
    strategy_match = re.search(
        r'graphDataStrategy\s*=\s*\[(.*?)\];',
        html,
        re.DOTALL
    )

    if strategy_match:
        strategy_data_str = strategy_match.group(1)
        # Extract all {date, close} objects
        entries = re.findall(
            r'\{\s*"date"\s*:\s*"([^"]+)"\s*,\s*"close"\s*:\s*([0-9.]+)\s*\}',
            strategy_data_str
        )

        if entries:
            # First entry gives start date
            first_date = entries[0][0]
            result["strategy_start_date"] = first_date

            # Last entry gives final value
            last_value = float(entries[-1][1])
            start_value = 100000000.0

            # Calculate return percentage
            return_pct = (last_value - start_value) / start_value * 100
            result["strategy_return"] = f"{return_pct:.2f}%"

    # Extract graphDataSPY array
    spy_match = re.search(
        r'graphDataSPY\s*=\s*\[(.*?)\];',
        html,
        re.DOTALL
    )

    if spy_match:
        spy_data_str = spy_match.group(1)
        # Extract all {date, close} objects
        entries = re.findall(
            r'\{\s*"date"\s*:\s*"([^"]+)"\s*,\s*"close"\s*:\s*([0-9.]+)\s*\}',
            spy_data_str
        )

        if entries:
            # Last entry gives final value
            last_value = float(entries[-1][1])
            start_value = 100000000.0

            # Calculate return percentage
            return_pct = (last_value - start_value) / start_value * 100
            result["spy_return"] = f"{return_pct:.2f}%"

    return result


def fetch_strategy_page(
    first: str,
    last: str,
    session: requests.Session,
    cache_dir: Optional[Path],
) -> Optional[str]:
    """Fetch strategy page HTML which contains graphDataStrategy and graphDataSPY arrays."""
    full_name = f"{first} {last}".strip()
    safe_name = quote(full_name.replace(" ", " "))
    url = f"https://www.quiverquant.com/strategies/s/{safe_name}/"

    # Check cache first
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath = cache_path_for(url, cache_dir)
        if cpath.exists():
            return cpath.read_text(encoding="utf-8", errors="ignore")

    try:
        resp = session.get(url)
        if resp.status_code == 200:
            html = resp.text
            # Cache if successful
            if cache_dir:
                cpath = cache_path_for(url, cache_dir)
                cpath.write_text(html, encoding="utf-8")
            return html
    except requests.RequestException:
        pass  # Strategy page may not exist for all politicians

    return None


# ---------- JSON API functions ----------
def fetch_json_data(
    bioguide: str,
    first: str,
    last: str,
    session: requests.Session,
    token: Optional[str] = None,
) -> Optional[dict]:
    """Fetch politician data from JSON API endpoint."""
    url = f"https://www.quiverquant.com/get_politician_page_tab_data/{bioguide}"
    headers = {
        "Referer": build_profile_url(first, last, bioguide) + "/net-worth",
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }

    # Add auth token if available
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = session.get(url, headers=headers)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "30")
            wait_time = min(float(retry_after), 60.0)
            logging.warning("Rate limited on JSON API, waiting %.1fs...", wait_time)
            time.sleep(wait_time)
            resp = session.get(url, headers=headers)

        if resp.status_code >= 400:
            logging.debug("JSON API returned %d for %s", resp.status_code, bioguide)
            return None

        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        logging.debug("JSON API error for %s: %s", bioguide, exc)
        return None


def parse_json_data(data: dict) -> Dict[str, str]:
    """Extract net worth and holdings from JSON API response."""
    result = {
        "net_worth_estimate": "",
        "normalized_net_worth": "",
        "top_holdings": "",
    }

    holdings_data = data.get("holdings_data") or {}

    # Net worth - prefer live value
    # Note: values may be JSON strings that need parsing
    try:
        live_vals = holdings_data.get("politician_net_worth_live") or []
        hist_vals = holdings_data.get("politician_net_worth_data") or []

        # Parse if string
        if isinstance(live_vals, str):
            live_vals = json.loads(live_vals)
        if isinstance(hist_vals, str):
            hist_vals = json.loads(hist_vals)

        if live_vals and isinstance(live_vals, list) and live_vals[0]:
            net_worth = float(live_vals[0])
            result["net_worth_estimate"] = f"{net_worth:.2f}"
            result["normalized_net_worth"] = normalize_money_display(net_worth)
        elif hist_vals and isinstance(hist_vals, list) and hist_vals[0]:
            # hist_vals is [[value, year], ...]
            latest = hist_vals[0][0] if isinstance(hist_vals[0], list) else hist_vals[0]
            net_worth = float(latest)
            result["net_worth_estimate"] = f"{net_worth:.2f}"
            result["normalized_net_worth"] = normalize_money_display(net_worth)
    except (TypeError, ValueError, IndexError, json.JSONDecodeError):
        pass

    # Holdings by asset type
    holdings_amounts = holdings_data.get("holdings_amounts") or {}
    # Parse if string
    if isinstance(holdings_amounts, str):
        try:
            holdings_amounts = json.loads(holdings_amounts)
        except json.JSONDecodeError:
            holdings_amounts = {}

    if isinstance(holdings_amounts, dict) and holdings_amounts:
        # Sort by value descending, take top 5
        items = sorted(
            holdings_amounts.items(),
            key=lambda kv: kv[1] if kv[1] is not None else 0,
            reverse=True,
        )[:5]

        formatted = []
        for name, value in items:
            if value is None:
                continue
            # Format as millions if large
            if value >= 1_000_000:
                formatted.append(f"{name}: ${value/1_000_000:.1f}M")
            elif value >= 1_000:
                formatted.append(f"{name}: ${value/1_000:.1f}K")
            else:
                formatted.append(f"{name}: ${value:,.0f}")

        result["top_holdings"] = "; ".join(formatted)

    return result


# ---------- Main scraping function ----------
def scrape_politician(
    bioguide: str,
    first: str,
    last: str,
    session: requests.Session,
    cache_dir: Optional[Path],
    include_strategy: bool,
    token: Optional[str] = None,
) -> Dict[str, str]:
    """Scrape all data for a single politician."""
    result = {
        "net_worth_estimate": "",
        "normalized_net_worth": "",
        "top_holdings": "",
        "top_traded_sectors": "",
        "strategy_return": "",
        "spy_return": "",
        "strategy_start_date": "",
    }

    # 1. Fetch JSON API for net worth + holdings
    json_data = fetch_json_data(bioguide, first, last, session, token)
    if json_data:
        result.update(parse_json_data(json_data))

    # 2. Fetch HTML page for sectors
    url = build_profile_url(first, last, bioguide)
    html = get_html(url, session, cache_dir)
    if html:
        result["top_traded_sectors"] = parse_sector_data(html)

    # 3. Fetch strategy page for returns (contains graphDataStrategy and graphDataSPY)
    if include_strategy:
        strategy_html = fetch_strategy_page(first, last, session, cache_dir)
        if strategy_html:
            strategy_data = parse_strategy_returns_from_graphdata(strategy_html)
            if strategy_data.get("strategy_return"):
                result.update(strategy_data)

    return result


# ---------- Data loading ----------
def load_members(path: Path) -> List[Dict[str, str]]:
    """Load member records from input CSV."""
    members = []
    seen = set()

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bioguide = (row.get("bioguide_id") or "").strip()
            if not bioguide or bioguide in seen:
                continue
            seen.add(bioguide)

            first = (
                row.get("first_name")
                or row.get("first")
                or row.get("given_name")
                or ""
            ).strip()
            last = (
                row.get("last_name")
                or row.get("last")
                or row.get("family_name")
                or ""
            ).strip()
            middle = (row.get("middle_name") or row.get("middle") or "").strip()

            members.append({
                "bioguide_id": bioguide,
                "first_name": first,
                "last_name": last,
                "middle_name": middle,
            })

    return members


def load_done_ids(path: Path) -> Set[str]:
    """Load bioguide IDs that already have data in output CSV."""
    done = set()
    if not path.exists():
        return done

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bioguide = (row.get("bioguide_id") or "").strip()
            # Only mark as done if we have actual data
            has_data = any([
                (row.get("net_worth_estimate") or "").strip(),
                (row.get("top_traded_sectors") or "").strip(),
                (row.get("strategy_return") or "").strip(),
            ])
            if bioguide and has_data:
                done.add(bioguide)

    return done


# ---------- Main ----------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape Quiver politician profiles for net worth, holdings, sectors, and strategy returns."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=f"Per-worker delay after each politician in seconds (default: {DEFAULT_DELAY_SEC})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent workers (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--skip-strategy",
        action="store_true",
        help="Skip fetching strategy performance page to reduce requests",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache fetched HTML (set '' to disable)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows that already have data in output",
    )
    parser.add_argument(
        "--test",
        nargs="?",
        const=5,
        type=int,
        metavar="N",
        help="Process only first N politicians for testing (default: 5)",
    )
    parser.add_argument(
        "--bioguide",
        nargs="+",
        metavar="ID",
        help="Process only specific bioguide IDs",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    # Load optional API token
    token = load_token()
    if token:
        logging.info("Using API token for authenticated requests")
    else:
        logging.info("No API token found; using unauthenticated requests")

    # Validate input
    if not INPUT_CSV.exists():
        logging.error("Input CSV not found: %s", INPUT_CSV)
        return 1

    # Load members
    members = load_members(INPUT_CSV)
    if not members:
        logging.error("No members found in %s", INPUT_CSV)
        return 1
    logging.info("Loaded %d members from %s", len(members), INPUT_CSV)

    # Filter by bioguide if specified
    if args.bioguide:
        target_ids = set(args.bioguide)
        members = [m for m in members if m["bioguide_id"] in target_ids]
        logging.info("Filtered to %d members by bioguide ID", len(members))

    # Resume mode
    done_ids = load_done_ids(OUTPUT_CSV) if args.resume else set()
    if args.resume:
        logging.info("Resume mode: %d members already have data", len(done_ids))

    # Filter out already-done members (unless specific bioguides requested)
    if args.resume and not args.bioguide:
        members = [m for m in members if m["bioguide_id"] not in done_ids]

    # Test mode
    if args.test:
        members = members[:args.test]
        logging.info("Test mode: processing %d members", len(members))

    if not members:
        logging.info("No members to process")
        return 0

    logging.info("Will process %d members", len(members))
    logging.info(
        "Delay: %.2fs, Timeout: %ds, Concurrency: %d, Strategy: %s",
        args.delay,
        args.timeout,
        args.concurrency,
        "skip" if args.skip_strategy else "include",
    )

    # Setup
    pool_size = max(args.concurrency * 2, 10)
    thread_local = threading.local()

    def get_session() -> requests.Session:
        """Thread-local session to avoid cross-thread sharing."""
        sess = getattr(thread_local, "session", None)
        if sess is None:
            sess = build_session(
                timeout=args.timeout,
                pool_maxsize=pool_size,
            )
            thread_local.session = sess
        return sess

    cache_dir = args.cache_dir if args.cache_dir else None

    # Prepare output
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Determine write mode
    write_header = not OUTPUT_CSV.exists() or not args.resume
    mode = "w" if write_header else "a"

    stats = {"processed": 0, "success": 0, "empty": 0}

    indexed_members = list(enumerate(members, 1))

    def process_member(item: tuple[int, Dict[str, str]]) -> Dict[str, str]:
        idx, member = item
        bioguide = member["bioguide_id"]
        first = member["first_name"]
        last = member["last_name"]
        middle = member["middle_name"]

        logging.info(
            "[%d/%d] Scraping: %s %s (%s)",
            idx,
            len(indexed_members),
            first,
            last,
            bioguide,
        )

        session = get_session()
        data = scrape_politician(
            bioguide,
            first,
            last,
            session,
            cache_dir,
            include_strategy=not args.skip_strategy,
            token=token,
        )

        # Per-worker delay: sleep after each politician to be respectful
        # With 6 workers, this allows ~6 politicians processed per delay period
        if args.delay > 0:
            time.sleep(args.delay + random.uniform(0, args.delay * 0.2))

        row = {
            "bioguide_id": bioguide,
            "first_name": first,
            "last_name": last,
            "middle_name": middle,
            **data,
        }
        return row

    with OUTPUT_CSV.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
        if write_header:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for row in executor.map(process_member, indexed_members):
                writer.writerow(row)
                stats["processed"] += 1
                if (
                    row.get("net_worth_estimate")
                    or row.get("top_traded_sectors")
                    or row.get("strategy_return")
                ):
                    stats["success"] += 1
                else:
                    stats["empty"] += 1

    # Summary
    logging.info("=" * 50)
    logging.info("SUMMARY")
    logging.info("  Processed: %d", stats["processed"])
    logging.info("  With data: %d", stats["success"])
    logging.info("  Empty/errors: %d", stats["empty"])
    logging.info("Output: %s", OUTPUT_CSV)
    logging.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
