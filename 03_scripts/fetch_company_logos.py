#!/usr/bin/env python3
"""Fetch and cache company logos for stock tickers traded by Congress members.

This script reads unique tickers from `01_data/members_trades_clean.csv`,
resolves company domains, fetches logos from Logo.dev, and caches them locally.

Output:
    01_data/logos/{TICKER}.png - Cached logo images
    01_data/ticker_logos.csv   - Index mapping tickers to logo paths

Usage (from the project root):
    python 03_scripts/fetch_company_logos.py

Optional flags:
    --test N         Process only first N tickers for testing
    --ticker SYMBOL  Process only specific ticker(s)
    --force          Re-fetch logos even if cached
    --dry-run        Show what would be fetched without downloading
    --verbose        Verbose logging
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

# Logging setup
LOG_FORMAT = "%(asctime)s %(levelname)s [LOGOS]: %(message)s"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADES_CSV = PROJECT_ROOT / "01_data" / "members_trades_clean.csv"
LOGOS_DIR = PROJECT_ROOT / "01_data" / "logos"
OUTPUT_CSV = PROJECT_ROOT / "01_data" / "ticker_logos.csv"

# Logo.dev API configuration
LOGO_DEV_TOKEN = os.getenv("LOGO_DEV_TOKEN", "pk_N6crV2sNRkCV1ubmzTyS5A")
LOGO_DEV_BASE_URL = "https://img.logo.dev"

# Rate limiting
DEFAULT_DELAY_SEC = 0.5  # Be respectful to the API
DEFAULT_TIMEOUT = 15

# Common ticker -> domain mappings for well-known companies
# This handles cases where company name doesn't map cleanly to domain
TICKER_DOMAIN_OVERRIDES: Dict[str, str] = {
    # Big Tech
    "AAPL": "apple.com",
    "GOOGL": "google.com",
    "GOOG": "google.com",
    "MSFT": "microsoft.com",
    "AMZN": "amazon.com",
    "META": "meta.com",
    "NVDA": "nvidia.com",
    "TSLA": "tesla.com",
    "NFLX": "netflix.com",
    "INTC": "intel.com",
    "AMD": "amd.com",
    "CRM": "salesforce.com",
    "ORCL": "oracle.com",
    "IBM": "ibm.com",
    "CSCO": "cisco.com",
    "ADBE": "adobe.com",
    "PYPL": "paypal.com",
    "SQ": "squareup.com",
    "SHOP": "shopify.com",
    "UBER": "uber.com",
    "LYFT": "lyft.com",
    "ABNB": "airbnb.com",
    "SNAP": "snap.com",
    "TWTR": "twitter.com",
    "X": "x.com",
    "PINS": "pinterest.com",
    "SPOT": "spotify.com",
    "ZM": "zoom.us",
    "DOCU": "docusign.com",
    "NET": "cloudflare.com",
    "SNOW": "snowflake.com",
    "PLTR": "palantir.com",
    "COIN": "coinbase.com",
    "RBLX": "roblox.com",
    "U": "unity.com",
    "EA": "ea.com",
    "ATVI": "activision.com",
    "TTWO": "take2games.com",

    # Finance
    "JPM": "jpmorganchase.com",
    "BAC": "bankofamerica.com",
    "WFC": "wellsfargo.com",
    "C": "citigroup.com",
    "GS": "goldmansachs.com",
    "MS": "morganstanley.com",
    "BLK": "blackrock.com",
    "SCHW": "schwab.com",
    "V": "visa.com",
    "MA": "mastercard.com",
    "AXP": "americanexpress.com",
    "DFS": "discover.com",

    # Healthcare
    "JNJ": "jnj.com",
    "PFE": "pfizer.com",
    "UNH": "unitedhealthgroup.com",
    "MRK": "merck.com",
    "ABBV": "abbvie.com",
    "LLY": "lilly.com",
    "TMO": "thermofisher.com",
    "ABT": "abbott.com",
    "BMY": "bms.com",
    "AMGN": "amgen.com",
    "GILD": "gilead.com",
    "MRNA": "modernatx.com",
    "BNTX": "biontech.de",
    "CVS": "cvs.com",
    "WBA": "walgreens.com",

    # Retail
    "WMT": "walmart.com",
    "TGT": "target.com",
    "COST": "costco.com",
    "HD": "homedepot.com",
    "LOW": "lowes.com",
    "NKE": "nike.com",
    "SBUX": "starbucks.com",
    "MCD": "mcdonalds.com",
    "KO": "coca-cola.com",
    "PEP": "pepsico.com",
    "PG": "pg.com",
    "CL": "colgatepalmolive.com",
    "KMB": "kimberly-clark.com",

    # Industrial / Energy
    "XOM": "exxonmobil.com",
    "CVX": "chevron.com",
    "COP": "conocophillips.com",
    "SLB": "slb.com",
    "HAL": "halliburton.com",
    "BA": "boeing.com",
    "LMT": "lockheedmartin.com",
    "RTX": "rtx.com",
    "NOC": "northropgrumman.com",
    "GD": "gd.com",
    "GE": "ge.com",
    "HON": "honeywell.com",
    "MMM": "3m.com",
    "CAT": "caterpillar.com",
    "DE": "deere.com",
    "F": "ford.com",
    "GM": "gm.com",
    "TM": "toyota.com",

    # Telecom / Media
    "T": "att.com",
    "VZ": "verizon.com",
    "TMUS": "t-mobile.com",
    "CMCSA": "comcast.com",
    "DIS": "disney.com",
    "WBD": "wbd.com",
    "PARA": "paramount.com",
    "FOX": "fox.com",
    "FOXA": "fox.com",
    "NWSA": "newscorp.com",
    "NWS": "newscorp.com",

    # ETFs and Indices
    "SPY": "ssga.com",
    "QQQ": "invesco.com",
    "IWM": "ishares.com",
    "DIA": "ssga.com",
    "VTI": "vanguard.com",
    "VOO": "vanguard.com",
    "VEA": "vanguard.com",
    "VWO": "vanguard.com",
    "BND": "vanguard.com",
    "AGG": "ishares.com",
    "GLD": "spdrgoldshares.com",
    "SLV": "ishares.com",
    "USO": "uscfinvestments.com",
    "XLF": "ssga.com",
    "XLK": "ssga.com",
    "XLE": "ssga.com",
    "XLV": "ssga.com",
    "XLI": "ssga.com",
    "XLY": "ssga.com",
    "XLP": "ssga.com",
    "XLU": "ssga.com",
    "XLB": "ssga.com",
    "XLRE": "ssga.com",

    # Additional popular tickers
    "STLA": "stellantis.com",
    "STLD": "steeldynamics.com",
    "STM": "st.com",
    "STT": "statestreet.com",
    "STX": "seagate.com",
    "STZ": "cbrands.com",
    "SU": "suncor.com",
    "SWK": "stanleyblackanddecker.com",
    "SWKS": "skyworksinc.com",
    "SWN": "swn.com",
    "SYF": "synchrony.com",
    "SYK": "stryker.com",
    "SYY": "sysco.com",
    "TAP": "molsoncoors.com",
    "TDG": "transdigm.com",
    "TDY": "teledyne.com",
    "TEL": "te.com",
    "TER": "teradyne.com",
    "TFC": "truist.com",
    "TFX": "teleflex.com",
    "TJX": "tjx.com",
    "TMO": "thermofisher.com",
    "TMUS": "t-mobile.com",
    "TOL": "tollbrothers.com",
    "TROW": "troweprice.com",
    "TRV": "travelers.com",
    "TSM": "tsmc.com",
    "TSN": "tysonfoods.com",
    "TT": "tranetechnologies.com",
    "TTC": "thetorocompany.com",
    "TTD": "thetradedesk.com",
    "TTEK": "tetratech.com",
    "TXN": "ti.com",
    "TXRH": "texasroadhouse.com",
    "TXT": "textron.com",
    "TYL": "tylertech.com",
    "UAL": "united.com",
    "UDR": "udr.com",
    "UHS": "uhsinc.com",
    "ULTA": "ulta.com",
    "UNH": "unitedhealthgroup.com",
    "UNP": "up.com",
    "UPS": "ups.com",
    "URI": "unitedrentals.com",
    "USB": "usbank.com",
    "VALE": "vale.com",
    "VFC": "vfc.com",
    "VIG": "vanguard.com",
    "VLO": "valero.com",
    "VMC": "vulcanmaterials.com",
    "VMW": "vmware.com",
    "VNO": "vno.com",
    "VRSK": "verisk.com",
    "VRSN": "verisign.com",
    "VRTX": "vrtx.com",
    "VST": "vistraenergy.com",
    "VTR": "ventasreit.com",
    "VTRS": "viatris.com",
    "VZ": "verizon.com",
    "WAB": "wabteccorp.com",
    "WAT": "waters.com",
    "WBA": "walgreens.com",
    "WBD": "wbd.com",
    "WDC": "westerndigital.com",
    "WEC": "wecenergygroup.com",
    "WELL": "welltower.com",
    "WFC": "wellsfargo.com",
    "WHR": "whirlpool.com",
    "WM": "wm.com",
    "WMB": "williams.com",
    "WMT": "walmart.com",
    "WRB": "berkley.com",
    "WRK": "westrock.com",
    "WST": "westpharma.com",
    "WTW": "wtwco.com",
    "WY": "weyerhaeuser.com",
    "WYNN": "wynnresorts.com",
    "XEL": "xcelenergy.com",
    "XOM": "exxonmobil.com",
    "XRAY": "dentsply.com",
    "XYL": "xylem.com",
    "YUM": "yum.com",
    "ZBH": "zimmerbiomet.com",
    "ZBRA": "zebra.com",
    "ZION": "zionsbank.com",
    "ZTS": "zoetis.com",
}

# Minimum file size for a valid logo (in bytes) - used for domain fallback
# Placeholder images from Logo.dev are typically ~600-1700 bytes (just letters/numbers)
# Valid logos (even simple ones like Apple) are typically 2KB+
MIN_LOGO_SIZE_BYTES = 2000  # 2KB minimum for valid logos

# Known placeholder file sizes from Logo.dev ticker endpoint
# These are exact sizes of the generic letter images (A, B, C, etc.)
# Important: Valid logos like Microsoft are 616 bytes, so we only block confirmed placeholder sizes
PLACEHOLDER_SIZES = {1546, 1440, 1643, 1740, 1484, 1102}  # Known placeholder sizes

# Patterns to identify invalid "tickers" that aren't stocks
INVALID_TICKER_PATTERNS = [
    # Treasury bill/bond maturity descriptions
    re.compile(r"^\d+[\.\-]?(WEEK|MONTH|YEAR)", re.IGNORECASE),
    re.compile(r"^(WEEK|MONTH|YEAR)", re.IGNORECASE),
    re.compile(r"MATURE$", re.IGNORECASE),
    # CUSIP numbers (9 alphanumeric characters, often starting with digits)
    re.compile(r"^\d{6}[A-Z0-9]{3}$"),  # Standard CUSIP format
    re.compile(r"^9\d{8}$"),  # Treasury CUSIP starting with 9
    # Date patterns
    re.compile(r"^\d{2}/\d{2}/\d{2,4}$"),  # MM/DD/YY or MM/DD/YYYY
    # Pure numbers
    re.compile(r"^\d+$"),
    # Percentage values
    re.compile(r"^\d+%$"),
    # Very long codes (likely CUSIPs or other identifiers)
    re.compile(r"^[A-Z0-9]{7,}$"),  # 7+ alphanumeric chars without exchange suffix
    # Quoted strings (data parsing issues)
    re.compile(r'^"'),
]

# Valid ticker format: 1-6 uppercase letters, optionally with exchange suffix
# Examples: AAPL, MSFT, STLA, TSM, BRK.A, 0QZI.IL (foreign)
VALID_TICKER_PATTERN = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,3})?$")


def is_valid_stock_ticker(ticker: str) -> bool:
    """Check if ticker looks like a valid stock ticker.

    Filters out:
    - Treasury bills/bonds (e.g., "1.MONTH", "13.WEEK MATURE")
    - CUSIP numbers (e.g., "912796CR8", "36966R5P7")
    - Dates (e.g., "07/01/28")
    - Pure numbers (e.g., "2", "4491235")
    - Other non-stock identifiers

    Returns True for valid-looking tickers like AAPL, MSFT, STLA.
    """
    if not ticker:
        return False

    ticker = ticker.strip()

    # Check against invalid patterns
    for pattern in INVALID_TICKER_PATTERNS:
        if pattern.search(ticker):
            return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in ticker):
        return False

    # Split off exchange suffix if present
    base_ticker = ticker.split(".")[0]

    # Reject tickers starting with a digit (foreign exchange codes like 0QZI.IL, 1AMT.MI)
    # These rarely have logos in Logo.dev and often return placeholders
    if base_ticker and base_ticker[0].isdigit():
        return False

    # Base ticker should be 1-6 characters, mostly letters
    if len(base_ticker) > 6:
        return False

    # Count letters vs digits in base ticker
    letters = sum(1 for c in base_ticker if c.isalpha())
    digits = sum(1 for c in base_ticker if c.isdigit())

    # Should be mostly letters (allow some digits like in BRK.A)
    if digits > letters:
        return False

    return True


# Company name patterns to clean for domain guessing
# These get stripped from the END of the name repeatedly
COMPANY_SUFFIXES = re.compile(
    r"\s*,?\s*(Inc\.?|Corp\.?|Corporation|Company|Co\.?|LLC|Ltd\.?|Limited|PLC|"
    r"Holdings?|Group|International|Intl\.?|Technologies|Technology|Tech|"
    r"Enterprises?|Solutions|Services|Systems|Pharmaceuticals?|Therapeutics|"
    r"Bancorp|Financial|Capital|Partners|& Co\.?|Class [A-Z]|Common Stock|Cmn|"
    r"Ordinary Shares?|Common Shares?|Depositary Shares?|ADR|ADS|American Depositary.*|"
    r"N\.?V\.?|S\.?A\.?|AG|SE|SpA|GmbH|AB|Oyj|plc|New|Trust|REIT|ETF|Fund)$",
    re.IGNORECASE
)

# Words/phrases that should be stripped from ANYWHERE in company name
NOISE_WORDS = re.compile(
    r"\b(Common|Stock|Shares?|Depositary|American|Registered|Series [A-Z0-9]+|"
    r"Class [A-Z0-9]+|New|The|of|and|&)\b",
    re.IGNORECASE
)

OUTPUT_FIELDNAMES = [
    "ticker",
    "company",
    "domain",
    "logo_path",
    "logo_url",
    "fetched_date",
    "status",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and cache company logos for stock tickers."
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Process only first N tickers for testing",
    )
    parser.add_argument(
        "--ticker",
        nargs="+",
        metavar="SYMBOL",
        help="Process only specific ticker(s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch logos even if cached",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SEC,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY_SEC})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args(argv)


def load_tickers_from_trades(path: Path) -> List[Dict[str, str]]:
    """Load unique tickers and company names from trades CSV."""
    if not path.exists():
        return []

    tickers: Dict[str, Dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            if not ticker or ticker in tickers:
                continue

            # Get company name from various possible fields
            company = (
                row.get("company") or
                row.get("description") or
                row.get("asset") or
                ""
            ).strip()

            tickers[ticker] = {
                "ticker": ticker,
                "company": company,
            }

    # Sort by ticker
    return sorted(tickers.values(), key=lambda x: x["ticker"])


def load_existing_logos(path: Path) -> Dict[str, Dict[str, str]]:
    """Load existing logo index."""
    if not path.exists():
        return {}

    index = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            if ticker:
                index[ticker] = dict(row)

    return index


def clean_company_name(name: str) -> str:
    """Clean company name for domain guessing.

    Aggressively strips suffixes, noise words, and legal entity designators
    to get just the core brand/company name.
    """
    if not name:
        return ""

    cleaned = name

    # Repeatedly strip suffixes from the end until stable
    prev = ""
    while prev != cleaned:
        prev = cleaned
        cleaned = COMPANY_SUFFIXES.sub("", cleaned).strip()

    # Remove noise words from anywhere
    cleaned = NOISE_WORDS.sub(" ", cleaned)

    # Remove extra whitespace and punctuation
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # If the result is too short or too long, it's probably not useful
    if len(cleaned) < 2 or len(cleaned) > 30:
        # Try to extract just the first word or two
        words = name.split()[:2]
        cleaned = " ".join(w for w in words if len(w) > 1 and not w.upper() in ("THE", "A", "AN"))
        cleaned = re.sub(r"[^\w\s-]", "", cleaned).strip()

    return cleaned


def guess_domain(ticker: str, company: str) -> Optional[str]:
    """Guess company domain from ticker and company name."""
    # Check overrides first
    if ticker in TICKER_DOMAIN_OVERRIDES:
        return TICKER_DOMAIN_OVERRIDES[ticker]

    # Try to guess from company name
    if company:
        cleaned = clean_company_name(company)
        if cleaned:
            # Simple heuristic: lowercase, remove spaces, add .com
            domain_base = cleaned.lower().replace(" ", "").replace("-", "")
            if domain_base:
                return f"{domain_base}.com"

    # Fallback: try ticker itself as domain (works for some companies)
    return f"{ticker.lower()}.com"


def fetch_logo_by_ticker(
    ticker: str,
    output_path: Path,
    *,
    size: int = 128,
    format: str = "png",
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """Fetch logo from Logo.dev using ticker endpoint (preferred).

    Uses: https://img.logo.dev/ticker/{SYMBOL}

    Returns:
        (success, message) tuple
    """
    # Clean ticker - remove any exchange suffix for now, use base symbol
    clean_ticker = ticker.split(".")[0].upper()

    url = (
        f"{LOGO_DEV_BASE_URL}/ticker/{clean_ticker}"
        f"?token={LOGO_DEV_TOKEN}"
        f"&size={size}"
        f"&format={format}"
    )

    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            content_size = len(response.content)

            # Check if it's an image
            if "image" in content_type or content_size > 100:
                # Reject known placeholder sizes (generic letter images like "A", "B", etc.)
                if content_size in PLACEHOLDER_SIZES:
                    return False, f"Placeholder image ({content_size} bytes)"

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                return True, "OK (ticker)"
            else:
                return False, "No image returned"
        elif response.status_code == 404:
            return False, "Ticker not found"
        else:
            return False, f"HTTP {response.status_code}"

    except requests.Timeout:
        return False, "Timeout"
    except requests.RequestException as exc:
        return False, f"Request error: {exc}"


def fetch_logo_by_domain(
    domain: str,
    output_path: Path,
    *,
    size: int = 128,
    format: str = "png",
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, str]:
    """Fetch logo from Logo.dev using domain endpoint (fallback).

    Uses: https://img.logo.dev/{domain}

    Returns:
        (success, message) tuple
    """
    url = (
        f"{LOGO_DEV_BASE_URL}/{domain}"
        f"?token={LOGO_DEV_TOKEN}"
        f"&size={size}"
        f"&format={format}"
    )

    try:
        response = requests.get(url, timeout=timeout)

        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            content_size = len(response.content)

            if "image" in content_type or content_size > 100:
                # Reject small placeholder images
                if content_size < MIN_LOGO_SIZE_BYTES:
                    return False, f"Placeholder image ({content_size} bytes)"

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)
                return True, "OK (domain)"
            else:
                return False, "No image returned"
        elif response.status_code == 404:
            return False, "Domain not found"
        else:
            return False, f"HTTP {response.status_code}"

    except requests.Timeout:
        return False, "Timeout"
    except requests.RequestException as exc:
        return False, f"Request error: {exc}"


def fetch_logo(
    ticker: str,
    domain: Optional[str],
    output_path: Path,
    *,
    size: int = 128,
    format: str = "png",
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, str, str]:
    """Fetch logo from Logo.dev, trying ticker endpoint first, then domain.

    Returns:
        (success, message, method_used) tuple
    """
    # First try the ticker endpoint (works for most stock symbols)
    success, msg = fetch_logo_by_ticker(
        ticker, output_path, size=size, format=format, timeout=timeout
    )
    if success:
        return True, msg, f"ticker/{ticker}"

    # If ticker failed and we have a domain, try domain endpoint
    if domain:
        success, msg = fetch_logo_by_domain(
            domain, output_path, size=size, format=format, timeout=timeout
        )
        if success:
            return True, msg, domain

    return False, msg, ""


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    # Load tickers from trades
    tickers = load_tickers_from_trades(TRADES_CSV)
    if not tickers:
        logging.error("No tickers found in %s", TRADES_CSV)
        return 1

    logging.info("Found %d unique tickers in trades data", len(tickers))

    # Filter out invalid tickers (Treasury bills, CUSIPs, dates, etc.)
    original_count = len(tickers)
    tickers = [t for t in tickers if is_valid_stock_ticker(t["ticker"])]
    filtered_count = original_count - len(tickers)
    if filtered_count > 0:
        logging.info("Filtered out %d invalid entries (Treasury bills, CUSIPs, etc.)", filtered_count)
    logging.info("Processing %d valid stock tickers", len(tickers))

    # Filter by specific tickers if requested
    if args.ticker:
        target_tickers = {t.upper() for t in args.ticker}
        tickers = [t for t in tickers if t["ticker"] in target_tickers]
        logging.info("Filtered to %d tickers", len(tickers))

    # Test mode
    if args.test:
        tickers = tickers[:args.test]
        logging.info("Test mode: processing %d tickers", len(tickers))

    if not tickers:
        logging.info("No tickers to process")
        return 0

    # Load existing logo index
    existing = load_existing_logos(OUTPUT_CSV) if not args.force else {}
    if existing and not args.force:
        logging.info("Found %d existing logo entries", len(existing))

    # Setup output
    LOGOS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Process tickers
    results: List[Dict[str, str]] = []
    stats = {"fetched": 0, "cached": 0, "failed": 0, "skipped": 0}

    for idx, item in enumerate(tickers, 1):
        ticker = item["ticker"]
        company = item["company"]

        # Check if already processed
        if ticker in existing and not args.force:
            existing_entry = existing[ticker]
            logo_path = existing_entry.get("logo_path", "")
            if logo_path and Path(PROJECT_ROOT / logo_path).exists():
                logging.debug("[%d/%d] %s: Already cached", idx, len(tickers), ticker)
                results.append(existing_entry)
                stats["cached"] += 1
                continue

        # Guess domain
        domain = guess_domain(ticker, company)
        if not domain:
            logging.warning("[%d/%d] %s: Could not determine domain", idx, len(tickers), ticker)
            results.append({
                "ticker": ticker,
                "company": company,
                "domain": "",
                "logo_path": "",
                "logo_url": "",
                "fetched_date": datetime.now().isoformat(),
                "status": "No domain",
            })
            stats["failed"] += 1
            continue

        # Output path
        logo_filename = f"{ticker}.png"
        logo_path = LOGOS_DIR / logo_filename
        logo_path_relative = f"01_data/logos/{logo_filename}"

        if args.dry_run:
            logging.info("[%d/%d] %s: Would try ticker/%s, then %s", idx, len(tickers), ticker, ticker, domain)
            stats["skipped"] += 1
            continue

        logging.info("[%d/%d] %s: Fetching logo (ticker first, then %s)...", idx, len(tickers), ticker, domain)

        success, message, method_used = fetch_logo(ticker, domain, logo_path)

        if success:
            logo_url = f"{LOGO_DEV_BASE_URL}/{method_used}?token={LOGO_DEV_TOKEN}&size=128&format=png"
            logging.debug("  -> Saved to %s (via %s)", logo_path_relative, method_used)
            stats["fetched"] += 1
            status = message  # "OK (ticker)" or "OK (domain)"
        else:
            logo_url = ""
            logging.warning("  -> Failed: %s", message)
            stats["failed"] += 1
            status = message
            logo_path_relative = ""

        results.append({
            "ticker": ticker,
            "company": company,
            "domain": method_used if success else domain,
            "logo_path": logo_path_relative if success else "",
            "logo_url": logo_url,
            "fetched_date": datetime.now().isoformat(),
            "status": status,
        })

        # Rate limiting
        if idx < len(tickers):
            time.sleep(args.delay)

    # Write results
    if not args.dry_run and results:
        # Merge with existing entries that weren't reprocessed
        all_results = {r["ticker"]: r for r in results}
        for ticker, entry in existing.items():
            if ticker not in all_results:
                all_results[ticker] = entry

        # Sort and write
        sorted_results = sorted(all_results.values(), key=lambda x: x["ticker"])

        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
            writer.writeheader()
            for row in sorted_results:
                writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDNAMES})

        logging.info("Wrote %d entries to %s", len(sorted_results), OUTPUT_CSV)

    # Summary
    logging.info("=" * 50)
    logging.info("SUMMARY")
    logging.info("  Fetched: %d", stats["fetched"])
    logging.info("  Cached:  %d", stats["cached"])
    logging.info("  Failed:  %d", stats["failed"])
    if args.dry_run:
        logging.info("  Skipped: %d (dry-run)", stats["skipped"])
    logging.info("  Logos:   %s", LOGOS_DIR)
    logging.info("  Index:   %s", OUTPUT_CSV)
    logging.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
