#!/usr/bin/env python3
"""Append the latest Quiver live-congress trading data to the local snapshot.

This helper keeps ``01_data/members_trades_clean.csv`` growing beyond the
1,000-row window returned by QuiverQuant's ``/beta/live/congresstrading``
endpoint.  It fetches the current feed, normalizes the payload to match the
overlay's fallback schema, de-duplicates rows that are already present, and
rewrites the CSV sorted by trade date (most recent first).

Usage (from the project root):

    python 03_scripts/update_trades_snapshot.py

Optional flags::

    --data-file PATH     Override the CSV location (defaults to 01_data/...).
    --token-file PATH    Point to a specific quiver_token.txt.
    --dry-run            Skip writing changes; prints the summary instead.
    --verbose            Show debug logs while processing.

The script looks for ``QUIVER_API_TOKEN`` in the environment or, if unset,
falls back to resolving ``quiver_token.txt`` using the same search order as the
other Quiver helpers in this repo.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests

try:  # Local helper that already handles token lookup + error type
    from quiver_api import load_token, QuiverQuantError  # type: ignore
except Exception as exc:  # pragma: no cover - defensive fallback
    raise SystemExit(
        "Unable to import 03_scripts/quiver_api.py. "
        "Run this script from the repository root."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "01_data" / "members_trades_clean.csv"
LIVE_ENDPOINT = "https://api.quiverquant.com/beta/live/congresstrading"
FIELD_ORDER = [
    "BioGuideID",
    "first_name",
    "last_name",
    "representative",
    "house",
    "party",
    "transaction",
    "ticker",
    "ticker_type",
    "description",
    "range",
    "amount",
    "excess_return",
    "price_change",
    "spy_change",
    "report_date",
    "transaction_date",
    "last_modified",
]

NAME_INDEX_SOURCES = [
    PROJECT_ROOT / "01_data" / "members_face_lookup.csv",
    PROJECT_ROOT / "01_data" / "members_personal_profiles.csv",
]

DATE_KEYS = (
    "Traded",
    "Filed",
    "Date",
    "TransactionDate",
    "transaction_date",
)

AMOUNT_KEYS = (
    "Amount",
    "Range",
    "AmountRange",
    "Trade_Size_USD",
    "trade_size_usd",
)

REPRESENTATIVE_KEYS = (
    "Representative",
    "Senator",
    "Name",
    "Politician",
)

BIOGUIDE_KEYS = (
    "BioGuideID",
    "BioguideID",
    "Bioguide",
    "bioguide_id",
)


class FetchError(RuntimeError):
    """Raised when the live endpoint cannot be downloaded."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Quiver live congress trades into the local snapshot CSV."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to members_trades_clean.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Optional override for quiver_token.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but do not write any files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging for debugging.",
    )
    return parser.parse_args(argv)


def _slugify(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", text.lower())


def _name_key(first: str, last: str) -> str:
    return f"{_slugify(first)}::{_slugify(last)}"


def _split_name(full_name: str) -> Tuple[str, str]:
    if not full_name:
        return "", ""
    name = re.sub(r"\s+", " ", str(full_name)).strip()
    if not name:
        return "", ""
    if "," in name:
        last, first = [part.strip() for part in name.split(",", 1)]
        return first, last
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]


def _first_nonempty(entry: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = entry.get(key)
        if value:
            return str(value).strip()
    return ""


def build_name_index() -> Dict[str, str]:
    index: Dict[str, str] = {}
    for path in NAME_INDEX_SOURCES:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bioguide = (row.get("bioguide_id") or row.get("BioGuideID") or "").strip()
                if not bioguide:
                    continue

                first_candidates = [
                    row.get("first_name"),
                    row.get("first_name1"),
                    row.get("preferred_name"),
                    row.get("first"),
                    row.get("preferred_first_name"),
                ]
                last_candidates = [
                    row.get("last_name"),
                    row.get("last_name1"),
                    row.get("surname"),
                    row.get("last"),
                    row.get("family_name"),
                ]
                first = next((c for c in first_candidates if c and c.strip()), "")
                last = next((c for c in last_candidates if c and c.strip()), "")
                if not first or not last:
                    continue
                key = _name_key(first, last)
                index.setdefault(key, bioguide)
    return index


def fetch_live_trades(token: str) -> List[dict]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    try:
        response = requests.get(LIVE_ENDPOINT, headers=headers, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise FetchError(f"Request failed: {exc}") from exc

    if response.status_code // 100 != 2:
        raise FetchError(
            f"HTTP {response.status_code} from live endpoint: {response.text.strip()}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise FetchError("Live endpoint returned invalid JSON") from exc

    if not isinstance(payload, list):
        raise FetchError("Expected a JSON array from live endpoint")

    return payload


def load_existing_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        data = [dict(row) for row in reader]
    return data


def dedupe_key(row: dict) -> Tuple[str, str, str, str, str]:
    first = _slugify(row.get("first_name", ""))
    last = _slugify(row.get("last_name", ""))
    bioguide = (row.get("BioGuideID") or "").strip().upper()
    ticker = (row.get("ticker") or "").strip().upper()
    date = (row.get("transaction_date") or "").strip()
    txn = (row.get("transaction") or "").strip().lower()
    amount = (row.get("range") or "").strip()

    ident = bioguide if bioguide else f"{first}:{last}"
    return ident, date, ticker, txn, amount


def normalize_trade(entry: dict, *, name_index: Dict[str, str]) -> dict | None:
    full_name = _first_nonempty(entry, REPRESENTATIVE_KEYS)
    first, last = _split_name(full_name)
    ticker = _first_nonempty(entry, ("Ticker", "ticker"))
    transaction = _first_nonempty(entry, ("Transaction", "transaction"))
    amount = _first_nonempty(entry, AMOUNT_KEYS)
    trade_date = _first_nonempty(entry, DATE_KEYS)

    if not (first and last and ticker and transaction and trade_date):
        logging.debug(
            "Skipping row (missing required fields): name=%r ticker=%r txn=%r date=%r",
            full_name,
            ticker,
            transaction,
            trade_date,
        )
        return None

    key = _name_key(first, last)
    bioguide_candidates = [
        _first_nonempty(entry, BIOGUIDE_KEYS),
        name_index.get(key, ""),
    ]
    bioguide = next((c for c in bioguide_candidates if c), "")

    cleaned = {
        "first_name": first,
        "last_name": last,
        "representative": full_name,
        "BioGuideID": bioguide,
        "house": (entry.get("House") or entry.get("house") or "").strip(),
        "party": (entry.get("Party") or entry.get("party") or "").strip(),
        "transaction": transaction,
        "ticker": ticker,
        "ticker_type": _first_nonempty(entry, ("TickerType", "ticker_type")),
        "description": _first_nonempty(entry, ("Description", "description")),
        "range": amount,
        "amount": _first_nonempty(entry, ("Amount", "amount")),
        "excess_return": _first_nonempty(entry, ("ExcessReturn", "excess_return")),
        "price_change": _first_nonempty(entry, ("PriceChange", "price_change")),
        "spy_change": _first_nonempty(entry, ("SPYChange", "spy_change")),
        "report_date": _first_nonempty(entry, ("ReportDate", "report_date")),
        "transaction_date": trade_date,
        "last_modified": _first_nonempty(entry, ("last_modified", "LastModified")),
    }
    return cleaned


def merge_trades(existing: List[dict], fresh: List[dict]) -> Tuple[List[dict], int]:
    index = OrderedDict((dedupe_key(row), row) for row in existing)
    added = 0
    for row in fresh:
        key = dedupe_key(row)
        if key in index:
            continue
        index[key] = row
        added += 1
    # Sort most recent first; fall back to the insertion order if parsing fails
    def sort_key(row: dict) -> Tuple[int, str]:
        date_str = row.get("transaction_date", "")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return (int(dt.timestamp()), date_str)
        except ValueError:
            return (0, date_str)

    sorted_rows = sorted(index.values(), key=sort_key, reverse=True)
    return sorted_rows, added


def write_rows(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELD_ORDER})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        token = load_token(args.token_file)
    except (FileNotFoundError, ValueError) as exc:
        logging.error("Unable to resolve Quiver API token: %s", exc)
        return 2

    logging.info("Fetching live congress trades from Quiver...")
    try:
        payload = fetch_live_trades(token)
    except (FetchError, QuiverQuantError) as exc:
        logging.error("Failed to download live feed: %s", exc)
        return 3

    logging.info("Loaded %d rows from live endpoint", len(payload))

    name_index = build_name_index()
    logging.debug("Name index contains %d unique entries", len(name_index))

    normalized_rows = []
    for entry in payload:
        row = normalize_trade(entry, name_index=name_index)
        if row:
            normalized_rows.append(row)

    if not normalized_rows:
        logging.warning("No valid rows returned after normalization; nothing to merge.")
        return 0

    existing_rows = load_existing_rows(args.data_file)
    logging.info("Existing snapshot contains %d rows", len(existing_rows))

    merged_rows, added = merge_trades(existing_rows, normalized_rows)
    logging.info(
        "After merging, snapshot will contain %d rows (+%d new)",
        len(merged_rows),
        added,
    )

    if args.dry_run:
        logging.info("Dry-run enabled; skipping file write.")
        return 0

    write_rows(args.data_file, merged_rows)
    logging.info("Updated snapshot written to %s", args.data_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
