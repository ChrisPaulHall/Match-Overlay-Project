from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
import unicodedata
import html
from collections import deque
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import monotonic
from typing import Dict, List, Optional, Tuple, Set

import contextlib
from urllib.parse import quote
import numpy as np
import cv2

try:  # pragma: no cover - support running as standalone module
    from .config import MatcherConfig
except ImportError:  # pragma: no cover
    from config import MatcherConfig  # type: ignore

# ----------------------------------------------------------------------------
# Globals populated via init()
# ----------------------------------------------------------------------------
CONFIG: MatcherConfig | None = None

NAME_MAP_CSV = ""
SCRAPED_SUMMARY_CSV = ""
QUIVER_TOP_HOLDINGS_CSV = ""
CONGRESS_COMMITTEES_CSV = ""
CONGRESS_TRADES_CLEAN_CSV = ""
PERSONAL_INFO_CARD_CSV = ""
QUIVER_TOKEN_FILE = ""
ID_MAP_CSV = ""

OVERLAY_JSON = ""
OVERLAY_JSON_TMP = ""
HEARTBEAT_JSON = ""
HEARTBEAT_JSON_TMP = ""

name_map: Dict[str, dict] = {}
name_map_lower: Dict[str, dict] = {}
summary_lookup: Dict[str, dict] = {}
summary_by_bioguide: Dict[str, dict] = {}
committees_lookup: Dict[str, List[dict]] = {}
committees_last_index: Dict[str, List[dict]] = {}
committees_by_bioguide: Dict[str, List[dict]] = {}
FULL_SLUGS: List[str] = []
LAST_SLUGS: List[str] = []
FULL_SLUG_SET: set[str] = set()
LAST_SLUG_SET: set[str] = set()
ALLOWED_NAME_TOKENS: set[str] = set()
TITLE_TOKEN_SLUGS: set[str] = set()
NAME_SUFFIX_SLUGS: set[str] = set()
full_slug_to_bioguide: Dict[str, str] = {}
last_slug_to_bioguide_map: Dict[str, set[str]] = {}
filename_to_bioguide: Dict[str, str] = {}
bioguide_to_filenames: Dict[str, List[str]] = {}
bioguide_to_info: Dict[str, dict] = {}
holdings_index: Dict[str, dict] = {}
holdings_index_by_bioguide: Dict[str, dict] = {}
personal_info_index: Dict[str, dict] = {}
personal_info_index_by_bioguide: Dict[str, dict] = {}
trades_index: Dict[str, List[dict]] = {}
trades_index_by_bioguide: Dict[str, List[dict]] = {}
TICKER_NAME_MAP: Dict[str, str] = {}

TRADES_CACHE: Dict[Tuple, Tuple[list, str]] = {}
TRADES_CACHE_FILE = ""
_RECENT_TRADES_CACHE: List[dict] = []
_RECENT_TRADES_MTIME: float = 0.0

PENDING_REVIEW_DIR = ""
SAVED_HASHES: set[str] = set()
SAVED_COUNT_BY_PERSON: Dict[str, int] = {}
# Track scores saved per person in current session to avoid redundant captures
SESSION_SAVED_SCORES: Dict[str, List[float]] = {}

# Quiver integration
QUIVER_TOKEN: str = ""
QQ_OK = False
with contextlib.suppress(Exception):
    import quiverquant  # type: ignore
    QQ_OK = True

QUIVER_HELPERS_OK = False
try:  # pragma: no cover - optional dependency
    from quiver_api import QuiverQuantError, fetch_congress_trading
    QUIVER_HELPERS_OK = True
except Exception:  # pragma: no cover
    with contextlib.suppress(Exception):
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        quiver_module = importlib.import_module("core.quiver_api")
        QuiverQuantError = quiver_module.QuiverQuantError  # type: ignore[assignment]
        fetch_congress_trading = quiver_module.fetch_congress_trading  # type: ignore[assignment]
        QUIVER_HELPERS_OK = True
    if not QUIVER_HELPERS_OK:
        QuiverQuantError = RuntimeError  # type: ignore[assignment]
        fetch_congress_trading = None  # type: ignore[assignment]

_TRADES_CACHE: Dict[Tuple, dict] = {}
_TRADES_TTL = 172800  # seconds (48 hours)
_TRADES_CACHE_FLUSH_INTERVAL = 300.0  # seconds (batch disk writes every 5 minutes)
_TRADES_CACHE_DIRTY = False
_TRADES_CACHE_LAST_FLUSH = 0.0

# Overlay payload cache (metadata rarely changes; trades weekly)
_OVERLAY_CACHE: Dict[str, Dict[str, Any]] = {}
_OVERLAY_CACHE_TTL = 24 * 60 * 60.0  # seconds (cache for 24 hours)


def _cache_key_to_str(key: Tuple) -> str:
    try:
        kind, ident, limit = key
        return f"{kind}|{ident}|{limit}"
    except Exception:
        return "|".join(str(part) for part in key)


def _cache_key_from_str(raw: str) -> Tuple:
    parts = raw.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid cache key: {raw}")
    kind, ident, limit = parts
    return kind, ident, int(limit)


def _persist_trades_cache(*, force: bool = False) -> None:
    """Write the in-memory trades cache to disk, respecting the flush interval."""
    global _TRADES_CACHE_DIRTY, _TRADES_CACHE_LAST_FLUSH

    if not TRADES_CACHE_FILE:
        return
    if not _TRADES_CACHE_DIRTY and not force:
        return

    now_ts = time.time()
    if not force and (now_ts - _TRADES_CACHE_LAST_FLUSH) < _TRADES_CACHE_FLUSH_INTERVAL:
        return

    payload = {}
    for key, entry in _TRADES_CACHE.items():
        payload[_cache_key_to_str(key)] = {
            "ts": float(entry.get("ts") or now_ts),
            "ttl": float(entry.get("ttl", _TRADES_TTL)),
            "data": entry.get("data", []),
        }
    tmp_path = TRADES_CACHE_FILE + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, TRADES_CACHE_FILE)
        _TRADES_CACHE_LAST_FLUSH = now_ts
        _TRADES_CACHE_DIRTY = False
    except Exception as exc:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)
        logging.debug("Trade cache persist failed: %s", exc)


def _load_trades_cache() -> None:
    global _TRADES_CACHE_DIRTY
    if not TRADES_CACHE_FILE or not os.path.exists(TRADES_CACHE_FILE):
        return
    try:
        with open(TRADES_CACHE_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logging.debug("Trade cache load failed: %s", exc)
        return

    now_ts = time.time()
    now_mono = monotonic()
    dirty = False

    for raw_key, entry in raw.items():
        try:
            key = _cache_key_from_str(raw_key)
        except Exception:
            dirty = True
            continue

        ttl_val = float(entry.get("ttl", _TRADES_TTL))
        stored_ts = float(entry.get("ts", 0.0))
        if ttl_val > 0 and stored_ts:
            if (now_ts - stored_ts) > ttl_val:
                dirty = True
                continue

        _TRADES_CACHE[key] = {
            "t": now_mono,
            "ts": stored_ts or now_ts,
            "ttl": ttl_val,
            "data": entry.get("data", []),
        }

    if dirty:
        _TRADES_CACHE_DIRTY = True
        _persist_trades_cache(force=True)


def _load_ticker_names() -> None:
    TICKER_NAME_MAP.clear()
    if not CONFIG:
        return
    company_file = os.path.join(CONFIG.output_dir, "company_tickers.json")
    if not os.path.exists(company_file):
        return
    try:
        with open(company_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception as exc:
        logging.debug("Ticker name load failed: %s", exc)
        return

    if isinstance(raw, dict):
        entries = raw.values()
    else:
        entries = raw  # pragma: no cover - unexpected format

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker") or "").strip().upper()
        title = str(entry.get("title") or "").strip()
        if not ticker or not title:
            continue
        TICKER_NAME_MAP[ticker] = _clean_company_title(title)


def _clean_company_title(title: str) -> str:
    if not title:
        return ""
    temp = re.sub(r"[.,]+", " ", title)
    temp = re.sub(r"\s+", " ", temp).strip()
    tokens = [tok for tok in temp.split(" ") if tok]
    if not tokens:
        return title.strip()

    suffixes = {
        "inc", "incorporated", "corp", "corporation", "company", "co", "co",
        "holdings", "group", "plc", "llc", "ltd", "lp", "partners", "sa", "spa",
        "ag", "nv", "se", "ab", "pty", "limited", "inc.", "corp.", "co.", "l.p."
    }
    while tokens and tokens[-1].lower().rstrip(".") in suffixes:
        tokens.pop()
    if tokens and tokens[-1].lower().rstrip(".") == "com":
        tokens.pop()

    result = " ".join(tokens).strip()
    if not result:
        result = title.strip()

    def _smart_title(word: str) -> str:
        if word.isupper() or word.isnumeric():
            return word
        if len(word) <= 3 and word.isalpha():
            return word.upper()
        return word.capitalize()

    pretty = " ".join(_smart_title(w) for w in result.split(" "))
    return pretty.strip()


def _get_cached_trades(key: Tuple, ttl: int = _TRADES_TTL):
    global _TRADES_CACHE_DIRTY
    rec = _TRADES_CACHE.get(key)
    if not rec:
        return None

    ttl_val = float(rec.get("ttl", ttl))
    if ttl_val <= 0:
        return rec.get("data")

    cached_mono = float(rec.get("t", 0.0) or 0.0)
    if cached_mono and (monotonic() - cached_mono) < ttl_val:
        return rec.get("data")

    cached_ts = float(rec.get("ts", 0.0) or 0.0)
    if cached_ts and (time.time() - cached_ts) < ttl_val:
        rec["t"] = monotonic()
        return rec.get("data")

    _TRADES_CACHE.pop(key, None)
    _TRADES_CACHE_DIRTY = True
    _persist_trades_cache()
    return None


def _set_cached_trades(key: Tuple, data, ttl: int = _TRADES_TTL) -> None:
    global _TRADES_CACHE_DIRTY
    _TRADES_CACHE[key] = {
        "t": monotonic(),
        "ts": time.time(),
        "ttl": ttl,
        "data": data,
    }
    _TRADES_CACHE_DIRTY = True
    _persist_trades_cache()


def flush_trades_cache(force: bool = False) -> None:
    """Public helper to flush pending trade-cache updates to disk."""
    _persist_trades_cache(force=force)


def _resolve_quiver_token(config: MatcherConfig) -> str:
    token = str(config.get("quiver_token", "") or "").strip()
    if token:
        return token
    token = os.environ.get("QUIVER_API_TOKEN") or os.environ.get("QQ_API_TOKEN") or ""
    if token:
        return token.strip()
    try:
        with open(os.path.join(config.data_dir, "quiver_token.txt"), "r", encoding="utf-8") as fh:
            token = (fh.read() or "").strip()
            if token:
                return token
    except Exception:
        pass
    if QUIVER_TOKEN_FILE and os.path.exists(QUIVER_TOKEN_FILE):
        try:
            with open(QUIVER_TOKEN_FILE, "r", encoding="utf-8") as fh:
                token = (fh.read() or "").strip()
                if token:
                    return token
        except Exception:
            pass
    return ""

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
LAST_PREFIX_WORDS = {"da","de","del","della","der","di","la","le","van","von","bin","al","mac","mc","st","san","santa","du"}
FIRST_ALIAS = {
    "william": ["bill", "billy", "will"],
    "robert": ["bob", "bobby", "rob", "robbie"],
    "richard": ["dick", "rick", "ricky", "rich"],
    "michael": ["mike"],
    "thomas": ["tom", "tommy"],
    "joseph": ["joe", "joey"],
    "andrew": ["andy", "drew"],
    "stephen": ["steve"],
    "edward": ["ed", "eddie"],
    "charles": ["chuck", "charlie"],
    "james": ["jim", "jimmy", "jamie"],
    "john": ["jack", "johnny"],
    "christopher": ["chris"],
    "katherine": ["kate", "kathy", "katie", "katy"],
    "margaret": ["peggy", "maggie"],
    "elizabeth": ["liz", "beth", "eliza", "liza", "lizzy"],
    "nicholas": ["nick", "nicky"],
    "alexander": ["alex"],
    "alexandra": ["alex"],
    "kristopher": ["kris", "chris"],
    "jeffrey": ["jeff"],
    "patricia": ["pat", "trish", "tricia"],
}

NAME_WORD_RE = re.compile(r"[A-Za-z'\-]{2,}")

party_map = {
    "democratic": "D", "democrat": "D", "d": "D",
    "republican": "R", "gop": "R", "r": "R",
    "independent": "I", "ind": "I", "i": "I",
    "libertarian": "L", "lib": "L", "l": "L",
    "green": "G", "g": "G",
}

state_abbr = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT",
    "Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI",
    "Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
    "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
    "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX",
    "Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "District of Columbia":"DC","Puerto Rico":"PR"
}

STATE_NORMALIZATION = {
    "alabama": "AL", "ala": "AL", "al": "AL",
    "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "ariz": "AZ", "az": "AZ",
    "arkansas": "AR", "ark": "AR", "ar": "AR",
    "california": "CA", "calif": "CA", "cali": "CA", "ca": "CA",
    "colorado": "CO", "colo": "CO", "co": "CO",
    "connecticut": "CT", "conn": "CT", "ct": "CT",
    "delaware": "DE", "del": "DE", "de": "DE",
    "florida": "FL", "fla": "FL", "fl": "FL",
    "georgia": "GA", "ga": "GA",
    "hawaii": "HI", "hi": "HI",
    "idaho": "ID", "id": "ID",
    "illinois": "IL", "ill": "IL", "il": "IL",
    "indiana": "IN", "ind": "IN", "in": "IN",
    "iowa": "IA", "ia": "IA",
    "kansas": "KS", "kan": "KS", "kans": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY",
    "louisiana": "LA", "la": "LA",
    "maine": "ME", "me": "ME",
    "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "mass": "MA", "ma": "MA",
    "michigan": "MI", "mich": "MI", "mi": "MI",
    "minnesota": "MN", "minn": "MN", "mn": "MN",
    "mississippi": "MS", "miss": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO",
    "montana": "MT", "mont": "MT", "mt": "MT",
    "nebraska": "NE", "nebr": "NE", "neb": "NE", "ne": "NE",
    "nevada": "NV", "nev": "NV", "nv": "NV",
    "newhampshire": "NH", "nh": "NH",
    "newjersey": "NJ", "nj": "NJ",
    "newmexico": "NM", "nm": "NM",
    "newyork": "NY", "ny": "NY",
    "northcarolina": "NC", "northc": "NC", "nc": "NC",
    "northdakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH",
    "oklahoma": "OK", "okla": "OK", "ok": "OK",
    "oregon": "OR", "ore": "OR", "or": "OR",
    "pennsylvania": "PA", "penn": "PA", "pa": "PA",
    "rhodeisland": "RI", "ri": "RI",
    "southcarolina": "SC", "southc": "SC", "sc": "SC",
    "southdakota": "SD", "sd": "SD",
    "tennessee": "TN", "tenn": "TN", "tn": "TN",
    "texas": "TX", "tex": "TX", "tx": "TX",
    "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT",
    "virginia": "VA", "va": "VA",
    "washington": "WA", "wash": "WA", "wa": "WA",
    "westvirginia": "WV", "westv": "WV", "wv": "WV",
    "wisconsin": "WI", "wis": "WI", "wisc": "WI", "wi": "WI",
    "wyoming": "WY", "wyo": "WY", "wy": "WY",
    "districtofcolumbia": "DC", "districtcolumbia": "DC", "columbia": "DC", "dc": "DC",
    "puertorico": "PR", "pr": "PR",
    "virginislands": "VI", "usvirginislands": "VI", "vi": "VI",
}


def init(config: MatcherConfig) -> None:
    global CONFIG, NAME_MAP_CSV, SCRAPED_SUMMARY_CSV, QUIVER_TOP_HOLDINGS_CSV
    global CONGRESS_COMMITTEES_CSV, CONGRESS_TRADES_CLEAN_CSV, PERSONAL_INFO_CARD_CSV
    global QUIVER_TOKEN_FILE, ID_MAP_CSV, OVERLAY_JSON, OVERLAY_JSON_TMP
    global HEARTBEAT_JSON, HEARTBEAT_JSON_TMP, PENDING_REVIEW_DIR
    global QUIVER_TOKEN, TRADES_CACHE_FILE

    CONFIG = config

    NAME_MAP_CSV = os.path.join(config.data_dir, "members_face_lookup.csv")
    SCRAPED_SUMMARY_CSV = os.path.join(config.data_dir, "members_donor_summary.csv")
    holdings_candidates = [
        os.path.join(config.data_dir, "members_holdings_and_sectors.csv"),
        os.path.join(config.output_dir, "quiver_networth_holdings4.csv"),
        os.path.join(config.output_dir, "quiver_networth_holdings2.csv"),
    ]
    QUIVER_TOP_HOLDINGS_CSV = _first_existing_path(holdings_candidates)
    CONGRESS_COMMITTEES_CSV = os.path.join(config.data_dir, "committee_membership_flat.csv")
    CONGRESS_TRADES_CLEAN_CSV = os.path.join(config.data_dir, "members_trades_clean.csv")
    PERSONAL_INFO_CARD_CSV = os.path.join(config.data_dir, "members_personal_profiles.csv")
    QUIVER_TOKEN_FILE = os.path.join(config.data_dir, "quiver_token.txt")

    OVERLAY_JSON = os.path.join(config.output_dir, "overlay_data.json")
    OVERLAY_JSON_TMP = OVERLAY_JSON + ".tmp"
    HEARTBEAT_JSON = os.path.join(config.output_dir, "heartbeat.json")
    HEARTBEAT_JSON_TMP = HEARTBEAT_JSON + ".tmp"
    PENDING_REVIEW_DIR = os.path.join(config.output_dir, "pending_review")
    os.makedirs(PENDING_REVIEW_DIR, exist_ok=True)
    TRADES_CACHE_FILE = os.path.join(config.output_dir, "quiver_trades_cache.json")

    QUIVER_TOKEN = _resolve_quiver_token(config)

    logging.debug("  Loading name map...")
    _load_name_map()
    logging.debug("  Loading donor summary...")
    _load_summary()
    logging.debug("  Loading holdings data...")
    _load_holdings()
    logging.debug("  Loading personal info cards...")
    _load_personal_info_card()
    logging.debug("  Loading committee assignments...")
    _load_committees()
    logging.debug("  Building name lexicons...")
    _build_lexicons()
    logging.debug("  Loading trades CSV...")
    _load_trades_csv()
    logging.debug("  Loading trades cache...")
    _load_trades_cache()
    logging.debug("  Loading ticker names...")
    _load_ticker_names()
    logging.debug("  Overlay initialization complete")


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------

def _first_existing_path(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return paths[0] if paths else ""


def _norm_headers(reader: csv.DictReader):
    if not reader.fieldnames:
        return
    reader.fieldnames = [(h or "").strip().lstrip("\ufeff").lower().replace(" ", "_") for h in reader.fieldnames]


def _split_multi(val: str):
    if not val:
        return []
    parts = re.split(r"[;|,/]+|\s{2,}", val)
    return [p.strip() for p in parts if p and p.strip()]


def _normalize_basic(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _normalize_unicode(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return _normalize_basic(s)


def _money_once(s: str) -> str:
    s = (s or "").strip().rstrip(")")
    s = s[1:] if s.startswith("$") else s
    return f"${s}" if s else ""


def _name_tokens_lower(s: str):
    return [t for t in re.split(r"[^a-z]+", (s or "").lower()) if t]


def _collapse_last_parts(last_raw: str):
    if not last_raw:
        return []
    base = last_raw.strip()
    last_forms = {base}
    parts = [p for p in re.split(r"[\s\-']+", base) if p]
    if "-" in base:
        for p in base.split("-"):
            if len(p.strip()) >= 2:
                last_forms.add(p.strip())
    if len(parts) >= 2 and parts[0].lower() in LAST_PREFIX_WORDS:
        joined = "".join(parts)
        last_forms.add(joined)
        last_forms.add("".join(parts[:-1]) + " " + parts[-1])
    for p in parts:
        if len(p) >= 2:
            last_forms.add(p)
    return list(last_forms)


def _first_roots_from_fields(first_raw: str, nick_field: str):
    roots = set()
    fl = (first_raw or "").strip()
    if fl:
        tok = _name_tokens_lower(fl)
        if tok:
            primary = tok[0]
            roots.add(primary)
            for k, vs in FIRST_ALIAS.items():
                if primary == k or primary in vs:
                    roots.update([k] + vs)
        roots.add(fl[:1].lower().strip("."))
    for nx in _split_multi(nick_field or ""):
        nx_tok = _name_tokens_lower(nx)
        if nx_tok:
            roots.add(nx_tok[0])
    return {r for r in roots if r}


def _full_slug(s: str) -> str:
    return _normalize_unicode(s)


def slugify(text: str) -> str:
    return _normalize_unicode(text)


def _make_full_variants(row):
    first = (row.get("first_name") or row.get("first") or "").strip()
    last  = (row.get("last_name")  or row.get("last")  or "").strip()
    mid   = (row.get("middle_name") or row.get("middle") or row.get("mi") or "").strip()
    nick  = (row.get("nick_name") or row.get("nickname") or "").strip()
    aka   = (row.get("aka") or row.get("also_known_as") or "").strip()
    alt_last = (row.get("last_name1") or row.get("alt_last") or row.get("surname_alt") or "").strip()

    last_forms = _collapse_last_parts(last)
    if alt_last:
        last_forms += _collapse_last_parts(alt_last)

    first_roots = _first_roots_from_fields(first, nick)

    aka_pairs = []
    for a in _split_multi(aka):
        a = a.replace(",", " ").strip()
        toks = _name_tokens_lower(a)
        if len(toks) >= 2:
            aka_pairs.append((" ".join(toks[:-1]).title(), toks[-1].title()))

    seen = set()

    for lf in last_forms or [last]:
        disp = f"{first} {lf}".strip()
        slug = _full_slug(disp)
        if slug and slug not in seen:
            seen.add(slug); yield disp, slug

    if first:
        fi = first[:1] + "."
        for lf in last_forms or [last]:
            disp = f"{fi} {lf}".strip()
            slug = _full_slug(disp)
            if slug and slug not in seen:
                seen.add(slug); yield disp, slug

    if first and mid:
        mi = mid[:1] + "."
        for lf in last_forms or [last]:
            disp = f"{first} {mi} {lf}".strip()
            slug = _full_slug(disp)
            if slug and slug not in seen:
                seen.add(slug); yield disp, slug

    for fr in sorted(first_roots):
        for lf in last_forms or [last]:
            disp = f"{fr.title()} {lf}".strip()
            slug = _full_slug(disp)
            if slug and slug not in seen:
                seen.add(slug); yield disp, slug

    for af, al in aka_pairs:
        for lf in _collapse_last_parts(al) or [al]:
            disp = f"{af} {lf}".strip()
            slug = _full_slug(disp)
            if slug and slug not in seen:
                seen.add(slug); yield disp, slug


full_name_to_filename: Dict[str, str] = {}
last_name_to_filenames: Dict[str, List[str]] = {}
PRIORITY: Dict[str, int] = {}


def _push_full_slug(slug: str, filename: str, priority: int, bioguide: str) -> None:
    if not slug:
        return
    prev = PRIORITY.get(slug)
    if prev is None or priority < prev:
        PRIORITY[slug] = priority
        full_name_to_filename[slug] = filename
        if bioguide:
            full_slug_to_bioguide[slug] = bioguide


def _push_last_slug(slug: str, filename: str, bioguide: str) -> None:
    if not slug:
        return
    last_name_to_filenames.setdefault(slug, [])
    if filename not in last_name_to_filenames[slug]:
        last_name_to_filenames[slug].append(filename)
    if bioguide:
        last_slug_to_bioguide_map.setdefault(slug, set()).add(bioguide)


def _load_name_map() -> None:
    name_map.clear()
    name_map_lower.clear()
    full_name_to_filename.clear()
    last_name_to_filenames.clear()
    PRIORITY.clear()
    full_slug_to_bioguide.clear()
    last_slug_to_bioguide_map.clear()
    filename_to_bioguide.clear()
    bioguide_to_filenames.clear()
    bioguide_to_info.clear()

    if not NAME_MAP_CSV or not os.path.exists(NAME_MAP_CSV):
        logging.error("Name map CSV missing: %s", NAME_MAP_CSV)
        return

    with open(NAME_MAP_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _norm_headers(reader)
        for row in reader:
            fn = (row.get("filename") or row.get("image") or row.get("filepath") or "").strip()
            if fn:
                fn = os.path.basename(fn)
            if not fn:
                continue

            bid = (row.get("bioguide_id") or row.get("bioguideid") or row.get("bioguide")
                   or row.get("bioguide_id_") or row.get("bioguideid_") or row.get("BioGuideID") or "").strip()

            first = (row.get("first_name") or row.get("first") or "").strip()
            last  = (row.get("last_name")  or row.get("last")  or "").strip()

            name_map[fn] = {
                "first_name": first,
                "last_name":  last,
                "middle_name": (row.get("middle_name") or row.get("middle") or row.get("mi") or "").strip(),
                "nick_name": (row.get("nick_name") or row.get("nickname") or "").strip(),
                "aka": (row.get("aka") or row.get("also_known_as") or "").strip(),
                "title": (row.get("title") or "").strip(),
                "state": (row.get("state") or "").strip(),
                "party": (row.get("party") or "").strip(),
                "terms": (row.get("terms") or "").strip(),
                "bioguide_id": bid,
                "_row": row
            }
            name_map_lower[fn.lower()] = name_map[fn]

            if bid:
                filename_to_bioguide[fn] = bid
                bioguide_to_filenames.setdefault(bid, []).append(fn)
                bioguide_to_info.setdefault(bid, name_map[fn])
                root = re.sub(r"\d+$", "", os.path.splitext(fn)[0])
                if root:
                    filename_to_bioguide.setdefault(root, bid)

            if not last:
                continue
            seen_local = set()
            try:
                for disp, slug in _make_full_variants(name_map[fn]):
                    if slug in seen_local:
                        continue
                    seen_local.add(slug)
                    pr = 2
                    if _full_slug(f"{first} {last}") == slug:
                        pr = 0
                    elif name_map[fn].get("middle_name") and (_full_slug(f"{first} {name_map[fn]['middle_name'][:1]}. {last}") == slug):
                        pr = 1
                    elif _full_slug(first[:1] + " " + last) == slug:
                        pr = 3
                    _push_full_slug(slug, fn, pr, bid)
            except Exception:
                continue

            for lf in _collapse_last_parts(last):
                _push_last_slug(_full_slug(lf), fn, bid)


def _build_lexicons() -> None:
    global FULL_SLUGS, LAST_SLUGS, FULL_SLUG_SET, LAST_SLUG_SET
    global ALLOWED_NAME_TOKENS, TITLE_TOKEN_SLUGS, NAME_SUFFIX_SLUGS

    FULL_SLUGS = list(full_name_to_filename.keys())
    LAST_SLUGS = list(last_name_to_filenames.keys())
    FULL_SLUG_SET = set(FULL_SLUGS)
    LAST_SLUG_SET = set(LAST_SLUGS)

    tokens: set[str] = set()
    for info in name_map.values():
        for field in ("first_name", "last_name", "middle_name", "nick_name", "aka"):
            val = info.get(field) or ""
            for tok in _name_tokens_lower(val):
                slug = _normalize_unicode(tok)
                if slug:
                    tokens.add(slug)
    ALLOWED_NAME_TOKENS = tokens

    TITLE_TOKEN_SLUGS = { _normalize_unicode(x) for x in [
        "sen", "senator", "rep", "representative", "gov", "governor",
        "delegate", "del", "president", "speaker", "mayor", "chair", "chairman",
        "chairwoman", "dr", "mr", "mrs", "ms"
    ] if _normalize_unicode(x)}
    NAME_SUFFIX_SLUGS = { _normalize_unicode(s) for s in NAME_SUFFIXES }


def _load_summary() -> None:
    summary_lookup.clear()
    summary_by_bioguide.clear()
    if not SCRAPED_SUMMARY_CSV or not os.path.exists(SCRAPED_SUMMARY_CSV):
        logging.warning("Summary CSV missing: %s", SCRAPED_SUMMARY_CSV)
        return
    with open(SCRAPED_SUMMARY_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first = (row.get("first") or row.get("first_name") or "").strip()
            last = (row.get("last") or row.get("last_name") or "").strip()
            slug = _fullnorm_key(first, last)
            bid = (row.get("bioguide_id") or row.get("bioguide") or "").strip()
            if slug:
                summary_lookup[slug] = row
            if bid:
                summary_by_bioguide[bid] = row


def _load_holdings() -> None:
    holdings_index.clear()
    holdings_index_by_bioguide.clear()
    if not QUIVER_TOP_HOLDINGS_CSV or not os.path.exists(QUIVER_TOP_HOLDINGS_CSV):
        return
    with open(QUIVER_TOP_HOLDINGS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first = (row.get("first_name") or row.get("first") or "").strip()
            last = (row.get("last_name") or row.get("last") or "").strip()
            slug = _fullnorm_key(first, last)
            bid = (row.get("bioguide_id") or row.get("bioguide") or "").strip()
            if slug:
                holdings_index[slug] = row
            if bid:
                holdings_index_by_bioguide[bid] = row


def _load_personal_info_card() -> None:
    personal_info_index.clear()
    personal_info_index_by_bioguide.clear()
    if not PERSONAL_INFO_CARD_CSV or not os.path.exists(PERSONAL_INFO_CARD_CSV):
        logging.warning("Personal info card CSV missing: %s", PERSONAL_INFO_CARD_CSV)
        return
    with open(PERSONAL_INFO_CARD_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first = (row.get("first_name") or row.get("first") or "").strip()
            last = (row.get("last_name") or row.get("last") or "").strip()
            slug = _fullnorm_key(first, last)
            bid = (row.get("bioguide_id") or row.get("bioguide") or "").strip()
            if slug:
                personal_info_index[slug] = row
            if bid:
                personal_info_index_by_bioguide[bid] = row


def _load_committees() -> None:
    committees_lookup.clear()
    committees_last_index.clear()
    committees_by_bioguide.clear()
    if not CONGRESS_COMMITTEES_CSV or not os.path.exists(CONGRESS_COMMITTEES_CSV):
        return
    with open(CONGRESS_COMMITTEES_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first = (row.get("first_name") or row.get("first") or "").strip()
            last = (row.get("last_name") or row.get("last") or "").strip()
            slug = _fullnorm_key(first, last)
            bid = (row.get("bioguide_id") or row.get("bioguide") or "").strip()
            if not slug:
                continue
            committees_lookup.setdefault(slug, []).append(row)
            last_slug = _normalize_unicode(last)
            committees_last_index.setdefault(last_slug, []).append(row)
            if bid:
                committees_by_bioguide.setdefault(bid, []).append(row)


def _load_trades_csv() -> None:
    trades_index.clear()
    trades_index_by_bioguide.clear()
    if not CONGRESS_TRADES_CLEAN_CSV or not os.path.exists(CONGRESS_TRADES_CLEAN_CSV):
        return
    with open(CONGRESS_TRADES_CLEAN_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            first = (row.get("first_name") or row.get("first") or "").strip()
            last = (row.get("last_name") or row.get("last") or "").strip()
            slug = _fullnorm_key(first, last)
            bid = (row.get("bioguide_id") or row.get("bioguide") or "").strip()
            if slug:
                trades_index.setdefault(slug, []).append(row)
            if bid:
                trades_index_by_bioguide.setdefault(bid, []).append(row)


def _fullnorm_key(first: str, last: str) -> str:
    return _normalize_unicode(f"{first} {last}")


def resolve_full_slug(slug: str) -> Optional[str]:
    return full_name_to_filename.get(slug)


def resolve_last_slug(slug: str) -> List[str]:
    return last_name_to_filenames.get(slug, [])


def slug_to_bioguide(slug: str) -> Optional[str]:
    return full_slug_to_bioguide.get(slug)


def last_slug_bioguide_candidates(slug: str) -> List[str]:
    return sorted(last_slug_to_bioguide_map.get(slug, set()))


def get_bioguide_for_filename(filename: str) -> str:
    if not filename:
        return ""
    base = os.path.basename(filename)
    bid = filename_to_bioguide.get(base)
    if bid:
        return bid
    root = re.sub(r"\d+$", "", os.path.splitext(base)[0])
    if root:
        bid = filename_to_bioguide.get(root)
        if bid:
            return bid
    info = info_for_person("", base)
    return info.get("bioguide_id", "") if info else ""


def get_primary_filename(bioguide: str) -> Optional[str]:
    files = bioguide_to_filenames.get(bioguide)
    return files[0] if files else None


def info_for_person(bioguide: str, filename: str) -> dict:
    if bioguide and bioguide in bioguide_to_info:
        return bioguide_to_info[bioguide]
    if filename and filename in name_map:
        return name_map[filename]
    if filename:
        info = name_map_lower.get(filename.lower())
        if info:
            return info
        root = re.sub(r"\d+$", "", os.path.splitext(filename)[0])
        if root:
            root_lower = root.lower()
            for fn_lower, rec in name_map_lower.items():
                if os.path.splitext(fn_lower)[0] == root_lower:
                    return rec
    return {}


def _mk_quiver_url(first: str, last: str, bioguide: str = "") -> str:
    if bioguide:
        return f"https://www.quiverquant.com/congresstrading/politician/{quote((first + ' ' + last).strip())}-{bioguide}"
    return f"https://www.quiverquant.com/politicians/?q={quote((first + ' ' + last).strip())}"


def _fetch_quiver_bulk(bioguide: str, limit: int) -> List[dict]:
    if not (bioguide and QUIVER_TOKEN and QUIVER_HELPERS_OK and fetch_congress_trading):
        return []
    try:
        data = fetch_congress_trading(bioguide, token=QUIVER_TOKEN)
        if not data:
            return []

        def _parse_date(val):
            try:
                return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            except Exception:
                return datetime.min

        data.sort(key=lambda r: _parse_date(r.get("Filed") or r.get("Traded")), reverse=True)
        out: List[dict] = []
        for row in data[:max(1, int(limit))]:
            ticker = str(row.get("Ticker") or "").strip()
            ticker_upper = ticker.upper()
            ticker_type = str(row.get("TickerType") or "").strip()
            txn = str(row.get("Transaction") or "").strip()
            traded = str(row.get("Traded") or "").strip()
            filed = str(row.get("Filed") or "").strip()
            status = str(row.get("Status") or "").strip()
            range_val = str(
                row.get("Range")
                or row.get("range")
                or row.get("AmountRange")
                or row.get("amount_range")
                or row.get("RangeDisplay")
                or ""
            ).strip()
            amount = str(
                row.get("Trade_Size_USD")
                or row.get("trade_size_usd")
                or row.get("Amount")
                or row.get("amount")
                or row.get("Trade_Size_USD_num")
                or ""
            ).strip()
            company = str(row.get("Company") or "").strip()
            description = str(row.get("Description") or "").strip()
            company_clean = TICKER_NAME_MAP.get(ticker_upper, "")
            asset = description or company or company_clean
            out.append({
                "ticker": ticker_upper,
                "ticker_type": ticker_type,
                "asset": asset,
                "company": company,
                "description": description,
                "company_clean": company_clean,
                "transaction": txn,
                "status": status,
                "traded": traded,
                "filed": filed,
                "amount": amount,
                "range": range_val or amount,
                "amount_range": range_val or amount,
                "trade_size_usd": amount,
            })
        return out
    except QuiverQuantError as exc:
        logging.debug("Quiver bulk fetch error for %s: %s", bioguide, exc)
        return []
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Quiver bulk unexpected error for %s: %s", bioguide, exc)
        return []


def get_quiver_trades(first: str, last: str, *, bioguide: str = "", limit: int = 5) -> Tuple[List[dict], str]:
    key = ("quiver_trades", bioguide or _fullnorm_key(first, last), int(limit))
    cached = _get_cached_trades(key)
    if cached is not None:
        return cached.get("trades", []), cached.get("url", "")

    quiver_url = _mk_quiver_url(first, last, bioguide)
    trades: List[dict] = []

    if bioguide:
        trades = _fetch_quiver_bulk(bioguide, limit)

    _set_cached_trades(key, {"trades": trades, "url": quiver_url})
    return trades, quiver_url


def is_valid_token(word: str) -> bool:
    slug = _normalize_unicode(word)
    if not slug or len(slug) < 2:
        return False
    if slug in TITLE_TOKEN_SLUGS or slug in NAME_SUFFIX_SLUGS:
        return True
    if slug in LAST_SLUG_SET or slug in ALLOWED_NAME_TOKENS:
        if any(ch in "aeiouy" for ch in slug):
            return True
    return False


def score_ocr_text(text: str) -> float:
    words = NAME_WORD_RE.findall(text or "")
    if len(words) >= 2:
        for i in range(len(words) - 1):
            slug = _normalize_unicode(f"{words[i]} {words[i+1]}")
            if slug in FULL_SLUG_SET:
                return 100.0
    elif len(words) == 1:
        slug = _normalize_unicode(words[0])
        if slug in LAST_SLUG_SET:
            return 90.0
    return 0.0


def resolve_bioguide(first: str, last: str) -> str:
    slug = _fullnorm_key(first, last)
    bid = slug_to_bioguide(slug)
    if bid:
        return bid
    for info in name_map.values():
        if _fullnorm_key(info.get("first_name", ""), info.get("last_name", "")) == slug:
            return info.get("bioguide_id", "")
    return ""


def _committees_for_member(first: str, last: str, title: str, limit: int = 6, bioguide: str = "") -> List[dict]:
    slug = _fullnorm_key(first, last)
    entries = []
    if bioguide and bioguide in committees_by_bioguide:
        entries.extend(committees_by_bioguide.get(bioguide, []))
    if not entries:
        entries.extend(committees_lookup.get(slug, []))
    last_bucket = committees_last_index.get(_normalize_unicode(last), [])
    if not entries and last_bucket:
        roots = _first_roots_from_fields(first, "")
        for row in last_bucket:
            rf = (row.get("first_name") or row.get("first") or "").strip().lower()
            rf_tokens = [t for t in re.split(r"[^a-z]+", rf) if t]
            rf_primary = rf_tokens[0] if rf_tokens else rf.strip(".")
            if roots and rf_primary and rf_primary not in roots:
                continue
            entries.append(row)
    out = []
    seen = set()
    for row in entries:
        com = (row.get("committee") or "").strip()
        sub = (row.get("subcommittee") or "").strip()
        key = (com.lower(), sub.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({"committee": com, "subcommittee": sub})
        if len(out) >= limit:
            break
    return out


def pretty_date_ymd(iso_str: str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.strptime(iso_str.strip(), "%Y-%m-%d")
        day = int(dt.strftime("%d"))
        return f"{dt.strftime('%B')} {day}, {dt.year}"
    except Exception:
        return iso_str


def _format_birth_date(raw: str, age_hint: str = "") -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("–", "-").replace("—", "-")
    patterns = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y")
    now_year = datetime.now().year
    for fmt in patterns:
        try:
            dt = datetime.strptime(normalized, fmt)
            if "%y" in fmt and dt.year > now_year:
                dt = dt.replace(year=dt.year - 100)
            if "%y" in fmt and dt.year > now_year:
                # If subtracting 100 still leaves a future year, fall back to 100-year shift again
                dt = dt.replace(year=dt.year - 100)
            if "%y" in fmt and dt.year > now_year:
                continue
            if "%y" in fmt and dt.year < 1900 and age_hint:
                try:
                    age_val = int(float(age_hint))
                    approx_year = now_year - age_val
                    if approx_year % 100 == dt.year % 100:
                        dt = dt.replace(year=approx_year)
                    elif (approx_year - 1) % 100 == dt.year % 100:
                        dt = dt.replace(year=approx_year - 1)
                    elif (approx_year + 1) % 100 == dt.year % 100:
                        dt = dt.replace(year=approx_year + 1)
                except Exception:
                    pass
            return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except ValueError:
            continue
    return raw


def _normalize_age(age_val) -> str:
    if age_val is None:
        return ""
    if isinstance(age_val, (int, float)):
        return str(int(age_val))
    age_str = str(age_val).strip()
    if not age_str:
        return ""
    try:
        num = float(age_str)
    except ValueError:
        return age_str
    if abs(num - int(num)) < 1e-9:
        return str(int(num))
    return f"{num:.1f}".rstrip("0").rstrip(".")


def _clean_age_pretty(raw) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    # Normalize dashes and remove stray negative signs that sometimes appear.
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"-\s*(\d+)", r"\1", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,")


def _clean_birthplace(raw: str) -> str:
    place = (raw or "").strip().strip('"').strip()
    place = place.replace(".,", ".").replace(",,", ",")
    place = re.sub(r"\s{2,}", " ", place)
    place = place.strip(" ,")
    return place


def _normalize_state_token(token: str) -> str:
    if not token:
        return ""
    cleaned = re.sub(r"[^A-Za-z]", "", token).lower()
    if not cleaned:
        return ""
    if cleaned in STATE_NORMALIZATION:
        return STATE_NORMALIZATION[cleaned]
    # Try separating north/south/east/west prefix (e.g., ncarolina)
    for prefix in ("north", "south", "west", "east"):
        if cleaned.startswith(prefix) and cleaned[len(prefix):] in STATE_NORMALIZATION:
            return STATE_NORMALIZATION[cleaned[len(prefix):]]
    if len(cleaned) == 2:
        return cleaned.upper()
    return ""


def _state_from_token(token: str) -> str:
    if not token:
        return ""
    direct = _normalize_state_token(token)
    if direct:
        return direct
    for piece in re.split(r"[\s/]+", token):
        if not piece:
            continue
        candidate = _normalize_state_token(piece)
        if candidate:
            return candidate
    return ""


def _is_location_token(token: str) -> bool:
    lowered = token.lower().strip()
    if not lowered:
        return False
    if any(keyword in lowered for keyword in ("university", "college", "academy", "institute", "seminary", "school")):
        if "high school" not in lowered:
            return False
    if "county" in lowered or "parish" in lowered or "borough" in lowered:
        return True
    if _state_from_token(token):
        return True
    return False


def _format_birth_place(raw: str) -> str:
    cleaned = _clean_birthplace(raw)
    if not cleaned:
        return ""
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if not parts:
        return ""
    city = parts[0]
    state = ""
    for token in reversed(parts[1:]):
        state_candidate = ""
        if _is_location_token(token):
            state_candidate = _state_from_token(token)
        else:
            state_candidate = _state_from_token(token)
        if state_candidate:
            state = state_candidate
            break
    if not state and len(parts) >= 2:
        state = _state_from_token(parts[-1])
    city = re.sub(r"\s{2,}", " ", city).strip()
    if not state:
        return city
    return f"{city}, {state}"


DEGREE_KEYWORDS = {
    "associate",
    "a.a.",
    "a.s.",
    "a.a.s.",
    "b.a.",
    "ba.",
    "a.b.",
    "ab.",
    "ab",
    "a. b.",
    "artium baccalaureus",
    "b.s.",
    "bs.",
    "b.s",
    "bfa",
    "b.f.a.",
    "b.b.a.",
    "b.arch.",
    "b.eng.",
    "b.e.",
    "b.s.e.",
    "b.s.m.e.",
    "b.m.e.",
    "b.s.c.e.",
    "b.s.m.",
    "bacc",
    "bachelor",
    "m.a.",
    "ma.",
    "m.s.",
    "ms.",
    "m.f.",
    "m.f.a.",
    "m.b.a.",
    "mba",
    "m.p.a.",
    "mpa",
    "m.p.p.",
    "mpp",
    "m.p.h.",
    "mph",
    "m.h.a.",
    "m.ed.",
    "med.",
    "m.s.w.",
    "msw",
    "m.phil.",
    "m.phil",
    "m.div.",
    "mtheol",
    "m.t.s.",
    "m.eng.",
    "meng",
    "m.s.e.",
    "m.s.e.e.",
    "m.s.m.e.",
    "m.s.c.e.",
    "mps",
    "mps.",
    "master",
    "j.d.",
    "jd",
    "ll.m.",
    "llm",
    "ll.b.",
    "llb",
    "s.j.d.",
    "sjd",
    "ph.d.",
    "phd",
    "d.phil.",
    "dphil",
    "m.d.",
    "md",
    "d.o.",
    "do",
    "d.d.s.",
    "dds",
    "d.m.d.",
    "dmd",
    "d.v.m.",
    "dvm",
    "d.p.m.",
    "dpm",
    "pham.d",
    "pharm.d",
    "doctor",
    "d.min.",
    "ed.d.",
    "edd",
    "sc.d.",
    "scd",
    "d.sc.",
    "msfs",
    "m.s.f.s.",
    "mfa",
    "m.l.a.",
    "mla",
}


_DEGREE_SHORT_MAP: Dict[str, str] = {
    "AA": "AA",
    "AAA": "AAA",
    "AAS": "AAS",
    "AS": "AS",
    "AB": "AB",
    "BA": "BA",
    "BS": "BS",
    "SB": "SB",
    "BFA": "BFA",
    "BBA": "BBA",
    "BPA": "BPA",
    "BAS": "BAS",
    "BARCH": "BArch",
    "BENG": "BEng",
    "BE": "BE",
    "BSE": "BSE",
    "BSEE": "BSEE",
    "BSME": "BSME",
    "BSCE": "BSCE",
    "BME": "BME",
    "BSN": "BSN",
    "BSS": "BSS",
    "BGS": "BGS",
    "BLA": "BLA",
    "MA": "MA",
    "AM": "AM",
    "MS": "MS",
    "SM": "SM",
    "MBA": "MBA",
    "MPA": "MPA",
    "MPP": "MPP",
    "MPH": "MPH",
    "MPS": "MPS",
    "MSW": "MSW",
    "MFA": "MFA",
    "MSFS": "MSFS",
    "MENG": "MEng",
    "MSE": "MSE",
    "MLA": "MLA",
    "MNA": "MNA",
    "MNS": "MNS",
    "JD": "JD",
    "LLB": "LLB",
    "LLM": "LLM",
    "LLD": "LLD",
    "SJD": "SJD",
    "MD": "MD",
    "DO": "DO",
    "DDS": "DDS",
    "DMD": "DMD",
    "DVM": "DVM",
    "DPM": "DPM",
    "PHARMD": "PharmD",
    "PHD": "PhD",
    "DPHIL": "DPhil",
    "EDD": "EdD",
    "DED": "DEd",
    "SCD": "ScD",
    "DSC": "ScD",
    "DMIN": "D.Min.",
    "MDIV": "M.Div.",
    "MTH": "M.Th.",
    "BTH": "B.Th.",
}

_DEGREE_PHRASE_MAP: Dict[str, str] = {
    "ASSOCIATE OF ARTS": "AA",
    "ASSOCIATE OF SCIENCE": "AS",
    "ASSOCIATE OF APPLIED SCIENCE": "AAS",
    "ASSOCIATE OF BUSINESS": "AB",
    "BACHELOR OF ARTS": "BA",
    "BACHELOR OF SCIENCE": "BS",
    "BACHELOR OF SCIENCE IN ENGINEERING": "BSE",
    "BACHELOR OF SCIENCE IN ELECTRICAL ENGINEERING": "BSEE",
    "BACHELOR OF SCIENCE IN MECHANICAL ENGINEERING": "BSME",
    "BACHELOR OF SCIENCE IN CIVIL ENGINEERING": "BSCE",
    "BACHELOR OF BUSINESS ADMINISTRATION": "BBA",
    "BACHELOR OF FINE ARTS": "BFA",
    "BACHELOR OF ARCHITECTURE": "BArch",
    "BACHELOR OF LAWS": "LLB",
    "BACHELOR OF THEOLOGY": "B.Th.",
    "MASTER OF ARTS": "MA",
    "MASTER OF SCIENCE": "MS",
    "MASTER OF SCIENCE IN ENGINEERING": "MSE",
    "MASTER OF SCIENCE IN FOREIGN SERVICE": "MSFS",
    "MASTER OF BUSINESS ADMINISTRATION": "MBA",
    "MASTER OF PUBLIC ADMINISTRATION": "MPA",
    "MASTER OF PUBLIC AFFAIRS": "MPA",
    "MASTER OF PUBLIC POLICY": "MPP",
    "MASTER OF PUBLIC HEALTH": "MPH",
    "MASTER OF EDUCATION": "MEd",
    "MASTER OF ENGINEERING": "MEng",
    "MASTER OF FINE ARTS": "MFA",
    "MASTER OF LIBERAL ARTS": "MLA",
    "MASTER OF SOCIAL WORK": "MSW",
    "MASTER OF PROFESSIONAL STUDIES": "MPS",
    "MASTER OF LAWS": "LLM",
    "MASTER OF DIVINITY": "M.Div.",
    "MASTER OF THEOLOGY": "M.Th.",
    "MASTER OF NATIONAL SECURITY AFFAIRS": "MNSA",
    "DOCTOR OF MEDICINE": "MD",
    "DOCTOR OF OSTEOPATHIC MEDICINE": "DO",
    "DOCTOR OF DENTAL SURGERY": "DDS",
    "DOCTOR OF DENTAL MEDICINE": "DMD",
    "ARTIUM BACCALAUREUS": "AB",
    "DOCTOR OF PHILOSOPHY": "PhD",
    "DOCTOR OF EDUCATION": "EdD",
    "DOCTOR OF JURISPRUDENCE": "JD",
    "JURIS DOCTOR": "JD",
    "JURIS SCIENTIAE DOCTOR": "SJD",
    "DOCTOR OF LAWS": "LLD",
    "DOCTOR OF MINISTRY": "D.Min.",
}

_EDU_ROLE_KEYWORDS = (
    "professor",
    "teacher",
    "principal",
    "coach",
    "instructor",
    "lecturer",
)

_EDU_LEADING_PATTERN = re.compile(
    r"^(?:with\s+(?:an?\s+)?)?"
    r"(?:(?:an?\s+)?)?"
    r"(?:graduated|graduating|earned|earning|received|receiving|completed|completing|"
    r"obtained|obtaining|attended|attending|studied|studying|holds|holding|awarded|"
    r"was awarded|pursued|pursuing)\s+"
    r"(?:(?:an?\s+)?degree\s+in\s+)?"
    r"(?:(?:an?\s+)?(?:from|at|in)\s+)?",
    re.IGNORECASE,
)


def _normalize_education_token(token: str) -> str:
    cleaned = html.unescape(token.replace("•", " "))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,\u2022")
    if not cleaned:
        return ""
    previous = None
    # Remove leading verbs like "graduated", "earned", etc.
    while previous != cleaned:
        previous = cleaned
        cleaned = _EDU_LEADING_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"\bdegrees?\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwith honors\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,")


def _strip_education_year(segment: str) -> Tuple[str, str]:
    match = re.search(r"(18|19|20)\d{2}(?!.*(18|19|20)\d{2})", segment)
    if not match:
        return segment, ""
    year = match.group()
    without = (segment[:match.start()] + segment[match.end():]).strip(" ,;-")
    without = re.sub(r"\s{2,}", " ", without)
    return without, year


def _canonical_degree_from_token(token: str) -> str:
    if not token:
        return ""
    normalized = re.sub(r"\s{2,}", " ", token).strip()
    key = re.sub(r"[^A-Za-z0-9]", "", normalized).upper()
    if not key:
        return ""
    if key.endswith("DEGREE"):
        key = key[:-6]
    if key in _DEGREE_SHORT_MAP:
        return _DEGREE_SHORT_MAP[key]

    phrase_key = re.sub(r"[^\w\s]", "", normalized).upper()
    phrase_key = re.sub(r"\s+", " ", phrase_key).strip()
    if not phrase_key:
        return ""
    if phrase_key in _DEGREE_PHRASE_MAP:
        return _DEGREE_PHRASE_MAP[phrase_key]
    for phrase, display in _DEGREE_PHRASE_MAP.items():
        if phrase_key.startswith(phrase):
            return display

    if "ASSOCIATE" in phrase_key:
        if "APPLIED SCIENCE" in phrase_key:
            return "AAS"
        if "SCIENCE" in phrase_key:
            return "AS"
        if "ART" in phrase_key:
            return "AA"
    if "BACHELOR" in phrase_key:
        if "BUSINESS" in phrase_key and "ADMINISTRATION" in phrase_key:
            return "BBA"
        if "SCIENCE" in phrase_key:
            if "ENGINEERING" in phrase_key:
                if "ELECTRICAL" in phrase_key:
                    return "BSEE"
                if "MECHANICAL" in phrase_key:
                    return "BSME"
                if "CIVIL" in phrase_key:
                    return "BSCE"
                return "BSE"
            return "BS"
        if "ART" in phrase_key:
            return "BA"
        if "FINE ARTS" in phrase_key:
            return "BFA"
        if "THEOLOGY" in phrase_key:
            return "B.Th."
    if "MASTER" in phrase_key:
        if "BUSINESS" in phrase_key and "ADMINISTRATION" in phrase_key:
            return "MBA"
        if "PUBLIC POLICY" in phrase_key:
            return "MPP"
        if "PUBLIC ADMINISTRATION" in phrase_key or "PUBLIC AFFAIRS" in phrase_key:
            return "MPA"
        if "SCIENCE" in phrase_key:
            if "ENGINEERING" in phrase_key:
                return "MSE"
            return "MS"
        if "ART" in phrase_key or "ARTS" in phrase_key:
            return "MA"
        if "EDUCATION" in phrase_key:
            return "MEd"
        if "FINE ARTS" in phrase_key:
            return "MFA"
        if "LIBERAL ARTS" in phrase_key:
            return "MLA"
        if "SOCIAL WORK" in phrase_key:
            return "MSW"
        if "LIBERAL STUDIES" in phrase_key:
            return "MLS"
        if "DIVINITY" in phrase_key:
            return "M.Div."
        if "THEOLOGY" in phrase_key:
            return "M.Th."
    if "JURIS" in phrase_key and ("DOCTOR" in phrase_key or "DOCTORATE" in phrase_key):
        return "JD"
    if "LAW" in phrase_key and "MASTER" in phrase_key:
        return "LLM"
    if "LAW" in phrase_key and "BACHELOR" in phrase_key:
        return "LLB"
    if "DOCTOR" in phrase_key and "OSTEOPATH" in phrase_key:
        return "DO"
    if "DOCTOR" in phrase_key and "MEDICINE" in phrase_key:
        return "MD"
    if "DOCTOR" in phrase_key and "PHILOSOPHY" in phrase_key:
        return "PhD"
    if "DOCTOR" in phrase_key and "EDUCATION" in phrase_key:
        return "EdD"
    if "DOCTOR" in phrase_key and "MINISTRY" in phrase_key:
        return "D.Min."
    if "DOCTOR" in phrase_key and "LAWS" in phrase_key:
        return "LLD"
    return ""


def _looks_like_institution(token: str) -> bool:
    lowered = token.lower()
    if not lowered:
        return False
    if "high school" in lowered:
        return False
    if lowered.startswith("department of "):
        return False
    keywords = (
        "university",
        "college",
        "academy",
        "institute",
        "seminary",
        "school",
    )
    return any(word in lowered for word in keywords)


def _clean_institution_token(token: str) -> str:
    cleaned = _normalize_education_token(token)
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(?:the\s+)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" ,")


def _institution_score(token: str) -> int:
    lowered = token.lower()
    score = 0
    if "university" in lowered:
        score = max(score, 6)
    if "college" in lowered:
        score = max(score, 5)
    if "academy" in lowered:
        score = max(score, 4)
    if "institute" in lowered:
        score = max(score, 4)
    if "law school" in lowered or "school of law" in lowered:
        score = max(score, 5)
    if "medical school" in lowered:
        score = max(score, 5)
    if "business school" in lowered:
        score = max(score, 4)
    if "school" in lowered:
        score = max(score, 2)
    return score or 1


def _format_degree_entry(segment: str) -> str:
    seg = html.unescape(segment.replace("•", " "))
    seg = re.sub(r"\s+", " ", seg).strip(" ;,\u2022")
    if not seg:
        return ""

    lowered = seg.lower()
    if "high school" in lowered or "secondary school" in lowered:
        return ""
    if "attended" in lowered and "degree" not in lowered:
        return ""
    if any(keyword in lowered for keyword in _EDU_ROLE_KEYWORDS) and "degree" not in lowered:
        return ""

    seg, year = _strip_education_year(seg)
    tokens = [part.strip() for part in seg.split(",") if part.strip()]

    degree_display = ""
    institution_candidates: List[str] = []
    leftover_tokens: List[str] = []

    for token in tokens:
        normalized = _normalize_education_token(token)
        if not normalized:
            continue
        degree_candidate = _canonical_degree_from_token(normalized)
        if degree_candidate and not degree_display:
            degree_display = degree_candidate
            continue
        if _looks_like_institution(normalized):
            institution = _clean_institution_token(normalized)
            if institution:
                institution_candidates.append(institution)
            continue
        leftover_tokens.append(normalized)

    if not degree_display:
        normalized_seg = _normalize_education_token(seg)
        degree_display = _canonical_degree_from_token(normalized_seg)

    if not institution_candidates:
        for token in leftover_tokens:
            match = re.search(r"(?:from|at|in)\s+(.+)", token, re.IGNORECASE)
            if match:
                candidate = _clean_institution_token(match.group(1))
                if candidate:
                    institution_candidates.append(candidate)
        if not institution_candidates:
            normalized_seg = _normalize_education_token(seg)
            if _looks_like_institution(normalized_seg):
                candidate = _clean_institution_token(normalized_seg)
                if candidate:
                    institution_candidates.append(candidate)
            else:
                match = re.search(r"(?:from|at|in)\s+(.+)", normalized_seg, re.IGNORECASE)
                if match:
                    candidate = _clean_institution_token(match.group(1))
                    if candidate:
                        institution_candidates.append(candidate)

    if not degree_display or not institution_candidates:
        return ""

    seen_inst = set()
    unique_candidates: List[str] = []
    for institution in institution_candidates:
        key = institution.lower()
        if key not in seen_inst:
            seen_inst.add(key)
            unique_candidates.append(institution)

    institution = max(unique_candidates, key=_institution_score)
    components = [degree_display, institution]
    if year:
        components.append(year)
    return ", ".join(components)


def _extract_degree_bullets(education_raw: str) -> List[str]:
    if not education_raw:
        return []
    bullets: List[str] = []
    seen: Set[str] = set()
    segments = re.split(r"[;\n]+", education_raw)
    for segment in segments:
        seg = segment.strip(" ;,\u2022")
        if not seg:
            continue
        lowered = seg.lower()
        if "high school" in lowered:
            continue
        if not any(keyword in lowered for keyword in DEGREE_KEYWORDS):
            continue
        formatted = _format_degree_entry(seg)
        if not formatted:
            continue
        key = formatted.lower()
        if key not in seen:
            seen.add(key)
            bullets.append(formatted)
    return bullets


def find_personal_info_row(first: str, last: str, bioguide: str = "") -> dict:
    if bioguide and bioguide in personal_info_index_by_bioguide:
        return personal_info_index_by_bioguide[bioguide]
    return personal_info_index.get(_fullnorm_key(first, last), {})


def find_holdings_row(first: str, last: str, bioguide: str = "") -> dict:
    if bioguide and bioguide in holdings_index_by_bioguide:
        return holdings_index_by_bioguide[bioguide]
    return holdings_index.get(_fullnorm_key(first, last), {})


def _parse_top_holdings(raw: str) -> List[dict]:
    entries: List[dict] = []
    if not raw:
        return entries
    for item in raw.split(";"):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            name, amt = part.split(":", 1)
        else:
            name, amt = part, ""
        name = name.strip()
        amt = amt.strip()
        if amt:
            amt = _clean_money(amt)
        entries.append({"name": name, "amount": amt})
    return entries


def _parse_top_sectors(raw: str) -> List[dict]:
    entries: List[dict] = []
    if not raw:
        return entries
    for item in raw.split(";"):
        part = item.strip()
        if not part:
            continue
        if ":" in part:
            name, count = part.split(":", 1)
        else:
            name, count = part, ""
        name = name.strip()
        count = count.strip()
        if not name:
            continue
        display_count = count
        if count:
            try:
                numeric = float(count.replace(",", ""))
            except ValueError:
                numeric = None
            if numeric is not None:
                if numeric.is_integer():
                    display_count = str(int(numeric))
                else:
                    display_count = f"{numeric:.2f}".rstrip("0").rstrip(".")
        entries.append({"sector": name, "trades": display_count})
    return entries


def find_summary_row(first: str, last: str, bioguide: str = "") -> dict:
    if bioguide and bioguide in summary_by_bioguide:
        return summary_by_bioguide[bioguide]
    return summary_lookup.get(_fullnorm_key(first, last), {})


def get_latest_trades(first: str, last: str, *, bioguide: str = "", limit: int = 5) -> Tuple[List[dict], str]:
    trades, quiver_url = get_quiver_trades(first, last, bioguide=bioguide, limit=limit)
    if trades:
        return trades, quiver_url

    if bioguide and bioguide in trades_index_by_bioguide:
        rows = trades_index_by_bioguide.get(bioguide, [])
    else:
        rows = trades_index.get(_fullnorm_key(first, last), [])

    rows = sorted(rows, key=lambda r: (r.get("transaction_date") or ""), reverse=True)
    fallback = []
    for r in rows[:limit]:
        range_raw = (
            r.get("Range")
            or r.get("range")
            or r.get("amount_range")
            or r.get("RangeDisplay")
            or ""
        )
        amount_raw = (
            r.get("Trade_Size_USD")
            or r.get("trade_size_usd")
            or r.get("Trade_Size_USD_num")
            or r.get("Amount")
            or r.get("amount")
            or ""
        )
        range_val = str(range_raw).strip()
        amount_val = str(amount_raw).strip()
        fallback.append({
            "ticker": (r.get("ticker") or "").strip(),
            "asset": "",
            "transaction": (r.get("transaction") or "").strip(),
            "traded": (r.get("transaction_date") or "").strip(),
            "filed": "",
            "amount": amount_val,
            "range": range_val or amount_val,
            "amount_range": range_val or amount_val,
        })
    return fallback, quiver_url


def _clean_money(sval: str) -> str:
    if not sval:
        return ""
    sval = sval.strip()
    if not sval:
        return ""

    lowered = sval.lower()
    if lowered in {"n/a", "na", "unknown"}:
        return sval

    while sval.startswith("$$"):
        sval = sval[1:]

    had_dollar = sval.startswith("$")
    core = sval[1:].strip() if had_dollar else sval
    core = core.lstrip("$").strip()
    if core.lower() in {"n/a", "na", "unknown"}:
        return "N/A"

    if not core:
        return ""

    suffix = core[-1:].upper()
    if suffix in {"M", "B", "K"}:
        return f"${core}"

    numeric_candidate = core.replace(",", "")
    try:
        value = Decimal(numeric_candidate)
    except InvalidOperation:
        return sval if not had_dollar else f"${core}"

    if value == value.to_integral():
        formatted = f"{int(value):,}"
    else:
        formatted = format(value, ",f").rstrip("0").rstrip(".")

    return f"${formatted}"


def build_overlay_payload(bioguide_id: str, *, filename: Optional[str], face_score: float,
                          full_sim: float, last_sim: float, combined_score: float) -> dict:
    # Check cache first - use bioguide_id as key since it's stable
    cache_key = bioguide_id or "unknown"
    if cache_key in _OVERLAY_CACHE:
        cached_entry = _OVERLAY_CACHE[cache_key]
        cache_age = time.time() - cached_entry.get("timestamp", 0)
        if cache_age < _OVERLAY_CACHE_TTL:
            # Return cached payload with updated timestamp
            payload = cached_entry["payload"].copy()
            payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload["scores"]["face"] = float(face_score)
            payload["scores"]["combined"] = float(combined_score)
            return payload
    
    # Cache miss - build payload from scratch
    if not filename and bioguide_id:
        filename = get_primary_filename(bioguide_id)
    filename = filename or ""
    info = info_for_person(bioguide_id, filename)
    first = info.get("first_name", "")
    last = info.get("last_name", "")
    title = info.get("title", "")

    display_name = f"{first} {last}".strip()
    if title:
        party_full = (info.get("party") or "").strip().lower()
        state_full = (info.get("state") or "").strip()
        party_abbrv = party_map.get(party_full, (party_full[:1].upper() if party_full else ""))
        state_abbrv = state_abbr.get(state_full, state_full if len(state_full) == 2 else "")
        if party_abbrv and state_abbrv:
            display_name = f"{display_name}\n{title} {party_abbrv}-{state_abbrv}".strip()

    summary = find_summary_row(first, last, bioguide=bioguide_id)
    holdings_row = find_holdings_row(first, last, bioguide=bioguide_id)
    personal_info = find_personal_info_row(first, last, bioguide=bioguide_id)
    committees = _committees_for_member(first, last, title, bioguide=bioguide_id)
    trades, quiver_url = get_latest_trades(first, last, bioguide=bioguide_id)

    donors = []
    top_donors = (summary.get("top_contributors") or "").strip()
    for entry in (top_donors.split(";") if top_donors else [])[:5]:
        if "(" in entry and ")" in entry:
            name_d, amt = entry.rsplit("(", 1)
            donors.append({"name": name_d.strip(), "amount": _money_once(amt)})

    industries = []
    top_industries = (summary.get("top_industries") or "").strip()
    for entry in (top_industries.split(";") if top_industries else [])[:5]:
        if "(" in entry and ")" in entry:
            ind, amt = entry.rsplit("(", 1)
            industries.append({"name": ind.strip(), "amount": _money_once(amt)})

    networth_current = ""
    normalized_networth = ""

    holdings_list: List[dict] = []
    traded_sectors: List[dict] = []
    if holdings_row:
        holdings_est = (holdings_row.get("net_worth_estimate") or "").strip()
        holdings_norm = (holdings_row.get("normalized_net_worth") or "").strip()
        holdings_entries = _parse_top_holdings((holdings_row.get("top_holdings") or "").strip())
        sectors_entries = _parse_top_sectors((holdings_row.get("top_traded_sectors") or "").strip())
        if holdings_entries:
            holdings_list = holdings_entries
        if sectors_entries:
            traded_sectors = sectors_entries

        if holdings_norm:
            clean_norm = _clean_money(holdings_norm)
            if clean_norm:
                normalized_networth = clean_norm
                if not networth_current:
                    networth_current = clean_norm

        if holdings_est:
            formatted_est = _clean_money(holdings_est)
            if formatted_est:
                if not networth_current:
                    networth_current = formatted_est
                if not normalized_networth:
                    normalized_networth = formatted_est

    if not networth_current and normalized_networth:
        networth_current = normalized_networth


    dob_raw = ""
    if personal_info:
        dob_raw = (personal_info.get("date_of_birth") or personal_info.get("dob") or "").strip()

    age_val = ""
    if personal_info:
        age_val = _normalize_age(personal_info.get("age"))

    age_pretty_val = ""
    if personal_info:
        age_pretty_val = _clean_age_pretty(personal_info.get("age_pretty"))
    if not age_pretty_val and age_val:
        age_pretty_val = age_val

    tenure_val = ""
    if personal_info:
        candidate_tenure = personal_info.get("tenure_pretty")
        if candidate_tenure:
            tenure_val = str(candidate_tenure).strip()

    dob_display = _format_birth_date(dob_raw, age_val)

    birth_place = ""
    if personal_info:
        birth_place = _format_birth_place(
            personal_info.get("profile_birthplace") or personal_info.get("birthplace") or ""
        )

    military_item = ""
    if personal_info:
        has_service = (personal_info.get("has_military_service") or "").strip().lower()
        if has_service in {"yes", "y", "true", "1"}:
            military_item = (personal_info.get("military_service") or "").strip()
            if military_item:
                military_item = re.sub(r"\s*;\s*", "; ", military_item)
                military_item = re.sub(r"\s*,\s*", ", ", military_item)
                military_item = re.sub(r"\s{2,}", " ", military_item).strip()

    education_items: List[str] = []
    if personal_info:
        education_items = _extract_degree_bullets(personal_info.get("profile_education") or "")

    assumed_office_raw = ""
    if personal_info:
        assumed_office_raw = (personal_info.get("assumed_office") or "").strip()
    assumed_display = _format_birth_date(assumed_office_raw, "")

    education_entries = education_items

    bio_items: List[dict] = []
    if birth_place:
        bio_items.append({"label": "Birthplace", "value": birth_place})
    if dob_display:
        bio_items.append({"label": "Date of Birth", "value": dob_display})
    if age_pretty_val:
        bio_items.append({"label": "Age", "value": age_pretty_val})
    if tenure_val:
        bio_items.append({"label": "Time in Office", "value": tenure_val})
    if military_item:
        bio_items.append({"label": "Military Service", "value": military_item})
    if education_entries:
        first_degree = education_entries[0] if education_entries else ""
        bio_items.append({"label": "Education", "value": first_degree})
        for entry in education_entries[1:]:
            bio_items.append({"text": f"    {entry}"})

    overlay = {
        "name": display_name,
        "image": os.path.basename(filename) if filename else "",
        "bioguide_id": bioguide_id or info.get("bioguide_id", ""),
        "dob": dob_display,
        "age": age_val or "",
        "age_pretty": age_pretty_val or "",
        "tenure_pretty": tenure_val,
        "bio_card": {
            "items": bio_items
        },
        "committees": committees,
        "latest_trades": trades,
        "quiver_url": quiver_url,
        "networth_current": networth_current or "N/A",
        "normalized_net_worth": normalized_networth,
        "top_holdings": holdings_list,
        "top_traded_sectors": traded_sectors,
        "top_donors": donors,
        "top_industries": industries,
        "periods": {
            "contributors": (summary.get("contributors_period") or "").strip(),
            "industries": (summary.get("industries_period") or "").strip(),
        },
        "source_url": (summary.get("slug_url") or "").strip(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scores": {
            "face": float(face_score),
            "full": float(full_sim),
            "last": float(last_sim),
            "combined": float(combined_score),
        }
    }
    
    # Cache the payload for future use
    cache_key = bioguide_id or "unknown"
    _OVERLAY_CACHE[cache_key] = {
        "payload": overlay,
        "timestamp": time.time()
    }

    return overlay


def _normalize_recent_trade_entry(row: dict) -> Optional[dict]:
    """Normalize a single trade entry from congresstrades_clean.csv"""
    if not row:
        return None
    ticker = (row.get("ticker") or "").strip().upper()
    transaction = (row.get("transaction") or "").strip()
    trade_date = (row.get("transaction_date") or "").strip()
    if not (ticker and transaction and trade_date):
        return None
    try:
        sort_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    except ValueError:
        return None
    display_name = (row.get("representative") or "").strip()
    if not display_name:
        first = (row.get("first_name") or "").strip()
        last = (row.get("last_name") or "").strip()
        display_name = f"{first} {last}".strip()
    party = (row.get("party") or "").strip()
    chamber_raw = (row.get("house") or "").strip()
    chamber_lower = chamber_raw.lower()
    if chamber_lower.startswith("rep"):
        chamber_display = "House"
    elif chamber_lower.startswith("sen"):
        chamber_display = "Senate"
    else:
        chamber_display = chamber_raw
    amount_range = row.get("range") or ""
    amount_value = row.get("amount") or ""
    report_date = (row.get("report_date") or "").strip()
    description = (row.get("description") or "").strip()
    item = {
        "name": display_name,
        "party": party,
        "party_display": party[:1].upper() if party else "",
        "chamber": chamber_display,
        "ticker": ticker,
        "transaction": transaction,
        "amount_range": str(amount_range),
        "amount": str(amount_value),
        "company_clean": description,
        "asset": description,
        "description": description,
        "transaction_date": trade_date,
        "traded": trade_date,
        "report_date": report_date,
    }
    item["_sort_ts"] = sort_dt.timestamp()
    return item


def _load_recent_trades(limit: int = 50) -> List[dict]:
    """Load and cache recent trades from congresstrades_clean.csv"""
    global _RECENT_TRADES_CACHE, _RECENT_TRADES_MTIME
    path = CONGRESS_TRADES_CLEAN_CSV
    if not path or not os.path.exists(path):
        return []
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    if _RECENT_TRADES_CACHE and _RECENT_TRADES_MTIME == mtime and len(_RECENT_TRADES_CACHE) >= limit:
        return _RECENT_TRADES_CACHE[:limit]
    records: List[dict] = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                trade = _normalize_recent_trade_entry(row)
                if trade:
                    records.append(trade)
    except Exception as exc:
        logging.debug("Failed to load recent trades snapshot: %s", exc)
        return _RECENT_TRADES_CACHE[:limit] if _RECENT_TRADES_CACHE else []

    if not records:
        _RECENT_TRADES_CACHE = []
        _RECENT_TRADES_MTIME = mtime
        return []

    records.sort(key=lambda rec: (rec["_sort_ts"], rec.get("transaction_date", "")), reverse=True)
    for rec in records:
        rec.pop("_sort_ts", None)
    _RECENT_TRADES_CACHE = records
    _RECENT_TRADES_MTIME = mtime
    return records[:limit]


def _role_from_title(title: str) -> str:
    """Map free-form title to a coarse role bucket: 'senate', 'house', or ''."""
    t = (title or "").strip().lower()
    if not t:
        return ""
    if t.startswith("sen"):
        return "senate"
    if t.startswith("rep") or t.startswith("del"):
        return "house"
    return ""


def _preferred_networth_str_for(bioguide: str, first: str, last: str) -> str:
    """Choose the best available net worth string for a person using holdings data."""
    holdings = find_holdings_row(first, last, bioguide=bioguide)

    norm = ""
    curr = ""

    if holdings:
        h_norm = (holdings.get("normalized_net_worth") or "").strip()
        if h_norm:
            norm = _clean_money(h_norm)

        for key in ("current_net_worth", "net_worth_estimate"):
            raw = (holdings.get(key) or "").strip()
            if not raw:
                continue
            cleaned = _clean_money(raw)
            if cleaned:
                if not curr:
                    curr = cleaned
                if not norm:
                    norm = cleaned

    disp = norm or curr
    return disp or "N/A"


def build_networth_ticker_payload(preferred_first: Optional[str] = None) -> dict:
    """Build a payload for overlay_server to render a scrolling ticker of
    senators and representatives with their net worth.

    preferred_first optionally sets which chamber's section appears first.

    Returns a dict that includes a 'ticker' object alongside a generic title in 'name'.
    """
    # Ensure indices are loaded
    # name_map / bioguide_to_info are populated in init()
    senate_items = []
    house_items = []

    # Iterate through known people (by bioguide where available for stability)
    for bid, info in bioguide_to_info.items():
        first = (info.get("first_name") or "").strip()
        last = (info.get("last_name") or "").strip()
        title = (info.get("title") or "").strip()
        role = _role_from_title(title)
        if not first and not last:
            continue
        net_str = _preferred_networth_str_for(bid, first, last)
        party = (info.get("party") or "").strip()
        state = (info.get("state") or "").strip()
        party_label = party_map.get(party.lower(), party[:1].upper() if party else "")
        party_short = party_label or ""
        state_upper = state.upper() if state else ""
        suffix_parts = [p for p in (party_short, state_upper) if p]
        suffix = "-".join(suffix_parts)
        item = {
            "name": f"{first} {last}".strip(),
            "networth": net_str,
            "bioguide_id": bid,
            "party": party,
            "state": state_upper,
            "suffix": suffix,
            "party_display": party_short,
            "state_display": state_upper,
        }
        if role == "senate":
            senate_items.append((last.lower(), item))
        elif role == "house":
            house_items.append((last.lower(), item))

    # Fallback pass: include entries without bioguide mapping if present in name_map
    for fn, info in name_map.items():
        bid = (info.get("bioguide_id") or "").strip()
        if bid:
            continue  # already handled above
        first = (info.get("first_name") or "").strip()
        last = (info.get("last_name") or "").strip()
        title = (info.get("title") or "").strip()
        role = _role_from_title(title)
        if not first and not last:
            continue
        net_str = _preferred_networth_str_for("", first, last)
        party = (info.get("party") or "").strip()
        state = (info.get("state") or "").strip()
        party_label = party_map.get(party.lower(), party[:1].upper() if party else "")
        party_short = party_label or ""
        state_upper = state.upper() if state else ""
        suffix_parts = [p for p in (party_short, state_upper) if p]
        suffix = "-".join(suffix_parts)
        item = {
            "name": f"{first} {last}".strip(),
            "networth": net_str,
            "bioguide_id": "",
            "party": party,
            "state": state_upper,
            "suffix": suffix,
            "party_display": party_short,
            "state_display": state_upper,
        }
        if role == "senate":
            senate_items.append((last.lower(), item))
        elif role == "house":
            house_items.append((last.lower(), item))

    # Sort by net worth descending

    def parse_networth_value(val: str) -> float:
        if not val:
            return 0.0
        s = str(val).strip().lower()
        if not s or s in {"n/a", "na", "unknown", "--"}:
            return 0.0
        # Quick numeric path
        try:
            return float(s.replace(',', ''))
        except ValueError:
            pass
        num_pat = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
        nums = re.findall(num_pat, s)
        if not nums:
            return 0.0
        values = [Decimal(n.replace(',', '')) for n in nums]
        range_markers = ('between', ' - ', '–', ' to ')
        if any(marker in s for marker in range_markers) and len(values) >= 2:
            return float(sum(values) / len(values))
        low_markers = ('under', '$0 -', '$0–', '<$', 'less than', 'up to')
        if any(marker in s for marker in low_markers):
            return float(values[0])
        return float(values[-1])

    senate_items.sort(key=lambda t: parse_networth_value(t[1].get("networth")), reverse=True)
    house_items.sort(key=lambda t: parse_networth_value(t[1].get("networth")), reverse=True)

    senate = []
    for idx, (_, it) in enumerate(senate_items, 1):
        entry = dict(it)
        entry["rank"] = idx
        senate.append(entry)

    house = []
    for idx, (_, it) in enumerate(house_items, 1):
        entry = dict(it)
        entry["rank"] = idx
        house.append(entry)

    sections = {
        "senate": {"key": "senate", "title": "Senators", "meta": "Est. Net Worth", "items": senate},
        "house": {"key": "house", "title": "Representatives", "meta": "Est. Net Worth", "items": house},
    }

    trades_section = None
    recent_trades = _load_recent_trades(50)
    if recent_trades:
        trades_section = {
            "key": "recent_trades",
            "title": "Recent Trades",
            "meta": "Latest 50 trades",
            "items": recent_trades,
        }

    preferred_norm = (preferred_first or "").strip().lower()
    if preferred_norm == "house":
        section_order = ["house", "senate"]
    else:
        # Default to senators first; treat unknown preference the same as senate.
        section_order = ["senate", "house"]

    ordered_sections: List[dict] = []
    first_key, second_key = section_order
    ordered_sections.append(sections[first_key])
    if trades_section:
        ordered_sections.append(trades_section)
    ordered_sections.append(sections[second_key])

    ticker = {
        "title": "",
        "sections": ordered_sections,
    }

    return {
        "name": "",
        "ticker": ticker,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_trades_ticker_payload(limit: int = 50) -> dict:
    """Build a payload for overlay_server to render a scrolling ticker of recent trades only.

    Args:
        limit: Maximum number of trades to show (default 50)

    Returns:
        A dict with a 'ticker' object containing only the recent trades section
    """
    recent_trades = _load_recent_trades(limit)

    if not recent_trades:
        # If no trades, return empty ticker or fall back to senators
        logging.warning("No recent trades found, creating empty trades ticker")

    trades_section = {
        "key": "recent_trades",
        "title": "Recent Congressional Trades",
        "meta": f"Latest {len(recent_trades)} trades" if recent_trades else "No recent trades",
        "items": recent_trades,
    }

    ticker = {
        "title": "",
        "sections": [trades_section],
    }

    return {
        "name": "",
        "ticker": ticker,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def atomic_write_json(obj, path, tmp):
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def write_overlay(payload: dict) -> None:
    if not OVERLAY_JSON:
        return
    try:
        atomic_write_json(payload, OVERLAY_JSON, OVERLAY_JSON_TMP)
    except Exception:
        logging.exception("Failed to write overlay JSON")


def write_heartbeat(frame_no: int, model_tag: str) -> None:
    if not HEARTBEAT_JSON:
        return
    hb = {
        "frame": int(frame_no),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_tag": model_tag or "",
    }
    try:
        atomic_write_json(hb, HEARTBEAT_JSON, HEARTBEAT_JSON_TMP)
    except Exception:
        logging.exception("Failed to write heartbeat JSON")


def _normalize_threshold(val: float) -> float:
    try:
        if val > 1.0:
            return float(val) / 100.0
        return float(val)
    except Exception:
        return 0.85


def _ensure_save_dir() -> str:
    if CONFIG is None:
        return PENDING_REVIEW_DIR
    override = (CONFIG.get("save_face_dir", "") or "").strip()
    use_dir = override if override else PENDING_REVIEW_DIR
    os.makedirs(use_dir, exist_ok=True)
    return use_dir


def _score_from_filename(path: Path) -> float:
    stem = path.stem
    if "__" in stem:
        parts = stem.rsplit("__", 2)
        if len(parts) >= 3:
            try:
                return float(parts[-2])
            except ValueError:
                pass
    parts = stem.rsplit("_", 1)
    if len(parts) >= 2:
        candidate = parts[-1]
        try:
            value = float(candidate)
            if value > 1.0:
                return value / 100.0
            return value
        except ValueError:
            pass
    return 0.0


def _list_pending_entries(base: str, directory: Path) -> List[Tuple[float, float, Path]]:
    entries: List[Tuple[float, float, Path]] = []
    if not directory.exists():
        return entries

    seen: Set[Path] = set()
    patterns = (f"{base}__*.jpg", f"{base}_*.jpg")
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            score = _score_from_filename(path)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            entries.append((score, mtime, path))

    entries.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return entries


def _prune_member_entries(base: str, directory: Path, max_per: int) -> None:
    max_per = max(0, int(max_per))
    entries = _list_pending_entries(base, directory)

    if max_per == 0:
        to_remove = entries
    elif len(entries) <= max_per:
        to_remove: List[Tuple[float, float, Path]] = []
    else:
        to_remove = entries[max_per:]

    for _, _, path in to_remove:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception:
            logging.warning("Failed to remove pending review image %s", path, exc_info=True)

    remaining = _list_pending_entries(base, directory)
    if remaining:
        SAVED_COUNT_BY_PERSON[base] = min(len(remaining), max_per if max_per else len(remaining))
    else:
        SAVED_COUNT_BY_PERSON.pop(base, None)


def save_pending_review(
    image_bgr: np.ndarray,
    person_fn: str,
    bioguide_id: str,
    face_score: float,
    combined_score: float,
) -> None:
    """Save face crops for under-represented members to pending_review/ for human verification.

    Saves faces where:
    - face_score is in range [min_confidence, max_confidence] (default 0.70-0.85)
    - Member has fewer than max_db_images in the face database (default <10)
    - Score is sufficiently different from other saves this session (min_score_spread)
    - Haven't exceeded max saves per person per session (max_per_session)

    This captures diverse angles/expressions while avoiding:
    - Low confidence (potential false positives)
    - High confidence (near-duplicates of existing DB images)
    - Redundant captures from the same stream/session
    """
    if CONFIG is None or image_bgr is None or image_bgr.size == 0:
        return
    if not CONFIG.get("save_face_hits", False):
        return

    # Get confidence thresholds
    min_conf = float(CONFIG.get("save_face_min_confidence", 0.70))
    max_conf = float(CONFIG.get("save_face_max_confidence", 0.85))
    face_sim = float(face_score)

    # Check if face_score is in the target range (diverse but confident)
    if face_sim < min_conf or face_sim > max_conf:
        return

    # Check if member needs more images in DB
    max_db_images = int(CONFIG.get("save_face_max_db_images", 10))
    if bioguide_id and bioguide_id in bioguide_to_filenames:
        current_db_count = len(bioguide_to_filenames[bioguide_id])
        if current_db_count >= max_db_images:
            return  # Member already has enough images
    elif not bioguide_id:
        # No bioguide_id means we can't verify DB count, skip to be safe
        return

    out_dir = Path(_ensure_save_dir())
    base_raw = os.path.splitext(os.path.basename(person_fn))[0]
    base = re.sub(r"[^\w.-]", "_", base_raw.strip()) or "unknown"

    # Use bioguide_id as the grouping key so all variants of the same person share limits
    review_key_raw = bioguide_id.strip() or base
    review_key = re.sub(r"[^\w.-]", "_", review_key_raw) or "unknown"

    max_per = max(1, int(CONFIG.get("save_face_max_per_person", 5)))

    # Session-based limiting: avoid saving too many faces of same person from same stream
    max_per_session = int(CONFIG.get("save_face_max_per_session", 3))
    min_score_spread = float(CONFIG.get("save_face_min_score_spread", 0.05))

    session_scores = SESSION_SAVED_SCORES.get(review_key, [])[:]  # Copy to avoid mutation issues

    # Check if this score is too close to any existing session score
    too_close = False
    for existing_score in session_scores:
        if abs(face_sim - existing_score) < min_score_spread:
            too_close = True
            break

    if too_close:
        return  # Too similar to an existing save from this session

    # If we haven't hit the session limit, we can add freely
    # If we HAVE hit the limit, check if this score improves diversity (spread)
    replace_score = None
    if len(session_scores) >= max_per_session:
        # Calculate current spread (max - min)
        current_spread = max(session_scores) - min(session_scores) if len(session_scores) > 1 else 0

        # Try replacing each existing score and see which gives best spread
        best_spread = current_spread
        best_replace = None

        for i, existing in enumerate(session_scores):
            # What if we replaced this score with the new one?
            test_scores = [s for j, s in enumerate(session_scores) if j != i] + [face_sim]
            test_spread = max(test_scores) - min(test_scores)

            if test_spread > best_spread:
                best_spread = test_spread
                best_replace = existing

        if best_replace is None:
            # New score doesn't improve spread, skip it
            return

        replace_score = best_replace
        logging.info(
            "Replacing session score %.3f with %.3f for %s (spread: %.3f -> %.3f)",
            replace_score, face_sim, review_key, current_spread, best_spread
        )

    _prune_member_entries(review_key, out_dir, max_per)
    existing_entries = _list_pending_entries(review_key, out_dir)

    # Check if we have room for another entry
    if len(existing_entries) >= max_per:
        # Only replace if new face_score is better than worst existing
        worst_score = existing_entries[-1][0] if existing_entries else 0.0
        if face_sim <= worst_score:
            return

    try:
        ok, enc = cv2.imencode(".jpg", image_bgr)
        if not ok:
            return
        data = enc.tobytes()
        h = hashlib.md5(data).hexdigest()
        if h in SAVED_HASHES:
            return
        # Include bioguide in filename for easier review
        filename = f"{review_key}__{base}__{face_sim:.3f}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        path = out_dir / filename
        with open(path, "wb") as fh:
            fh.write(data)
        SAVED_HASHES.add(h)
        # Track this score for session-based limiting
        if review_key not in SESSION_SAVED_SCORES:
            SESSION_SAVED_SCORES[review_key] = []
        # If we're replacing a score, remove the old one first
        if replace_score is not None and replace_score in SESSION_SAVED_SCORES[review_key]:
            SESSION_SAVED_SCORES[review_key].remove(replace_score)
        SESSION_SAVED_SCORES[review_key].append(face_sim)
        logging.info(
            "Saved pending review face: %s (face=%.3f, db_count=%d, session_count=%d, spread=%.3f)",
            review_key, face_sim, len(bioguide_to_filenames.get(bioguide_id, [])),
            len(SESSION_SAVED_SCORES[review_key]),
            max(SESSION_SAVED_SCORES[review_key]) - min(SESSION_SAVED_SCORES[review_key]) if len(SESSION_SAVED_SCORES[review_key]) > 1 else 0
        )
    except Exception:
        logging.exception("Failed to save pending review crop")
        return

    _prune_member_entries(review_key, out_dir, max_per)


# Keep old function name as alias for backwards compatibility
save_high_confidence = save_pending_review
