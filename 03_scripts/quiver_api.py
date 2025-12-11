"""Reusable helpers for calling QuiverQuant endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import requests

API_URL = "https://api.quiverquant.com/beta/bulk/congresstrading"
ENV_TOKEN_KEY = "QUIVER_API_TOKEN"
DEFAULT_TOKEN_FILE = "quiver_token.txt"


class QuiverQuantError(RuntimeError):
    """Raised when an API call to QuiverQuant fails."""


def load_token(token_file: Optional[str] = None) -> str:
    """Resolve the API token from env var or a local file."""

    env_token = os.getenv(ENV_TOKEN_KEY)
    if env_token:
        return env_token.strip()

    candidate_paths: Iterable[Path]
    if token_file is not None:
        candidate_paths = (Path(token_file).expanduser(),)
    else:
        script_dir = Path(__file__).resolve().parent
        candidate_paths = (
            Path.cwd() / DEFAULT_TOKEN_FILE,
            script_dir.parent / "01_data" / DEFAULT_TOKEN_FILE,
            script_dir.parent / DEFAULT_TOKEN_FILE,
            script_dir / DEFAULT_TOKEN_FILE,
        )

    for path in candidate_paths:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()

    if token_file is not None:
        raise FileNotFoundError(f"Token file not found: {token_file}")

    raise ValueError(
        "No API token found. Set QUIVER_API_TOKEN or create quiver_token.txt."
    )


def fetch_congress_trading(
    bioguide_id: str,
    *,
    session: Optional[requests.Session] = None,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> list[dict]:
    """Fetch transactions for a single bioguide id."""

    if not bioguide_id:
        raise ValueError("bioguide_id must be a non-empty string")

    auth_token = token or load_token()
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }
    params = {"bioguide_id": bioguide_id}

    client = session or requests
    try:
        response = client.get(API_URL, headers=headers, params=params, timeout=timeout)
    except requests.RequestException as exc:  # network or transport error
        raise QuiverQuantError(f"Request failed: {exc}") from exc

    if response.status_code // 100 != 2:
        raise QuiverQuantError(
            f"API error {response.status_code}: {response.text.strip()}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise QuiverQuantError("Response payload is not valid JSON") from exc

    if not isinstance(payload, list):
        raise QuiverQuantError(
            "Unexpected payload type from API, expected a JSON array."
        )

    return payload
