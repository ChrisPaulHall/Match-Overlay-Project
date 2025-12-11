#!/usr/bin/env python3
"""Flatten committee-membership-current.yaml with canonical committee metadata.

This script pairs each roster entry in committee-membership-current.yaml with the
normalized committee metadata emitted by normalize_committees.py. The resulting
CSV/JSON outputs contain canonical committee and subcommittee names, making it
easy for downstream jobs to replace hand-maintained CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from normalize_committees import CommitteeRecord, normalize_committees


def split_name(full_name: str) -> tuple[str, str]:
    """Split a full name into first and last components."""
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


def load_membership(yaml_path: Path) -> dict[str, list[dict]]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def flatten_membership(
    membership_yaml: Path,
    committees_yaml: Path,
) -> List[dict]:
    membership = load_membership(membership_yaml)
    committee_records = {rec.code: rec for rec in normalize_committees(committees_yaml)}

    rows: List[dict] = []
    missing_codes: set[str] = set()

    for code, members in membership.items():
        committee_info: Optional[CommitteeRecord] = committee_records.get(code)
        if committee_info is None:
            missing_codes.add(code)
            continue

        for entry in members or []:
            bioguide = (entry.get("bioguide") or "").strip()
            if not bioguide:
                continue

            first_name, last_name = split_name(entry.get("name"))

            row = {
                "bioguide": bioguide,
                "bioguide_id": bioguide,
                "first_name": first_name,
                "last_name": last_name,
                "name": (entry.get("name") or "").strip(),
                "party": (entry.get("party") or "").strip(),
                "title": (entry.get("title") or "").strip(),
                "rank": entry.get("rank"),
                "committee_code": committee_info.code,
                "committee_name": committee_info.committee_name,
                "committee": committee_info.committee_name,
                "subcommittee_code": committee_info.code
                if committee_info.parent_code
                else "",
                "subcommittee_name": committee_info.subcommittee_name,
                "subcommittee": committee_info.subcommittee_name,
                "display_name": committee_info.display_name,
                "chamber": committee_info.chamber,
                "thomas_id": committee_info.thomas_id,
                "subcommittee_id": committee_info.subcommittee_id,
                "parent_code": committee_info.parent_code or "",
            }
            rows.append(row)

    if missing_codes:
        missing_str = ", ".join(sorted(missing_codes))
        print(f"Warning: {len(missing_codes)} committee codes missing from metadata: {missing_str}")

    rows.sort(key=lambda r: (r["committee_code"], r["subcommittee_code"], r["rank"] or 999, r["bioguide"]))
    return rows


def write_json(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows generated from membership YAML.")

    fieldnames = [
        "first_name",
        "last_name",
        "committee",
        "subcommittee",
        "bioguide_id",
        "bioguide",
        "name",
        "party",
        "title",
        "rank",
        "committee_code",
        "committee_name",
        "subcommittee_code",
        "subcommittee_name",
        "display_name",
        "chamber",
        "thomas_id",
        "subcommittee_id",
        "parent_code",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten committee membership YAML using canonical committee metadata."
    )
    parser.add_argument(
        "--membership-yaml",
        default="01_data/committee-membership-current.yaml",
        help="Path to committee membership YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--committees-yaml",
        default="01_data/committees-current.yaml",
        help="Path to committees metadata YAML (default: %(default)s)",
    )
    parser.add_argument(
        "--csv-out",
        default="01_data/committee_membership_flat.csv",
        help="Path for CSV output (default: %(default)s)",
    )
    parser.add_argument(
        "--json-out",
        default="01_data/committee_membership_flat.json",
        help="Path for JSON output (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    membership_path = Path(args.membership_yaml).expanduser().resolve()
    committees_path = Path(args.committees_yaml).expanduser().resolve()
    json_output = Path(args.json_out).expanduser().resolve()
    csv_output = Path(args.csv_out).expanduser().resolve()

    if not membership_path.exists():
        print(f"Membership YAML not found: {membership_path}")
        return 1
    if not committees_path.exists():
        print(f"Committees YAML not found: {committees_path}")
        return 1

    rows = flatten_membership(membership_path, committees_path)
    write_json(rows, json_output)
    write_csv(rows, csv_output)

    print(
        f"Wrote {len(rows)} membership rows to {json_output} "
        f"and {csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
