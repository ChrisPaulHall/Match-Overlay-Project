#!/usr/bin/env python3
"""Generate normalized committee metadata from committees-current.yaml.

This script treats the YAML committee roster as the authoritative source and
emits a flat mapping of committee and subcommittee codes to their canonical
names. Downstream processes can join on the `code` field to replace the CSV
lookups that were previously hand-maintained.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import yaml


@dataclass
class CommitteeRecord:
    code: str
    parent_code: Optional[str]
    chamber: str
    committee_name: str
    subcommittee_name: str
    display_name: str
    thomas_id: str
    subcommittee_id: str
    house_committee_id: str
    senate_committee_id: str
    joint_committee_id: str
    url: str
    minority_url: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "parent_code": self.parent_code or "",
            "chamber": self.chamber,
            "committee_name": self.committee_name,
            "subcommittee_name": self.subcommittee_name,
            "display_name": self.display_name,
            "thomas_id": self.thomas_id,
            "subcommittee_id": self.subcommittee_id,
            "house_committee_id": self.house_committee_id,
            "senate_committee_id": self.senate_committee_id,
            "joint_committee_id": self.joint_committee_id,
            "url": self.url,
            "minority_url": self.minority_url,
        }


def normalize_committees(yaml_path: Path) -> List[CommitteeRecord]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []

    records: List[CommitteeRecord] = []
    for entry in data:
        base_code = (entry.get("thomas_id") or "").strip()
        if not base_code:
            continue

        chamber = (entry.get("type") or "").strip()
        committee_name = (entry.get("name") or "").strip()
        url = (entry.get("url") or "").strip()
        minority_url = (entry.get("minority_url") or "").strip()
        house_committee_id = (entry.get("house_committee_id") or "").strip()
        senate_committee_id = (entry.get("senate_committee_id") or "").strip()
        joint_committee_id = (entry.get("joint_committee_id") or "").strip()

        base_record = CommitteeRecord(
            code=base_code,
            parent_code=None,
            chamber=chamber,
            committee_name=committee_name,
            subcommittee_name="",
            display_name=committee_name,
            thomas_id=base_code,
            subcommittee_id="",
            house_committee_id=house_committee_id,
            senate_committee_id=senate_committee_id,
            joint_committee_id=joint_committee_id,
            url=url,
            minority_url=minority_url,
        )
        records.append(base_record)

        for sub in entry.get("subcommittees") or []:
            raw_sub_id = (sub.get("thomas_id") or "").strip()
            if raw_sub_id.isdigit() and len(raw_sub_id) < 2:
                raw_sub_id = raw_sub_id.zfill(2)
            sub_code = f"{base_code}{raw_sub_id}" if raw_sub_id else base_code
            subcommittee_name = (sub.get("name") or "").strip()

            display_name = (
                f"{committee_name} - {subcommittee_name}"
                if subcommittee_name
                else committee_name
            )

            sub_record = CommitteeRecord(
                code=sub_code,
                parent_code=base_code,
                chamber=chamber,
                committee_name=committee_name,
                subcommittee_name=subcommittee_name,
                display_name=display_name,
                thomas_id=base_code,
                subcommittee_id=raw_sub_id,
                house_committee_id=house_committee_id,
                senate_committee_id=senate_committee_id,
                joint_committee_id=joint_committee_id,
                url=url,
                minority_url=minority_url,
            )
            records.append(sub_record)

    records.sort(key=lambda rec: (rec.code, rec.subcommittee_name))
    return records


def write_json(records: Iterable[CommitteeRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    as_dicts = [record.to_dict() for record in records]
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(as_dicts, fh, indent=2, ensure_ascii=False)


def write_csv(records: Iterable[CommitteeRecord], output_path: Path) -> None:
    records = list(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "code",
        "parent_code",
        "chamber",
        "committee_name",
        "subcommittee_name",
        "display_name",
        "thomas_id",
        "subcommittee_id",
        "house_committee_id",
        "senate_committee_id",
        "joint_committee_id",
        "url",
        "minority_url",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize committee metadata from committees-current.yaml."
    )
    parser.add_argument(
        "--committees-yaml",
        default="01_data/committees-current.yaml",
        help="Path to committees-current.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--json-out",
        default="01_data/committee_metadata.json",
        help="Path for JSON output (default: %(default)s)",
    )
    parser.add_argument(
        "--csv-out",
        default="01_data/committee_metadata.csv",
        help="Path for CSV output (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    committees_yaml = Path(args.committees_yaml).expanduser().resolve()
    json_output = Path(args.json_out).expanduser().resolve()
    csv_output = Path(args.csv_out).expanduser().resolve()

    if not committees_yaml.exists():
        print(f"Input YAML not found: {committees_yaml}")
        return 1

    records = normalize_committees(committees_yaml)
    write_json(records, json_output)
    write_csv(records, csv_output)

    print(
        f"Wrote {len(records)} committee records to {json_output} "
        f"and {csv_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
