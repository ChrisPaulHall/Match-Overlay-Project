#!/usr/bin/env python3
"""
Compare two members_donor_summary CSV files.
Usage: python compare_donor_files.py

Compares:
- 00members_donor_summary.csv (newer/scraped version)
- members_donor_summary.csv (production version)

Checks for:
- Missing members
- Empty contributor/industry data
- Period mismatches
- Significant data changes
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent.parent / "01_data"
NEWER_FILE = BASE_DIR / "00members_donor_summary.csv"
OLDER_FILE = BASE_DIR / "members_donor_summary.csv"


def load_csv(filepath):
    """Load CSV file and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def describe_file(path: Path, label: str) -> str:
    """Return a human-friendly description with last-modified timestamp."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        ts = mtime.strftime("%Y-%m-%d %H:%M")
        return f"{label} ({ts}): {path.name}"
    except OSError:
        return f"{label}: {path.name}"


def parse_dollar_amount(text):
    """Extract dollar amount from contributor string like 'Duke Energy ($10,350)'."""
    match = re.search(r'\$([0-9,]+)', text)
    if match:
        return int(match.group(1).replace(',', ''))
    return 0


def get_total_contributions(contributors_str):
    """Sum all dollar amounts in a contributors string."""
    if not contributors_str:
        return 0
    total = 0
    for match in re.finditer(r'\$([0-9,]+)', contributors_str):
        total += int(match.group(1).replace(',', ''))
    return total


def compare_files():
    """Main comparison function."""

    # Check files exist
    if not NEWER_FILE.exists():
        print(f"ERROR: Newer file not found: {NEWER_FILE}")
        print("       Run the scraper first: python 03_scripts/scraper_OS.py")
        sys.exit(1)
    if not OLDER_FILE.exists():
        print(f"ERROR: Older file not found: {OLDER_FILE}")
        sys.exit(1)

    # Load data
    print("Loading files...")
    print(f"  {describe_file(NEWER_FILE, 'Newer (scraped)')}")
    print(f"  {describe_file(OLDER_FILE, 'Older  (production)')}")
    print()

    newer_rows = load_csv(NEWER_FILE)
    older_rows = load_csv(OLDER_FILE)

    # Create dictionaries keyed by bioguide_id
    newer = {row['bioguide_id']: row for row in newer_rows}
    older = {row['bioguide_id']: row for row in older_rows}

    # Basic statistics
    print("=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    print(f"Newer file: {len(newer_rows)} rows")
    print(f"Older file: {len(older_rows)} rows")
    print(f"Difference: {len(newer_rows) - len(older_rows):+d} rows")
    print()

    # Bioguide ID differences
    print("=" * 80)
    print("MEMBER DIFFERENCES")
    print("=" * 80)
    ids_new = set(newer.keys())
    ids_old = set(older.keys())

    only_in_new = ids_new - ids_old
    only_in_old = ids_old - ids_new

    if only_in_new:
        print(f"Only in newer file ({len(only_in_new)}):")
        for bid in sorted(only_in_new):
            print(f"  + {newer[bid]['first_name']} {newer[bid]['last_name']} ({bid})")
    else:
        print("Only in newer file: None")

    if only_in_old:
        print(f"\nOnly in older file ({len(only_in_old)}):")
        for bid in sorted(only_in_old):
            print(f"  - {older[bid]['first_name']} {older[bid]['last_name']} ({bid})")
    else:
        print("\nOnly in older file: None")
    print()

    # Period analysis
    print("=" * 80)
    print("PERIOD ANALYSIS")
    print("=" * 80)

    period_mismatches = []
    for bid in ids_new & ids_old:
        new_period = newer[bid].get('contributors_period', '')
        old_period = older[bid].get('contributors_period', '')
        if new_period != old_period and new_period and old_period:
            period_mismatches.append({
                'bioguide_id': bid,
                'name': f"{newer[bid]['first_name']} {newer[bid]['last_name']}",
                'old_period': old_period,
                'new_period': new_period
            })

    if period_mismatches:
        print(f"Period mismatches: {len(period_mismatches)}")
        if len(period_mismatches) <= 10:
            for pm in period_mismatches:
                print(f"  {pm['name']}: {pm['old_period']} -> {pm['new_period']}")
        else:
            # Check if it's a bulk shift (all same change)
            old_periods = set(pm['old_period'] for pm in period_mismatches)
            new_periods = set(pm['new_period'] for pm in period_mismatches)
            if len(old_periods) == 1 and len(new_periods) == 1:
                print(f"  (Bulk period update: {list(old_periods)[0]} -> {list(new_periods)[0]})")
            else:
                print(f"  (Showing first 5 of {len(period_mismatches)})")
                for pm in period_mismatches[:5]:
                    print(f"  {pm['name']}: {pm['old_period']} -> {pm['new_period']}")
    else:
        print("All periods match between files")
    print()

    # Empty data analysis
    print("=" * 80)
    print("EMPTY DATA ANALYSIS")
    print("=" * 80)

    newer_empty_contributors = []
    newer_empty_industries = []
    went_empty = []

    for bid in ids_new & ids_old:
        new_contrib = newer[bid].get('top_contributors', '').strip()
        old_contrib = older[bid].get('top_contributors', '').strip()
        new_indust = newer[bid].get('top_industries', '').strip()
        old_indust = older[bid].get('top_industries', '').strip()

        name = f"{newer[bid]['first_name']} {newer[bid]['last_name']}"

        # Track if newer has empty data
        if not new_contrib:
            newer_empty_contributors.append({'name': name, 'bid': bid})
            if old_contrib:
                went_empty.append({'name': name, 'bid': bid, 'field': 'contributors', 'old': old_contrib[:50]})

        if not new_indust:
            newer_empty_industries.append({'name': name, 'bid': bid})
            if old_indust:
                went_empty.append({'name': name, 'bid': bid, 'field': 'industries', 'old': old_indust[:50]})

    print(f"Newer file - empty contributors: {len(newer_empty_contributors)}")
    print(f"Newer file - empty industries: {len(newer_empty_industries)}")

    if went_empty:
        print(f"\n!! DATA LOSS: {len(went_empty)} fields had data but are now empty:")
        for item in went_empty[:10]:
            print(f"  {item['name']} ({item['field']}): was '{item['old']}...'")
        if len(went_empty) > 10:
            print(f"  ... and {len(went_empty) - 10} more")
    else:
        print("\nNo data loss detected (no fields went from populated to empty)")
    print()

    # Contribution total analysis
    print("=" * 80)
    print("CONTRIBUTION TOTALS ANALYSIS")
    print("=" * 80)

    contrib_diffs = []
    for bid in ids_new & ids_old:
        new_total = get_total_contributions(newer[bid].get('top_contributors', ''))
        old_total = get_total_contributions(older[bid].get('top_contributors', ''))
        diff = new_total - old_total

        if abs(diff) > 5000:  # Flag differences over $5,000
            contrib_diffs.append({
                'bioguide_id': bid,
                'name': f"{newer[bid]['first_name']} {newer[bid]['last_name']}",
                'old': old_total,
                'new': new_total,
                'diff': diff,
                'abs_diff': abs(diff)
            })

    contrib_diffs.sort(key=lambda x: x['abs_diff'], reverse=True)

    print(f"Members with contribution total differences > $5,000: {len(contrib_diffs)}")
    if contrib_diffs:
        print("\nTop 10 largest changes:")
        for i, d in enumerate(contrib_diffs[:10], 1):
            direction = "+" if d['diff'] > 0 else ""
            print(f"  {i:2d}. {d['name']:30s} ${d['old']:>10,} -> ${d['new']:>10,} ({direction}${d['diff']:,})")
    print()

    # Text differences sample
    print("=" * 80)
    print("TOP CONTRIBUTORS TEXT CHANGES")
    print("=" * 80)

    text_diffs = 0
    samples = []
    for bid in ids_new & ids_old:
        new_text = newer[bid].get('top_contributors', '')
        old_text = older[bid].get('top_contributors', '')
        if new_text != old_text:
            text_diffs += 1
            if len(samples) < 3 and new_text and old_text:
                samples.append({
                    'name': f"{newer[bid]['first_name']} {newer[bid]['last_name']}",
                    'old': old_text[:80],
                    'new': new_text[:80]
                })

    print(f"Members with different top_contributors text: {text_diffs} / {len(ids_new & ids_old)}")
    if samples:
        print("\nSample differences:")
        for s in samples:
            print(f"\n  {s['name']}:")
            print(f"    OLD: {s['old']}...")
            print(f"    NEW: {s['new']}...")
    print()

    # Industries text differences
    print("=" * 80)
    print("TOP INDUSTRIES TEXT CHANGES")
    print("=" * 80)

    indust_diffs = 0
    for bid in ids_new & ids_old:
        if newer[bid].get('top_industries', '') != older[bid].get('top_industries', ''):
            indust_diffs += 1

    print(f"Members with different top_industries text: {indust_diffs} / {len(ids_new & ids_old)}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    issues = []
    warnings = []

    if went_empty:
        issues.append(f"!! {len(went_empty)} fields lost data (were populated, now empty)")
    if only_in_old:
        issues.append(f"!! {len(only_in_old)} members removed from newer file")

    if only_in_new:
        warnings.append(f"+ {len(only_in_new)} new members added")
    if period_mismatches:
        warnings.append(f"~ {len(period_mismatches)} period changes (expected if scraping newer data)")
    if text_diffs > 0:
        pct = 100 * text_diffs / len(ids_new & ids_old)
        warnings.append(f"~ {text_diffs} ({pct:.0f}%) contributor text changes")

    if issues:
        print("CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
        print()

    if warnings:
        print("Warnings (may be expected):")
        for w in warnings:
            print(f"  {w}")
        print()

    if not issues and not warnings:
        print("No issues or warnings detected")
        print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if issues:
        print("!! DO NOT deploy to production - investigate issues first")
        return 1
    elif went_empty:
        print("!! Review empty data before deploying")
        return 1
    else:
        print("Safe to deploy: mv 00members_donor_summary.csv members_donor_summary.csv")

        # Ask user if they want to replace the production file
        print()
        print("=" * 80)
        print("DEPLOY TO PRODUCTION?")
        print("=" * 80)
        response = input(f"Replace {OLDER_FILE.name} with {NEWER_FILE.name}? (y/N): ").strip().lower()

        if response in ('y', 'yes'):
            try:
                # Rename existing production file with _OLD suffix (before extension)
                old_backup = OLDER_FILE.with_name(f"{OLDER_FILE.stem}_OLD{OLDER_FILE.suffix}")
                if old_backup.exists():
                    print(f"Warning: {old_backup.name} already exists, removing it...")
                    old_backup.unlink()

                OLDER_FILE.rename(old_backup)
                print(f"Backed up {OLDER_FILE.name} -> {old_backup.name}")

                # Rename new file by removing 00 prefix
                new_name = NEWER_FILE.name
                if new_name.startswith('00'):
                    new_name = new_name[2:]  # Remove '00' prefix
                new_production = NEWER_FILE.parent / new_name
                NEWER_FILE.rename(new_production)
                print(f"Deployed {NEWER_FILE.name} -> {new_production.name}")
                print()
                print("Deployment complete!")
                return 0
            except Exception as e:
                print(f"Error during deployment: {e}")
                return 1
        else:
            print("Deployment cancelled.")
            return 0


if __name__ == "__main__":
    sys.exit(compare_files())
