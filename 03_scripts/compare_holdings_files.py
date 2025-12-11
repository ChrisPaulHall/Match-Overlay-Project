#!/usr/bin/env python3
"""
Compare two members_holdings_and_sectors CSV files.
Usage: python compare_holdings_files.py

Compares:
- 00members_holdings_and_sectors.csv (newer/scraped version)
- members_holdings_and_sectors.csv (production version)

Checks for:
- Missing members (data loss)
- Members whose net worth went to $0
- Empty holdings/sectors that previously had data
- Significant net worth changes
"""

import csv
import sys
from pathlib import Path

# File paths
BASE_DIR = Path(__file__).parent.parent / "01_data"
NEWER_FILE = BASE_DIR / "00members_holdings_and_sectors.csv"
OLDER_FILE = BASE_DIR / "members_holdings_and_sectors.csv"

# Thresholds
NET_WORTH_DIFF_THRESHOLD = 1000  # Flag differences over $1,000
ZERO_WORTH_THRESHOLD = 10000  # Flag when someone with >$10k goes to $0


def load_csv(filepath):
    """Load CSV file and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def format_money(value):
    """Format a number as currency."""
    if value is None or value == 0:
        return "$0"
    return f"${value:,.0f}"


def compare_files():
    """Main comparison function."""

    # Check files exist
    if not NEWER_FILE.exists():
        print(f"ERROR: Newer file not found: {NEWER_FILE}")
        print("       Run the scraper first: python 03_scripts/scraper_quiver.py --resume")
        sys.exit(1)
    if not OLDER_FILE.exists():
        print(f"ERROR: Older file not found: {OLDER_FILE}")
        sys.exit(1)

    # Load data
    print(f"Loading files...")
    print(f"  Newer (scraped): {NEWER_FILE.name}")
    print(f"  Older (production): {OLDER_FILE.name}")
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

    # Member differences
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

    # Net worth comparison
    print("=" * 80)
    print("NET WORTH ANALYSIS")
    print("=" * 80)

    net_worth_diffs = []
    members_went_to_zero = []

    for bid in ids_new & ids_old:
        try:
            nw_new = float(newer[bid]['net_worth_estimate']) if newer[bid].get('net_worth_estimate') else 0
            nw_old = float(older[bid]['net_worth_estimate']) if older[bid].get('net_worth_estimate') else 0
            diff = nw_new - nw_old

            name = f"{newer[bid]['first_name']} {newer[bid]['last_name']}"

            # Track members who went to zero
            if nw_old > ZERO_WORTH_THRESHOLD and nw_new == 0:
                members_went_to_zero.append({
                    'bioguide_id': bid,
                    'name': name,
                    'old': nw_old,
                    'new': nw_new
                })

            # Track significant differences
            if abs(diff) > NET_WORTH_DIFF_THRESHOLD:
                net_worth_diffs.append({
                    'bioguide_id': bid,
                    'name': name,
                    'old': nw_old,
                    'new': nw_new,
                    'diff': diff,
                    'abs_diff': abs(diff)
                })
        except (ValueError, KeyError) as e:
            print(f"Warning: Error processing {bid}: {e}")

    # Show members who went to zero (CRITICAL)
    if members_went_to_zero:
        print(f"\n!! CRITICAL: {len(members_went_to_zero)} members went to $0 in newer file:")
        members_went_to_zero.sort(key=lambda x: x['old'], reverse=True)
        for m in members_went_to_zero:
            print(f"  !! {m['name']:30s} {format_money(m['old']):>15s} -> $0")
        print()
    else:
        print("No members went from significant net worth to $0")
        print()

    # Show top net worth changes
    net_worth_diffs.sort(key=lambda x: x['abs_diff'], reverse=True)
    print(f"Members with net worth differences > {format_money(NET_WORTH_DIFF_THRESHOLD)}: {len(net_worth_diffs)}")

    if net_worth_diffs:
        print(f"Max absolute difference: {format_money(max(d['abs_diff'] for d in net_worth_diffs))}")
        print()
        print("Top 15 largest changes:")
        for i, d in enumerate(net_worth_diffs[:15], 1):
            direction = "+" if d['diff'] > 0 else ""
            print(f"  {i:2d}. {d['name']:30s} {format_money(d['old']):>15s} -> {format_money(d['new']):>15s} ({direction}{format_money(d['diff'])})")
    print()

    # Empty data analysis
    print("=" * 80)
    print("EMPTY DATA ANALYSIS")
    print("=" * 80)

    holdings_went_empty = []
    sectors_went_empty = []

    for bid in ids_new & ids_old:
        name = f"{newer[bid]['first_name']} {newer[bid]['last_name']}"

        new_holdings = newer[bid].get('top_holdings', '').strip()
        old_holdings = older[bid].get('top_holdings', '').strip()
        new_sectors = newer[bid].get('top_traded_sectors', '').strip()
        old_sectors = older[bid].get('top_traded_sectors', '').strip()

        if not new_holdings and old_holdings:
            holdings_went_empty.append({'name': name, 'bid': bid, 'old': old_holdings[:50]})

        if not new_sectors and old_sectors:
            sectors_went_empty.append({'name': name, 'bid': bid, 'old': old_sectors[:50]})

    if holdings_went_empty:
        print(f"!! {len(holdings_went_empty)} members lost holdings data:")
        for item in holdings_went_empty[:10]:
            print(f"  {item['name']}: was '{item['old']}...'")
        if len(holdings_went_empty) > 10:
            print(f"  ... and {len(holdings_went_empty) - 10} more")
    else:
        print("No members lost holdings data")

    if sectors_went_empty:
        print(f"\n!! {len(sectors_went_empty)} members lost sectors data:")
        for item in sectors_went_empty[:10]:
            print(f"  {item['name']}: was '{item['old']}...'")
        if len(sectors_went_empty) > 10:
            print(f"  ... and {len(sectors_went_empty) - 10} more")
    else:
        print("No members lost sectors data")
    print()

    # Holdings text differences
    print("=" * 80)
    print("TOP HOLDINGS TEXT CHANGES")
    print("=" * 80)

    holdings_diffs = 0
    samples = []
    for bid in ids_new & ids_old:
        h_new = newer[bid].get('top_holdings', '')
        h_old = older[bid].get('top_holdings', '')
        if h_new != h_old:
            holdings_diffs += 1
            if len(samples) < 3 and h_new and h_old:
                samples.append({
                    'name': f"{newer[bid]['first_name']} {newer[bid]['last_name']}",
                    'old': h_old[:80],
                    'new': h_new[:80]
                })

    print(f"Members with different holdings text: {holdings_diffs} / {len(ids_new & ids_old)}")
    if samples:
        print("\nSample differences:")
        for s in samples:
            print(f"\n  {s['name']}:")
            print(f"    OLD: {s['old']}...")
            print(f"    NEW: {s['new']}...")
    print()

    # Sectors differences
    print("=" * 80)
    print("TOP TRADED SECTORS CHANGES")
    print("=" * 80)

    sectors_diffs = []
    for bid in ids_new & ids_old:
        s_new = newer[bid].get('top_traded_sectors', '')
        s_old = older[bid].get('top_traded_sectors', '')
        if s_new != s_old:
            sectors_diffs.append({
                'name': f"{newer[bid]['first_name']} {newer[bid]['last_name']}",
                'old': s_old,
                'new': s_new
            })

    print(f"Members with different sectors: {len(sectors_diffs)} / {len(ids_new & ids_old)}")

    if sectors_diffs and len(sectors_diffs) <= 10:
        print("\nDifferences:")
        for sd in sectors_diffs:
            print(f"\n  {sd['name']}:")
            print(f"    OLD: {sd['old'] if sd['old'] else '(empty)'}")
            print(f"    NEW: {sd['new'] if sd['new'] else '(empty)'}")
    elif sectors_diffs:
        print(f"\n(Showing first 5 of {len(sectors_diffs)})")
        for sd in sectors_diffs[:5]:
            print(f"\n  {sd['name']}:")
            print(f"    OLD: {sd['old'] if sd['old'] else '(empty)'}")
            print(f"    NEW: {sd['new'] if sd['new'] else '(empty)'}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    critical_issues = []
    warnings = []

    if members_went_to_zero:
        critical_issues.append(f"!! {len(members_went_to_zero)} members went to $0 net worth")
    if only_in_old:
        critical_issues.append(f"!! {len(only_in_old)} members removed from newer file")
    if holdings_went_empty:
        critical_issues.append(f"!! {len(holdings_went_empty)} members lost holdings data")
    if sectors_went_empty:
        warnings.append(f"~ {len(sectors_went_empty)} members lost sectors data")

    if only_in_new:
        warnings.append(f"+ {len(only_in_new)} new members added")
    if len(net_worth_diffs) > len(ids_new & ids_old) * 0.5:
        pct = 100 * len(net_worth_diffs) / len(ids_new & ids_old)
        warnings.append(f"~ {len(net_worth_diffs)} ({pct:.0f}%) members have net worth changes")
    if holdings_diffs > 0:
        warnings.append(f"~ {holdings_diffs} members have holdings text changes")

    if critical_issues:
        print("CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"  {issue}")
        print()

    if warnings:
        print("Warnings (may be expected):")
        for w in warnings:
            print(f"  {w}")
        print()

    if not critical_issues and not warnings:
        print("No issues or warnings detected")
        print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if critical_issues:
        print("!! DO NOT deploy to production - investigate critical issues first")
        return 1
    else:
        print("Safe to deploy:")
        print("  mv 01_data/00members_holdings_and_sectors.csv 01_data/members_holdings_and_sectors.csv")
        return 0


if __name__ == "__main__":
    sys.exit(compare_files())
