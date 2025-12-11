#!/usr/bin/env python3
"""
Interactive Face Import - Review and import pending face captures.

This script reviews face images captured by the matcher from 01_data/pending_review/
and lets you accept or reject each one. Accepted faces are moved to faces_official
with proper naming (e.g., RLatta6.jpg).

Usage:
    # Interactive review (default):
    python 03_scripts/import_pending_faces.py

    # Auto-import all without review:
    python 03_scripts/import_pending_faces.py --auto

    # Dry run to see what would happen:
    python 03_scripts/import_pending_faces.py --dry-run

Controls during review:
    [a] Approve  - move to faces_official
    [r] Reject   - delete the image
    [s] Skip     - leave for later
    [q] Quit     - exit review session

faces_official download:
https://drive.google.com/drive/folders/1VjMNSBHbMNhX-oLgdK1u1NF8ttAcBxpC?usp=drive_link
"""
import argparse
import csv
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PENDING_DIR = PROJECT_ROOT / "01_data" / "pending_review"
DEFAULT_FACES_DIR = PROJECT_ROOT / "01_data" / "faces_official"
DEFAULT_LOOKUP_CSV = PROJECT_ROOT / "01_data" / "members_face_lookup.csv"


def load_bioguide_to_name_map(csv_path: Path) -> dict:
    """Load mapping from bioguide_id to face filename stem.

    Returns dict like: {'M001239': 'JMcGuire', 'G000558': 'BGuthrie'}
    """
    mapping = {}
    if not csv_path.exists():
        return mapping

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bioguide = row.get('bioguide_id', '').strip()
                filename = row.get('filename', '').strip()
                if bioguide and filename:
                    # Extract stem (remove .jpg extension and trailing numbers)
                    stem = Path(filename).stem
                    base_name = re.sub(r'\d+$', '', stem)
                    mapping[bioguide] = base_name
    except Exception as e:
        print(f"Warning: Could not load bioguide mapping: {e}")

    return mapping


def get_next_number(base_name: str, faces_dir: Path) -> int:
    """Find the next available number for a base name (e.g., EWarren -> 5)."""
    existing = list(faces_dir.glob(f"{base_name}*.jpg")) + \
               list(faces_dir.glob(f"{base_name}*.jpeg")) + \
               list(faces_dir.glob(f"{base_name}*.png"))

    max_num = -1
    pattern = re.compile(rf"^{re.escape(base_name)}(\d*)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

    for f in existing:
        m = pattern.match(f.name)
        if m:
            num_str = m.group(1)
            if num_str == "":
                max_num = max(max_num, 0)  # Base file (e.g., EWarren.jpg = 0)
            else:
                max_num = max(max_num, int(num_str))

    return max_num + 1


def parse_pending_filename(filename: str) -> tuple[str, str, float] | None:
    """Parse pending_review filename format: BioguideID__BaseName__Score__Timestamp.jpg

    Returns: (base_name, bioguide_id, score) or None if pattern doesn't match
    """
    # Pattern: L000566__RLatta5__0.731__20251210_210343.jpg
    # Format: BioguideID__BaseName__Score__Timestamp.jpg
    pattern = r"^([A-Z]\d+)__([A-Za-z]+)\d*__(\d+\.\d+)__\d+_\d+\.jpg$"
    m = re.match(pattern, filename, re.IGNORECASE)
    if m:
        bioguide = m.group(1)
        base_name = re.sub(r'\d+$', '', m.group(2))  # Remove trailing numbers from base name
        score = float(m.group(3))
        return base_name, bioguide, score
    return None


def open_image(filepath: Path) -> bool:
    """Open image in system viewer (Preview on macOS). Returns True if successful."""
    try:
        if platform.system() == "Darwin":  # macOS
            subprocess.run(["open", str(filepath)], check=True)
        elif platform.system() == "Windows":
            os.startfile(str(filepath))
        else:  # Linux
            subprocess.run(["xdg-open", str(filepath)], check=True)
        return True
    except Exception as e:
        print(f"    Could not open image: {e}")
        return False


def get_user_decision() -> str:
    """Get user input for accept/reject/skip/quit."""
    while True:
        try:
            response = input("    Action [a/r/s/q]: ").strip().lower()
            if response in ("a", "approve", "accept", "y", "yes", ""):
                return "approve"
            elif response in ("r", "reject", "d", "delete", "n", "no"):
                return "reject"
            elif response in ("s", "skip"):
                return "skip"
            elif response in ("q", "quit", "exit"):
                return "quit"
            else:
                print("    Invalid input. Use: [a]pprove, [r]eject, [s]kip, [q]uit")
        except (KeyboardInterrupt, EOFError):
            print("\n    Interrupted.")
            return "quit"


def count_existing_images(base_name: str, faces_dir: Path) -> int:
    """Count how many images exist for a member in the database."""
    existing = list(faces_dir.glob(f"{base_name}*.jpg")) + \
               list(faces_dir.glob(f"{base_name}*.jpeg")) + \
               list(faces_dir.glob(f"{base_name}*.png"))
    return len(existing)


def review_faces_interactive(
    source_dir: Path,
    faces_dir: Path,
    bioguide_map: dict,
    dry_run: bool = False
):
    """Interactively review and import pending faces."""
    if not source_dir.exists():
        print(f"Pending review directory not found: {source_dir}")
        print("No faces to review. The matcher saves captures here when running.")
        return

    if not faces_dir.exists():
        print(f"Error: faces_dir does not exist: {faces_dir}")
        return

    # Find all pending files
    files_to_process = []
    for f in sorted(source_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            parsed = parse_pending_filename(f.name)
            if parsed:
                files_to_process.append((f, parsed))

    if not files_to_process:
        print(f"No pending faces found in: {source_dir}")
        print("Expected format: BioguideID__BaseName__Score__Timestamp.jpg")
        print("  e.g., L000566__RLatta__0.731__20251210_210343.jpg")
        return

    print(f"\nFound {len(files_to_process)} pending face(s) to review")
    print(f"Target: {faces_dir}")
    print("\nCommands: [a]pprove  [r]eject  [s]kip  [q]uit")
    print("=" * 60)

    stats = {"approved": 0, "rejected": 0, "skipped": 0}

    for i, (filepath, (base_name, bioguide, score)) in enumerate(files_to_process, 1):
        # Check if bioguide maps to a different name
        resolved_name = bioguide_map.get(bioguide, base_name)
        if resolved_name != base_name:
            name_note = f" (resolved from {base_name})"
        else:
            name_note = ""

        existing_count = count_existing_images(resolved_name, faces_dir)
        next_num = get_next_number(resolved_name, faces_dir)
        new_name = f"{resolved_name}{next_num}.jpg" if next_num > 0 else f"{resolved_name}.jpg"
        new_path = faces_dir / new_name

        print(f"\n[{i}/{len(files_to_process)}] {filepath.name}")
        print(f"    Member:   {resolved_name}{name_note}")
        print(f"    Bioguide: {bioguide}")
        print(f"    Score:    {score:.3f}")
        print(f"    Existing: {existing_count} images in DB")
        print(f"    Save as:  {new_name}")

        # Open image for viewing
        open_image(filepath)

        # Get user decision
        decision = get_user_decision()

        if decision == "quit":
            print("\nReview stopped.")
            break
        elif decision == "approve":
            if not dry_run:
                # Avoid collision
                while new_path.exists():
                    next_num += 1
                    new_name = f"{resolved_name}{next_num}.jpg"
                    new_path = faces_dir / new_name

                shutil.move(str(filepath), str(new_path))
                print(f"    -> APPROVED: {new_name}")
            else:
                print(f"    -> [DRY RUN] Would save as {new_name}")
            stats["approved"] += 1
        elif decision == "reject":
            if not dry_run:
                filepath.unlink()
                print("    -> REJECTED (deleted)")
            else:
                print("    -> [DRY RUN] Would delete")
            stats["rejected"] += 1
        else:  # skip
            print("    -> SKIPPED")
            stats["skipped"] += 1

    # Summary
    print("\n" + "=" * 60)
    print("REVIEW SUMMARY")
    print("=" * 60)
    print(f"  Approved: {stats['approved']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Skipped:  {stats['skipped']}")

    if stats["approved"] > 0 and not dry_run:
        print("\nRun warm_embeddings.py to update the face cache:")
        print("  python core/warm_embeddings.py --faces_db 01_data/faces_official --force")


def import_faces_auto(
    source_dir: Path,
    faces_dir: Path,
    bioguide_map: dict,
    dry_run: bool = False
):
    """Auto-import all pending faces without review."""
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return

    if not faces_dir.exists():
        print(f"Error: faces_dir does not exist: {faces_dir}")
        return

    files_to_process = []
    for f in sorted(source_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            parsed = parse_pending_filename(f.name)
            if parsed:
                files_to_process.append((f, parsed))

    if not files_to_process:
        print("No pending_review files found to import.")
        print("Expected format: BioguideID__BaseName__Score__Timestamp.jpg")
        return

    print(f"Auto-importing {len(files_to_process)} file(s):\n")

    for filepath, (base_name, bioguide, score) in files_to_process:
        # Check if bioguide maps to a different name
        resolved_name = bioguide_map.get(bioguide, base_name)

        next_num = get_next_number(resolved_name, faces_dir)
        new_name = f"{resolved_name}{next_num}.jpg" if next_num > 0 else f"{resolved_name}.jpg"
        new_path = faces_dir / new_name

        # Avoid collision
        while new_path.exists():
            next_num += 1
            new_name = f"{resolved_name}{next_num}.jpg"
            new_path = faces_dir / new_name

        print(f"  {filepath.name}")
        print(f"    -> {new_name}  (score: {score:.3f}, bioguide: {bioguide})")

        if not dry_run:
            shutil.move(str(filepath), str(new_path))
            print("    [MOVED]")
        else:
            print("    [DRY RUN]")
        print()

    if not dry_run:
        print("Done! Run warm_embeddings.py to update the cache:")
        print("  python core/warm_embeddings.py --faces_db 01_data/faces_official --force")


def main():
    parser = argparse.ArgumentParser(
        description="Review and import pending face captures into faces_official",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive review (default):
  python 03_scripts/import_pending_faces.py

  # Auto-import all without review:
  python 03_scripts/import_pending_faces.py --auto

  # Dry run to preview:
  python 03_scripts/import_pending_faces.py --dry-run
        """
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_PENDING_DIR,
                        help=f"Source folder with pending faces (default: 01_data/pending_review)")
    parser.add_argument("--faces-db", type=Path, default=DEFAULT_FACES_DIR,
                        help=f"Target faces database (default: 01_data/faces_official)")
    parser.add_argument("--lookup-csv", type=Path, default=DEFAULT_LOOKUP_CSV,
                        help="CSV file with bioguide_id to filename mapping")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-import all without interactive review")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without making changes")
    args = parser.parse_args()

    source_dir = args.source
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir

    faces_dir = args.faces_db
    if not faces_dir.is_absolute():
        faces_dir = PROJECT_ROOT / faces_dir

    # Load bioguide -> name mapping for proper file naming
    bioguide_map = load_bioguide_to_name_map(args.lookup_csv)
    if bioguide_map:
        print(f"Loaded {len(bioguide_map)} bioguide -> name mappings")
    else:
        print("Warning: No bioguide mappings loaded. Using names from filenames.")

    print(f"Source: {source_dir}")
    print(f"Target: {faces_dir}")

    if args.auto:
        import_faces_auto(source_dir, faces_dir, bioguide_map, dry_run=args.dry_run)
    else:
        review_faces_interactive(source_dir, faces_dir, bioguide_map, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main() or 0)
