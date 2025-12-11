#!/usr/bin/env python3
"""
Import pending_review faces into faces_official with auto-renaming.

faces_official download:
https://drive.google.com/drive/folders/10J55kR0nNMOnNC4cl_aGvNOa90pVFS7p?usp=drive_link

Usage:
    # Drop files into faces_official, then run:
    python 03_scripts/import_pending_faces.py

    # Or specify a source folder:
    python 03_scripts/import_pending_faces.py --source /path/to/kept/faces

Files named like: EWarren__W000817__0.713__20251210_143022.jpg
Get renamed to:   EWarren5.jpg (next available number)
"""
import argparse
import os
import re
import shutil
from pathlib import Path


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
    """Parse pending_review filename format: BaseName__BioguideID__Score__Timestamp.jpg"""
    # Pattern: EWarren__W000817__0.713__20251210_143022.jpg
    # Also handles: EWarren2__W000817__0.713__20251210_143022.jpg (with existing number)
    pattern = r"^([A-Za-z]+)\d*__([A-Z]\d+)__(\d+\.\d+)__\d+_\d+\.jpg$"
    m = re.match(pattern, filename, re.IGNORECASE)
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None


def import_faces(source_dir: Path, faces_dir: Path, dry_run: bool = False):
    """Import and rename pending faces into the faces database."""
    if not faces_dir.exists():
        print(f"Error: faces_dir does not exist: {faces_dir}")
        return

    # Find files to import (either in source_dir or already in faces_dir with __ naming)
    files_to_process = []

    if source_dir and source_dir.exists() and source_dir != faces_dir:
        # Import from external source
        for f in source_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                parsed = parse_pending_filename(f.name)
                if parsed:
                    files_to_process.append((f, parsed, "move"))

    # Also check faces_dir for files with __ naming that need renaming
    for f in faces_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            parsed = parse_pending_filename(f.name)
            if parsed:
                files_to_process.append((f, parsed, "rename"))

    if not files_to_process:
        print("No pending_review files found to import.")
        print("Expected format: BaseName__BioguideID__Score__Timestamp.jpg")
        print(f"  e.g., EWarren__W000817__0.713__20251210_143022.jpg")
        return

    print(f"Found {len(files_to_process)} file(s) to process:\n")

    for filepath, (base_name, bioguide, score), action in files_to_process:
        next_num = get_next_number(base_name, faces_dir)
        new_name = f"{base_name}{next_num}.jpg" if next_num > 0 else f"{base_name}.jpg"
        new_path = faces_dir / new_name

        # Avoid collision
        while new_path.exists():
            next_num += 1
            new_name = f"{base_name}{next_num}.jpg"
            new_path = faces_dir / new_name

        print(f"  {filepath.name}")
        print(f"    -> {new_name}  (score: {score:.3f}, bioguide: {bioguide})")

        if not dry_run:
            if action == "move":
                shutil.copy2(filepath, new_path)
                print(f"    [COPIED]")
            else:  # rename
                filepath.rename(new_path)
                print(f"    [RENAMED]")
        else:
            print(f"    [DRY RUN - no changes made]")
        print()

    if not dry_run:
        print("Done! Run warm_embeddings.py to update the cache:")
        print("  python core/warm_embeddings.py --faces_db 01_data/faces_official")


def main():
    parser = argparse.ArgumentParser(description="Import pending_review faces into faces_official")
    parser.add_argument("--source", type=Path, help="Source folder with kept faces (optional)")
    parser.add_argument("--faces_db", type=Path, default=Path("01_data/faces_official"),
                        help="Target faces database folder")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    args = parser.parse_args()

    faces_dir = args.faces_db
    if not faces_dir.is_absolute():
        faces_dir = Path.cwd() / faces_dir

    source_dir = args.source
    if source_dir and not source_dir.is_absolute():
        source_dir = Path.cwd() / source_dir

    import_faces(source_dir or faces_dir, faces_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
