#!/usr/bin/env python3
"""Review and approve pending face crops for database augmentation.

faces_official download:
https://drive.google.com/drive/folders/10J55kR0nNMOnNC4cl_aGvNOa90pVFS7p?usp=drive_link

This script displays pending faces one at a time and lets you:
- [a] Approve: Move to faces_official directory
- [r] Reject: Delete the image
- [s] Skip: Leave for later review
- [q] Quit: Exit the review session

Usage:
    python 03_scripts/review_pending_faces.py
    python 03_scripts/review_pending_faces.py --pending-dir 02_outputs/pending_review
    python 03_scripts/review_pending_faces.py --auto-approve  # Approve all without prompting
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


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
                    # Extract stem (remove .jpg extension)
                    stem = Path(filename).stem
                    mapping[bioguide] = stem
    except Exception as e:
        print(f"Warning: Could not load bioguide mapping: {e}")

    return mapping


def is_bioguide_id(name: str) -> bool:
    """Check if a string looks like a bioguide ID (e.g., M001239, G000558)."""
    return bool(re.match(r'^[A-Z]\d{6}$', name))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def parse_filename(filename: str) -> dict:
    """Parse pending face filename to extract metadata.

    Expected format: {name}__{bioguide}__{score}__{timestamp}.jpg
    Example: JSmith__S000123__0.752__20251209_143022.jpg
    """
    stem = Path(filename).stem
    parts = stem.split("__")

    result = {
        "name": parts[0] if len(parts) > 0 else "unknown",
        "bioguide": parts[1] if len(parts) > 1 else "",
        "score": 0.0,
        "timestamp": "",
    }

    if len(parts) >= 3:
        try:
            result["score"] = float(parts[2])
        except ValueError:
            pass

    if len(parts) >= 4:
        result["timestamp"] = parts[3]

    return result


def display_image(image_path: Path, title: str = "Pending Face") -> None:
    """Display image in a window (if cv2 available)."""
    if not CV2_AVAILABLE:
        print(f"  (Install opencv-python to preview images)")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  (Could not load image)")
        return

    # Resize if too large
    h, w = img.shape[:2]
    max_dim = 400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow(title, img)
    cv2.waitKey(1)  # Brief display, don't block


def close_display() -> None:
    """Close any open cv2 windows."""
    if CV2_AVAILABLE:
        cv2.destroyAllWindows()


def review_faces(
    pending_dir: Path,
    faces_db_dir: Path,
    auto_approve: bool = False,
    bioguide_map: dict | None = None,
) -> dict:
    """Review pending faces interactively.

    Args:
        pending_dir: Directory containing pending face images
        faces_db_dir: Target faces database directory
        auto_approve: If True, approve all without prompting
        bioguide_map: Optional dict mapping bioguide_id -> face_name

    Returns:
        dict with counts: approved, rejected, skipped
    """
    if bioguide_map is None:
        bioguide_map = {}

    if not pending_dir.exists():
        print(f"Pending directory does not exist: {pending_dir}")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    pending_files = sorted(pending_dir.glob("*.jpg"))
    if not pending_files:
        print("No pending faces to review.")
        return {"approved": 0, "rejected": 0, "skipped": 0}

    print(f"\nFound {len(pending_files)} pending faces to review.")
    print(f"Target directory: {faces_db_dir}")
    print("\nCommands: [a]pprove  [r]eject  [s]kip  [q]uit\n")
    print("-" * 60)

    stats = {"approved": 0, "rejected": 0, "skipped": 0}

    for i, image_path in enumerate(pending_files, 1):
        meta = parse_filename(image_path.name)

        print(f"\n[{i}/{len(pending_files)}] {image_path.name}")
        print(f"  Name:     {meta['name']}")
        print(f"  Bioguide: {meta['bioguide']}")
        print(f"  Score:    {meta['score']:.3f}")

        # Count existing images for this person
        if meta['name']:
            existing = list(faces_db_dir.glob(f"{meta['name']}*.jpg"))
            print(f"  Existing: {len(existing)} images in DB")

        display_image(image_path, f"Review: {meta['name']}")

        if auto_approve:
            action = 'a'
            print("  Auto-approving...")
        else:
            while True:
                try:
                    action = input("  Action [a/r/s/q]: ").strip().lower()
                except EOFError:
                    action = 'q'

                if action in ('a', 'r', 's', 'q'):
                    break
                print("  Invalid input. Use: a=approve, r=reject, s=skip, q=quit")

        if action == 'q':
            print("\nQuitting review session.")
            close_display()
            break
        elif action == 'a':
            # Resolve the proper face name
            # Check if name looks like a bioguide ID and resolve it
            face_name = meta['name']
            if is_bioguide_id(face_name) and face_name in bioguide_map:
                resolved_name = bioguide_map[face_name]
                print(f"  Resolved {face_name} -> {resolved_name}")
                face_name = resolved_name
            elif is_bioguide_id(face_name) and meta['bioguide'] in bioguide_map:
                # Try using the bioguide field if name field is the bioguide
                resolved_name = bioguide_map[meta['bioguide']]
                print(f"  Resolved via bioguide {meta['bioguide']} -> {resolved_name}")
                face_name = resolved_name
            elif is_bioguide_id(face_name):
                print(f"  WARNING: Could not resolve bioguide {face_name} to name")
                print(f"           File will be saved as {face_name}.jpg")

            # Generate new filename for faces_db
            # Find next available number for this person
            existing = list(faces_db_dir.glob(f"{face_name}*.jpg"))
            if existing:
                # Extract numbers from existing files
                numbers = []
                for f in existing:
                    match = re.search(r'(\d+)\.jpg$', f.name)
                    if match:
                        numbers.append(int(match.group(1)))
                    elif f.stem == face_name:
                        numbers.append(0)
                next_num = max(numbers, default=0) + 1
                new_name = f"{face_name}{next_num}.jpg"
            else:
                new_name = f"{face_name}.jpg"

            dest_path = faces_db_dir / new_name
            shutil.move(str(image_path), str(dest_path))
            print(f"  APPROVED -> {new_name}")
            stats["approved"] += 1

        elif action == 'r':
            image_path.unlink()
            print("  REJECTED (deleted)")
            stats["rejected"] += 1

        elif action == 's':
            print("  SKIPPED")
            stats["skipped"] += 1

    close_display()

    print("\n" + "=" * 60)
    print(f"Review complete:")
    print(f"  Approved: {stats['approved']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Skipped:  {stats['skipped']}")

    if stats['approved'] > 0:
        print(f"\nRemember to regenerate embeddings:")
        print(f"  python core/warm_embeddings.py --faces_db {faces_db_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Review and approve pending face crops for database augmentation."
    )
    parser.add_argument(
        "--pending-dir",
        type=Path,
        default=PROJECT_ROOT / "02_outputs" / "pending_review",
        help="Directory containing pending face images",
    )
    parser.add_argument(
        "--faces-db",
        type=Path,
        default=PROJECT_ROOT / "01_data" / "faces_official",
        help="Target faces database directory",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve all pending faces without prompting",
    )
    parser.add_argument(
        "--lookup-csv",
        type=Path,
        default=PROJECT_ROOT / "01_data" / "members_face_lookup.csv",
        help="CSV file with bioguide_id to filename mapping",
    )

    args = parser.parse_args()

    # Load bioguide -> name mapping for proper file naming
    bioguide_map = load_bioguide_to_name_map(args.lookup_csv)
    if bioguide_map:
        print(f"Loaded {len(bioguide_map)} bioguide -> name mappings from {args.lookup_csv.name}")
    else:
        print(f"Warning: No bioguide mappings loaded. Files may not be named correctly.")

    review_faces(
        pending_dir=args.pending_dir,
        faces_db_dir=args.faces_db,
        auto_approve=args.auto_approve,
        bioguide_map=bioguide_map,
    )


if __name__ == "__main__":
    main()
