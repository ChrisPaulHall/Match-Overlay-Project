#!/usr/bin/env python3
"""
Renumber Face Database Images
==============================

Renumbers images for each member sequentially without gaps.

Example:
  Before: BHagerty1.jpg, BHagerty3.jpg, BHagerty11.jpg
  After:  BHagerty.jpg, BHagerty1.jpg, BHagerty2.jpg

The first image has no number, subsequent images are numbered 1, 2, 3...

Usage:
    python 03_scripts/renumber_images_in_place.py --db 01_data/faces_official --dry-run
    python 03_scripts/renumber_images_in_place.py --db 01_data/faces_official
"""

import argparse
import logging
import re
from pathlib import Path
from collections import defaultdict
import shutil

logging.basicConfig(level=logging.INFO, format='%(message)s')


def extract_member_name(filename: str) -> str:
    """Extract member base name (without numbers and extension)."""
    # Remove extension
    name = Path(filename).stem
    # Remove trailing numbers
    base_name = re.sub(r'\d+$', '', name)
    return base_name


def natural_sort_key(filename: str):
    """Natural sort key for filenames with numbers."""
    # Extract number if present, else use 0
    match = re.search(r'(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1))
    return 0


def renumber_database(db_dir: Path, dry_run: bool = True):
    """Renumber all images in database sequentially."""

    # Find all images
    image_files = list(db_dir.glob('*.jpg'))

    if not image_files:
        logging.error(f"No images found in {db_dir}")
        return

    logging.info(f"Found {len(image_files)} images in {db_dir}")

    # Group by member
    members = defaultdict(list)
    for img_path in image_files:
        base_name = extract_member_name(img_path.name)
        members[base_name].append(img_path)

    logging.info(f"Found {len(members)} unique members")
    logging.info("")

    # Track changes
    total_renamed = 0
    total_unchanged = 0

    # Process each member
    for member in sorted(members.keys()):
        images = members[member]

        # Sort images naturally (by existing number)
        images.sort(key=lambda x: natural_sort_key(x.name))

        # Determine new names
        rename_map = []
        for i, img_path in enumerate(images):
            if i == 0:
                # First image has no number
                new_name = f"{member}.jpg"
            else:
                # Subsequent images numbered 1, 2, 3...
                new_name = f"{member}{i}.jpg"

            new_path = db_dir / new_name

            # Only rename if different
            if img_path.name != new_name:
                rename_map.append((img_path, new_path))

        # Display changes for this member
        if rename_map:
            logging.info(f"{member}: {len(images)} images, {len(rename_map)} to rename")
            for old_path, new_path in rename_map:
                logging.info(f"  {old_path.name} → {new_path.name}")
            total_renamed += len(rename_map)
        else:
            total_unchanged += len(images)

    logging.info("")
    logging.info("=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total images: {len(image_files)}")
    logging.info(f"To rename: {total_renamed}")
    logging.info(f"Already correct: {total_unchanged}")
    logging.info("")

    if dry_run:
        logging.info("🔍 DRY RUN MODE - No changes made")
        logging.info("Run without --dry-run to apply changes")
        return

    # Confirm before proceeding
    if total_renamed > 0:
        logging.info("⚠️  This will rename files in place!")
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            logging.info("Cancelled by user")
            return

        # Perform renames in two passes to avoid conflicts
        # Pass 1: Rename to temporary names
        logging.info("")
        logging.info("Pass 1: Renaming to temporary names...")
        temp_renames = []

        for member in sorted(members.keys()):
            images = members[member]
            images.sort(key=lambda x: natural_sort_key(x.name))

            for i, img_path in enumerate(images):
                if i == 0:
                    new_name = f"{member}.jpg"
                else:
                    new_name = f"{member}{i}.jpg"

                new_path = db_dir / new_name

                if img_path.name != new_name:
                    # Rename to temp name first
                    temp_name = f"_temp_{img_path.name}"
                    temp_path = db_dir / temp_name

                    img_path.rename(temp_path)
                    temp_renames.append((temp_path, new_path))

        # Pass 2: Rename from temp to final names
        logging.info("Pass 2: Renaming to final names...")
        for temp_path, final_path in temp_renames:
            temp_path.rename(final_path)

        logging.info("")
        logging.info(f"✅ Successfully renamed {total_renamed} images!")


def main():
    parser = argparse.ArgumentParser(
        description='Renumber face database images sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (show what would change, don't modify files)
  python renumber_images.py --db 01_data/faces_official --dry-run

  # Actually rename files
  python renumber_images.py --db 01_data/faces_official

Renumbering scheme:
  First image:  MemberName.jpg    (no number)
  Second image: MemberName1.jpg   (numbered 1)
  Third image:  MemberName2.jpg   (numbered 2)
  etc.
        """
    )

    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='Face database directory to renumber'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would change without modifying files'
    )

    args = parser.parse_args()

    db_dir = Path(args.db)
    if not db_dir.exists():
        logging.error(f"Directory not found: {db_dir}")
        return 1

    renumber_database(db_dir, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    exit(main())
