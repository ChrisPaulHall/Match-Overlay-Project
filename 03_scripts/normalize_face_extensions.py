#!/usr/bin/env python3
"""
Rename all image files in the face DB folders so they end with `.jpg`.

If the extension change would collide with an existing file, append the next
available numeric suffix (e.g. `Name1.jpg`, `Name2.jpg`, …) until a free name
is found.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable


def _load_settings():
    try:
        return importlib.import_module("core.settings").load_settings()
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        return importlib.import_module("core.settings").load_settings()


SETTINGS = _load_settings()
# Only process the main faces DB by default to avoid warnings for optional paths
DEFAULT_DIRS = [Path(SETTINGS.paths.faces_db_default)]

IMAGE_EXTS = {".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif"}


def parse_bool(value):
    """Accept common true/false strings for CLI flags."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def split_name(base: str) -> tuple[str, int | None]:
    """Split a filename into (prefix, trailing_number or None)."""
    m = re.match(r"^(.*?)(\d+)$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, None


def iter_target_names(prefix: str, starting_index: int | None, directory: Path) -> Iterable[Path]:
    """Yield candidate target paths for conflict resolution."""
    if starting_index is None:
        yield directory / f"{prefix}.jpg"
        idx = 1
    else:
        yield directory / f"{prefix}{starting_index}.jpg"
        idx = starting_index + 1
    while True:
        yield directory / f"{prefix}{idx}.jpg"
        idx += 1


def normalize_file(path: Path, dry_run: bool) -> tuple[bool, str]:
    """
    Rename a single file if needed.
    Returns (changed?, message).
    """
    if not path.is_file():
        return False, ""

    suffix = path.suffix
    if suffix.lower() == ".jpg":
        # Already normalized, but ensure final extension is exactly lowercase .jpg
        if suffix == ".jpg":
            return False, ""
        # only case differs
    else:
        if suffix.lower() not in IMAGE_EXTS and suffix.lower() != ".jpg":
            return False, ""

    stem = path.stem
    prefix, trailing = split_name(stem)

    # Determine the first candidate name
    if suffix.lower() == ".jpg" and suffix != ".jpg":
        # Case-only change keeps same stem
        candidate_iter = iter_target_names(stem, None, path.parent)
    else:
        candidate_iter = iter_target_names(prefix, trailing, path.parent)

    for candidate in candidate_iter:
        # If candidate is the current file (case-insensitive samefile), handle separately
        same_physical = False
        try:
            if candidate.exists():
                same_physical = os.path.samefile(candidate, path)
        except FileNotFoundError:
            same_physical = False

        if same_physical:
            if candidate.name == path.name:
                # Nothing to change
                return False, ""
            # Adjust case via temp rename
            if dry_run:
                return True, f"[DRY] {path.name} -> {candidate.name}"
            tmp = path.with_name(f"__tmp_case__{uuid.uuid4().hex}.jpg")
            os.rename(path, tmp)
            os.rename(tmp, candidate)
            return True, f"{path.name} -> {candidate.name}"

        if candidate.exists():
            continue

        if dry_run:
            return True, f"[DRY] {path.name} -> {candidate.name}"

        tmp = None
        try:
            tmp = path.with_name(f"__tmp_ext__{uuid.uuid4().hex}{candidate.suffix}")
            os.rename(path, tmp)
            os.rename(tmp, candidate)
        except Exception as exc:
            if tmp and tmp.exists():
                os.rename(tmp, path)
            raise RuntimeError(f"Failed to rename {path.name} -> {candidate.name}: {exc}") from exc
        return True, f"{path.name} -> {candidate.name}"

    return False, ""


def normalize_directory(directory: Path, dry_run: bool) -> list[str]:
    if not directory.exists():
        return [f"❌ Directory does not exist: {directory}"]

    messages: list[str] = []
    for entry in sorted(directory.iterdir()):
        changed, msg = normalize_file(entry, dry_run)
        if changed and msg:
            messages.append(msg)
    if not messages:
        messages.append("No changes needed.")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize face DB image extensions to .jpg (with conflict handling)."
    )
    parser.add_argument(
        "--dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_DIRS],
        help="Directories to process (default: faces_official).",
    )
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="BOOL",
        help="Preview changes without renaming files (optionally pass true/false).",
    )
    args = parser.parse_args()

    for raw_dir in args.dirs:
        directory = Path(raw_dir).expanduser()
        print("=" * 80)
        print(f"Directory: {directory}")
        try:
            results = normalize_directory(directory, args.dry_run)
            for line in results:
                print(line)
        except Exception as exc:
            print(f"❌ Error processing {directory}: {exc}", file=sys.stderr)
    print("=" * 80)
    if args.dry_run:
        print("Run again without --dry-run to apply these changes.")


if __name__ == "__main__":
    main()
