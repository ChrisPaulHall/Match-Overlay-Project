#!/usr/bin/env python3
"""
Analyze Pending Faces - Calculate impact of adding pending images to database.

This script analyzes images in pending_review/ and calculates how much each
would improve a member's face recognition coverage. It considers:

1. Diversity: How different is the image from existing ones?
2. Consistency: Is it clearly the same person?
3. Coverage: How many images does the member already have?
4. Quality: Face detection confidence and size

Usage:
    python 03_scripts/analyze_pending_faces.py
    python 03_scripts/analyze_pending_faces.py --threshold 0.4
    python 03_scripts/analyze_pending_faces.py --show-all

Author: Face Overlay Project
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PENDING_DIR = PROJECT_ROOT / "01_data" / "pending_review"
DEFAULT_FACES_DIR = PROJECT_ROOT / "01_data" / "faces_official"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "02_outputs"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "04_reports"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Try imports
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except ImportError:
    INSIGHT_OK = False


@dataclass
class PendingAnalysis:
    """Analysis results for a pending image."""
    filepath: Path
    bioguide: str
    base_name: str
    capture_score: float  # Original detection confidence

    # Computed metrics
    embedding: Optional[np.ndarray] = None
    diversity_score: float = 0.0  # Avg distance from existing
    min_similarity: float = 0.0   # Closest match (consistency check)
    max_similarity: float = 0.0   # Most similar existing
    existing_count: int = 0       # Images already in DB
    face_quality: float = 0.0     # Detection confidence for this analysis

    # Final scores
    impact_score: float = 0.0
    recommendation: str = "unknown"
    reason: str = ""


def load_embedding_cache(cache_dir: Path, faces_dir: Path) -> Tuple[List[str], Optional[np.ndarray]]:
    """Load existing embeddings from cache.

    Checks multiple locations where cache might be stored.
    """
    # Possible cache locations
    candidates = [
        (faces_dir / "_arcface_cache.npz", faces_dir / "_arcface_cache.json"),
        (cache_dir / "face_embeddings.npz", cache_dir / "face_embeddings.json"),
        (cache_dir / "embeddings_cache.npz", cache_dir / "embeddings_cache.json"),
    ]

    for cache_npz, cache_json in candidates:
        if cache_npz.exists() and cache_json.exists():
            try:
                with open(cache_json, encoding="utf-8") as f:
                    meta = json.load(f)
                data = np.load(cache_npz)
                files = meta.get("files", [])
                emb = data["emb"]
                logging.info(f"Found cache at: {cache_npz}")
                return files, emb
            except Exception as e:
                logging.warning(f"Failed to load cache from {cache_npz}: {e}")
                continue

    return [], None


def get_member_embeddings(
    files: List[str],
    embeddings: np.ndarray,
    base_name: str
) -> Tuple[List[str], np.ndarray]:
    """Get all embeddings for a specific member."""
    pattern = re.compile(rf"^{re.escape(base_name)}\d*\.(?:jpg|jpeg|png)$", re.IGNORECASE)

    indices = []
    matched_files = []
    for i, f in enumerate(files):
        fname = Path(f).name
        if pattern.match(fname):
            indices.append(i)
            matched_files.append(fname)

    if not indices:
        return [], np.array([])

    return matched_files, embeddings[indices]


def parse_pending_filename(filename: str) -> Optional[Tuple[str, str, float]]:
    """Parse pending_review filename format.

    Format: BioguideID__BaseName__Score__Timestamp.jpg
    Returns: (base_name, bioguide_id, score) or None
    """
    pattern = r"^([A-Z]\d+)__([A-Za-z]+)\d*__(\d+\.\d+)__\d+_\d+\.jpg$"
    m = re.match(pattern, filename, re.IGNORECASE)
    if m:
        bioguide = m.group(1)
        base_name = re.sub(r'\d+$', '', m.group(2))
        score = float(m.group(3))
        return base_name, bioguide, score
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compute_embedding(face_app, image_path: Path) -> Tuple[Optional[np.ndarray], float]:
    """Compute embedding for an image. Returns (embedding, quality_score).

    Handles both full images and pre-cropped face images.
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None, 0.0

        h, w = img.shape[:2]

        # For small/pre-cropped images, pad to make detection work better
        if max(h, w) < 300:
            # Pad image to at least 300x300 with border
            target = 320
            pad_h = max(0, (target - h) // 2)
            pad_w = max(0, (target - w) // 2)
            img = cv2.copyMakeBorder(
                img, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_CONSTANT, value=(128, 128, 128)
            )

        faces = face_app.get(img)
        if not faces:
            # Try with lower threshold for cropped faces
            return None, 0.0

        # Use largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Get embedding
        emb = getattr(largest, 'normed_embedding', None)
        if emb is None:
            emb = getattr(largest, 'embedding', None)

        if emb is None:
            return None, 0.0

        # Quality = detection score * face size factor
        det_score = getattr(largest, 'det_score', 0.5)
        bbox = largest.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img.shape[0] * img.shape[1]
        size_factor = min(1.0, (face_area / img_area) * 10)  # Normalize

        quality = float(det_score) * (0.5 + 0.5 * size_factor)

        return emb, quality

    except Exception as e:
        logging.debug(f"Error processing {image_path}: {e}")
        return None, 0.0


def analyze_pending_image(
    pending: PendingAnalysis,
    existing_files: List[str],
    existing_embeddings: np.ndarray,
    consistency_threshold: float = 0.35
) -> PendingAnalysis:
    """Analyze a pending image against existing embeddings."""

    if pending.embedding is None:
        pending.recommendation = "error"
        pending.reason = "Could not extract face embedding"
        return pending

    pending.existing_count = len(existing_files)

    if pending.existing_count == 0:
        # No existing images - high impact by default
        pending.diversity_score = 1.0
        pending.min_similarity = 0.0
        pending.max_similarity = 0.0
        pending.impact_score = 0.9 * pending.face_quality
        pending.recommendation = "HIGH"
        pending.reason = "First image for this member"
        return pending

    # Compute similarities to all existing images
    similarities = []
    for i in range(len(existing_embeddings)):
        sim = cosine_similarity(pending.embedding, existing_embeddings[i])
        similarities.append(sim)

    similarities = np.array(similarities)

    pending.min_similarity = float(np.min(similarities))
    pending.max_similarity = float(np.max(similarities))
    avg_similarity = float(np.mean(similarities))

    # Diversity = 1 - average similarity (higher = more different)
    pending.diversity_score = 1.0 - avg_similarity

    # Check consistency - is this actually the same person?
    if pending.max_similarity < consistency_threshold:
        pending.recommendation = "SUSPICIOUS"
        pending.reason = f"Low similarity to existing ({pending.max_similarity:.2f}) - may be wrong person"
        pending.impact_score = 0.0
        return pending

    # Calculate impact score
    # Higher diversity = more impact
    # More existing images = less impact (diminishing returns)
    # Higher quality = more impact

    coverage_factor = 1.0 / (1.0 + np.sqrt(pending.existing_count / 5.0))

    pending.impact_score = (
        pending.diversity_score * 0.4 +
        pending.face_quality * 0.3 +
        coverage_factor * 0.3
    )

    # Determine recommendation
    if pending.impact_score >= 0.5:
        pending.recommendation = "HIGH"
        if pending.existing_count < 5:
            pending.reason = f"Adds diversity, member only has {pending.existing_count} images"
        else:
            pending.reason = f"Good diversity score ({pending.diversity_score:.2f})"
    elif pending.impact_score >= 0.3:
        pending.recommendation = "MEDIUM"
        pending.reason = f"Moderate diversity ({pending.diversity_score:.2f}), {pending.existing_count} existing"
    else:
        pending.recommendation = "LOW"
        if pending.max_similarity > 0.8:
            pending.reason = f"Very similar to existing ({pending.max_similarity:.2f})"
        else:
            pending.reason = f"Member well-covered ({pending.existing_count} images)"

    return pending


def count_member_images(faces_dir: Path, base_name: str) -> int:
    """Count existing images for a member in faces directory."""
    pattern = re.compile(rf"^{re.escape(base_name)}\d*\.(?:jpg|jpeg|png)$", re.IGNORECASE)
    count = 0
    for f in faces_dir.iterdir():
        if pattern.match(f.name):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Analyze pending faces and calculate impact scores',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all pending faces
  python 03_scripts/analyze_pending_faces.py

  # Show all results (including low impact)
  python 03_scripts/analyze_pending_faces.py --show-all

  # Use stricter consistency threshold
  python 03_scripts/analyze_pending_faces.py --threshold 0.45
        """
    )

    parser.add_argument('--pending-dir', type=Path, default=DEFAULT_PENDING_DIR,
                        help='Directory with pending faces')
    parser.add_argument('--faces-db', type=Path, default=DEFAULT_FACES_DIR,
                        help='Face database directory')
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                        help='Directory with embedding cache')
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT_DIR,
                        help='Directory to save analysis report')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Consistency threshold (0.35=lenient, 0.45=strict)')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all results including low impact')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    if not CV2_OK:
        logging.error("OpenCV not available. Install with: pip install opencv-python-headless")
        return 1

    if not INSIGHT_OK:
        logging.error("InsightFace not available. Install with: pip install insightface")
        return 1

    pending_dir = args.pending_dir
    if not pending_dir.exists():
        logging.error(f"Pending directory not found: {pending_dir}")
        return 1

    # Load existing embedding cache
    logging.info("Loading embedding cache...")
    cached_files, cached_embeddings = load_embedding_cache(args.cache_dir, args.faces_db)

    if cached_embeddings is None or len(cached_files) == 0:
        logging.warning("No embedding cache found. Will count files only.")
        cached_files = []
        cached_embeddings = np.array([])
    else:
        logging.info(f"Loaded {len(cached_files)} cached embeddings")

    # Initialize InsightFace with settings optimized for cropped face images
    logging.info("Initializing InsightFace...")
    try:
        face_app = FaceAnalysis(name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        # Use smaller det_size and lower threshold for pre-cropped face images
        face_app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)
    except Exception as e:
        logging.error(f"Failed to initialize InsightFace: {e}")
        return 1

    # Find pending files
    pending_files = []
    for f in sorted(pending_dir.iterdir()):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            parsed = parse_pending_filename(f.name)
            if parsed:
                base_name, bioguide, score = parsed
                pending_files.append(PendingAnalysis(
                    filepath=f,
                    bioguide=bioguide,
                    base_name=base_name,
                    capture_score=score
                ))

    if not pending_files:
        print("\nNo pending faces found in:", pending_dir)
        print("Expected format: BioguideID__BaseName__Score__Timestamp.jpg")
        return 0

    logging.info(f"Found {len(pending_files)} pending images to analyze")

    # Analyze each pending image
    results: List[PendingAnalysis] = []

    for i, pending in enumerate(pending_files):
        if (i + 1) % 10 == 0:
            logging.info(f"Progress: {i+1}/{len(pending_files)}")

        # Compute embedding for pending image
        emb, quality = compute_embedding(face_app, pending.filepath)
        pending.embedding = emb
        pending.face_quality = quality

        # Get existing embeddings for this member
        if len(cached_files) > 0 and cached_embeddings.size > 0:
            member_files, member_embeddings = get_member_embeddings(
                cached_files, cached_embeddings, pending.base_name
            )
        else:
            member_files = []
            member_embeddings = np.array([])

        # If no cached embeddings, just count files
        if len(member_embeddings) == 0:
            pending.existing_count = count_member_images(args.faces_db, pending.base_name)
            if pending.embedding is not None:
                pending.impact_score = 0.5  # Unknown, but has valid embedding
                pending.recommendation = "UNKNOWN"
                pending.reason = f"No cached embeddings to compare ({pending.existing_count} files exist)"
            else:
                pending.recommendation = "error"
                pending.reason = "Could not extract embedding"
        else:
            # Full analysis
            analyze_pending_image(pending, member_files, member_embeddings, args.threshold)

        results.append(pending)

    # Sort by recommendation priority and impact score
    priority = {"HIGH": 0, "MEDIUM": 1, "SUSPICIOUS": 2, "LOW": 3, "UNKNOWN": 4, "error": 5}
    results.sort(key=lambda x: (priority.get(x.recommendation, 99), -x.impact_score))

    # Group by recommendation
    high = [r for r in results if r.recommendation == "HIGH"]
    medium = [r for r in results if r.recommendation == "MEDIUM"]
    low = [r for r in results if r.recommendation == "LOW"]
    suspicious = [r for r in results if r.recommendation == "SUSPICIOUS"]
    unknown = [r for r in results if r.recommendation == "UNKNOWN"]
    errors = [r for r in results if r.recommendation == "error"]

    # Output results
    if args.json:
        output = []
        for r in results:
            output.append({
                "file": r.filepath.name,
                "bioguide": r.bioguide,
                "member": r.base_name,
                "recommendation": r.recommendation,
                "impact_score": round(r.impact_score, 3),
                "diversity": round(r.diversity_score, 3),
                "max_similarity": round(r.max_similarity, 3),
                "existing_count": r.existing_count,
                "quality": round(r.face_quality, 3),
                "reason": r.reason
            })
        print(json.dumps(output, indent=2))
        return 0

    # Generate report content
    def generate_report_lines() -> List[str]:
        lines = []
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines.append("=" * 70)
        lines.append("PENDING FACE ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {timestamp}")
        lines.append(f"Analyzed: {len(results)} pending images")
        lines.append(f"Consistency threshold: {args.threshold}")
        lines.append("")

        def add_group(title: str, items: List[PendingAnalysis], marker: str, show_details: bool = True):
            if not items:
                return
            lines.append("")
            lines.append(f"{marker} {title} ({len(items)}):")
            lines.append("-" * 70)
            for r in items:
                lines.append(f"  {r.filepath.name}")
                lines.append(f"    Member: {r.base_name} ({r.bioguide})")
                if show_details:
                    lines.append(f"    Impact: {r.impact_score:.2f} | Diversity: {r.diversity_score:.2f} | "
                                 f"Quality: {r.face_quality:.2f}")
                    lines.append(f"    Existing: {r.existing_count} images | Max sim: {r.max_similarity:.2f}")
                lines.append(f"    Reason: {r.reason}")
                lines.append("")

        add_group("HIGH IMPACT - Recommend Approve", high, "[HIGH]")
        add_group("MEDIUM IMPACT - Consider Approving", medium, "[MEDIUM]")
        add_group("SUSPICIOUS - Review Carefully", suspicious, "[SUSPICIOUS]")
        add_group("LOW IMPACT - Probably Skip", low, "[LOW]")
        add_group("UNKNOWN - No Cache Data", unknown, "[UNKNOWN]")

        if errors:
            lines.append("")
            lines.append(f"[ERROR] Could not process ({len(errors)}):")
            lines.append("-" * 70)
            for r in errors:
                lines.append(f"  - {r.filepath.name}: {r.reason}")

        # Summary
        lines.append("")
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"  High impact:   {len(high):3d}  (recommend approve)")
        lines.append(f"  Medium impact: {len(medium):3d}  (consider)")
        lines.append(f"  Suspicious:    {len(suspicious):3d}  (review carefully)")
        lines.append(f"  Low impact:    {len(low):3d}  (probably skip)")
        lines.append(f"  Unknown:       {len(unknown):3d}  (no cache data)")
        lines.append(f"  Errors:        {len(errors):3d}  (could not process)")
        lines.append("")

        if high:
            lines.append("Next step: Review high-impact images with:")
            lines.append("  python 03_scripts/import_pending_faces.py")

        return lines

    # Generate report
    report_lines = generate_report_lines()

    # Write to file
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f"pending_analysis_{timestamp}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logging.info(f"Report saved to: {report_path}")

    # Also print to console (with emojis)
    print("\n" + "=" * 70)
    print("PENDING FACE ANALYSIS")
    print("=" * 70)
    print(f"Analyzed: {len(results)} pending images")
    print(f"Consistency threshold: {args.threshold}")
    print()

    def print_group(title: str, items: List[PendingAnalysis], emoji: str, show_details: bool = True):
        if not items:
            return
        print(f"\n{emoji} {title} ({len(items)}):")
        print("-" * 70)
        for r in items:
            print(f"  {r.filepath.name}")
            print(f"    Member: {r.base_name} ({r.bioguide})")
            if show_details:
                print(f"    Impact: {r.impact_score:.2f} | Diversity: {r.diversity_score:.2f} | "
                      f"Quality: {r.face_quality:.2f}")
                print(f"    Existing: {r.existing_count} images | Max sim: {r.max_similarity:.2f}")
            print(f"    Reason: {r.reason}")
            print()

    print_group("HIGH IMPACT - Recommend Approve", high, "\U0001f7e2")
    print_group("MEDIUM IMPACT - Consider Approving", medium, "\U0001f7e1")
    print_group("SUSPICIOUS - Review Carefully", suspicious, "\U0001f534")

    if args.show_all:
        print_group("LOW IMPACT - Probably Skip", low, "\u26aa")
        print_group("UNKNOWN - No Cache Data", unknown, "\u2753")
    else:
        if low:
            print(f"\n\u26aa LOW IMPACT: {len(low)} images (use --show-all to see)")
        if unknown:
            print(f"\u2753 UNKNOWN: {len(unknown)} images (use --show-all to see)")

    if errors:
        print(f"\n\u274c ERRORS: {len(errors)} images could not be processed")
        for r in errors:
            print(f"  - {r.filepath.name}: {r.reason}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  \U0001f7e2 High impact:   {len(high):3d}  (recommend approve)")
    print(f"  \U0001f7e1 Medium impact: {len(medium):3d}  (consider)")
    print(f"  \U0001f534 Suspicious:    {len(suspicious):3d}  (review carefully)")
    print(f"  \u26aa Low impact:    {len(low):3d}  (probably skip)")
    print()

    print(f"Report saved to: {report_path}")

    if high:
        print("\nNext step: Review high-impact images with:")
        print("  python 03_scripts/import_pending_faces.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
