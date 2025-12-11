#!/usr/bin/env python3
"""
Comprehensive Member Database Analysis
======================================

Analyzes face database to identify members with weakest representation.

Combines:
1. Image count per member
2. Image quality scores (blur, brightness, resolution)
3. Face detection validation (verifies InsightFace can detect a face)
4. Cross-reference with members_face_lookup.csv

Generates prioritized CSV of members needing attention.

Usage:
    python 03_scripts/analyze_member_database_strength.py --db 01_data/faces_official
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import numpy as np
from collections import defaultdict
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "04_reports"
DEFAULT_LOOKUP_FILE = PROJECT_ROOT / "01_data" / "members_face_lookup.csv"

# Minimum dimensions for acceptable face images
MIN_WIDTH = 80
MIN_HEIGHT = 80
PREFERRED_MIN_WIDTH = 150
PREFERRED_MIN_HEIGHT = 150

# Try to import face recognition
FACE_APP = None
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(message)s')


def init_face_detector():
    """Initialize InsightFace detector (lazy load)."""
    global FACE_APP
    if FACE_APP is None and INSIGHTFACE_AVAILABLE:
        try:
            FACE_APP = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            FACE_APP.prepare(ctx_id=-1, det_size=(640, 640))
            logging.info("  InsightFace detector initialized")
        except Exception as e:
            logging.warning(f"  Could not initialize InsightFace: {e}")
    return FACE_APP


def detect_faces(image_path: Path) -> Dict:
    """Detect faces in image using InsightFace.

    Returns:
        Dict with 'face_count', 'face_detected', 'multiple_faces', 'detection_error'
    """
    app = init_face_detector()
    if app is None:
        return {
            'face_count': -1,
            'face_detected': None,
            'multiple_faces': False,
            'detection_error': 'InsightFace not available'
        }

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {
                'face_count': 0,
                'face_detected': False,
                'multiple_faces': False,
                'detection_error': 'Could not read image'
            }

        faces = app.get(image)
        face_count = len(faces)

        return {
            'face_count': face_count,
            'face_detected': face_count >= 1,
            'multiple_faces': face_count > 1,
            'detection_error': None
        }
    except Exception as e:
        return {
            'face_count': 0,
            'face_detected': False,
            'multiple_faces': False,
            'detection_error': str(e)
        }


def assess_image_quality(image_path: Path, run_face_detection: bool = True) -> Dict:
    """Assess quality of single image including resolution and face detection."""
    result = {
        'error': None,
        'width': 0,
        'height': 0,
        'blur': 0.0,
        'brightness': 0.0,
        'quality_score': 0.0,
        'resolution_ok': False,
        'resolution_good': False,
        'acceptable': False,
        'face_count': -1,
        'face_detected': None,
        'multiple_faces': False,
        'detection_error': None,
    }

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            result['error'] = 'Could not read image'
            return result

        # Resolution check
        h, w = image.shape[:2]
        result['width'] = w
        result['height'] = h
        result['resolution_ok'] = w >= MIN_WIDTH and h >= MIN_HEIGHT
        result['resolution_good'] = w >= PREFERRED_MIN_WIDTH and h >= PREFERRED_MIN_HEIGHT

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        result['blur'] = laplacian_var

        # Brightness
        brightness = gray.mean()
        result['brightness'] = brightness

        # Quality score components
        blur_score = min(1.0, laplacian_var / 500.0)
        brightness_score = 1.0 if 50 <= brightness <= 230 else 0.5
        resolution_score = 1.0 if result['resolution_good'] else (0.7 if result['resolution_ok'] else 0.3)

        # Combined quality score
        result['quality_score'] = (blur_score * 0.4 + brightness_score * 0.3 + resolution_score * 0.3)
        result['acceptable'] = (
            laplacian_var >= 50 and
            50 <= brightness <= 230 and
            result['resolution_ok']
        )

    except Exception as e:
        result['error'] = str(e)
        return result

    # Face detection (optional, expensive)
    if run_face_detection and INSIGHTFACE_AVAILABLE:
        face_result = detect_faces(image_path)
        result.update(face_result)

    return result


def load_lookup_file(lookup_path: Path) -> Tuple[Dict[str, Dict], Set[str]]:
    """Load members_face_lookup.csv and return mapping by filename and set of bioguide IDs.

    Returns:
        (filename_to_info, all_bioguide_ids)
    """
    filename_to_info = {}
    all_bioguide_ids = set()

    if not lookup_path.exists():
        logging.warning(f"Lookup file not found: {lookup_path}")
        return filename_to_info, all_bioguide_ids

    try:
        with open(lookup_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get('filename', '').strip()
                bioguide = row.get('bioguide_id', '').strip()

                if bioguide:
                    all_bioguide_ids.add(bioguide)

                if filename:
                    # Extract base name (remove extension and numbers)
                    base = re.sub(r'\d*\.jpg$', '', filename, flags=re.IGNORECASE)
                    filename_to_info[base] = {
                        'filename': filename,
                        'bioguide_id': bioguide,
                        'first_name': row.get('first_name', ''),
                        'last_name': row.get('last_name', ''),
                        'title': row.get('title', ''),
                        'party': row.get('party', ''),
                        'state': row.get('state', ''),
                        'chamber': row.get('chamber', ''),
                    }
    except Exception as e:
        logging.error(f"Error loading lookup file: {e}")

    return filename_to_info, all_bioguide_ids


def organize_by_member(db_dir: Path) -> Dict[str, List[Path]]:
    """Organize images by member."""
    members = defaultdict(list)

    for img_path in db_dir.glob('*.jpg'):
        # Extract base name (remove numbers)
        base_name = re.sub(r'\d+$', '', img_path.stem)
        members[base_name].append(img_path)

    return dict(members)


def calculate_member_strength(
    member: str,
    images: List[Path],
    lookup_info: Optional[Dict],
    run_face_detection: bool = True
) -> Dict:
    """Calculate overall database strength for a member."""

    # Image count score (0-1)
    image_count = len(images)
    if image_count >= 8:
        count_score = 1.0
    elif image_count >= 5:
        count_score = 0.8
    elif image_count >= 3:
        count_score = 0.6
    elif image_count >= 2:
        count_score = 0.4
    else:
        count_score = 0.2

    # Analyze each image
    image_details = []
    qualities = []
    faces_detected = 0
    faces_missing = 0
    multiple_faces = 0
    undersized_images = 0

    for img in images:
        q = assess_image_quality(img, run_face_detection=run_face_detection)
        image_details.append({
            'path': str(img),
            'filename': img.name,
            **q
        })

        if 'quality_score' in q and q.get('error') is None:
            qualities.append(q['quality_score'])

        # Face detection stats
        if q.get('face_detected') is True:
            faces_detected += 1
        elif q.get('face_detected') is False:
            faces_missing += 1
        if q.get('multiple_faces'):
            multiple_faces += 1

        # Resolution stats
        if not q.get('resolution_ok', True):
            undersized_images += 1

    # Quality score
    if qualities:
        avg_quality = np.mean(qualities)
        min_quality = np.min(qualities)
        quality_score = (avg_quality * 0.7 + min_quality * 0.3)
    else:
        avg_quality = 0.0
        min_quality = 0.0
        quality_score = 0.0

    # Face detection penalty
    face_detection_score = 1.0
    if run_face_detection and INSIGHTFACE_AVAILABLE and image_count > 0:
        detection_rate = faces_detected / image_count
        face_detection_score = detection_rate

    # Overall strength (weighted combination)
    if run_face_detection and INSIGHTFACE_AVAILABLE:
        strength = (count_score * 0.3 + quality_score * 0.4 + face_detection_score * 0.3)
    else:
        strength = (count_score * 0.4 + quality_score * 0.6)

    # Categorize
    if strength >= 0.8:
        category = 'Strong'
        priority = 4
    elif strength >= 0.6:
        category = 'Good'
        priority = 3
    elif strength >= 0.4:
        category = 'Fair'
        priority = 2
    else:
        category = 'Weak'
        priority = 1

    # Build issues list
    issues = []
    if image_count < 3:
        issues.append(f"Too few images ({image_count})")
    if faces_missing > 0:
        issues.append(f"No face detected in {faces_missing} image(s)")
    if multiple_faces > 0:
        issues.append(f"Multiple faces in {multiple_faces} image(s)")
    if undersized_images > 0:
        issues.append(f"{undersized_images} undersized image(s)")
    if avg_quality < 0.6:
        issues.append(f"Low quality score ({avg_quality:.2f})")

    # Lookup info
    bioguide_id = ''
    full_name = member
    title = ''
    party = ''
    state = ''
    in_lookup = False

    if lookup_info:
        in_lookup = True
        bioguide_id = lookup_info.get('bioguide_id', '')
        first = lookup_info.get('first_name', '')
        last = lookup_info.get('last_name', '')
        if first and last:
            full_name = f"{first} {last}"
        title = lookup_info.get('title', '')
        party = lookup_info.get('party', '')
        state = lookup_info.get('state', '')

    return {
        'member': member,
        'full_name': full_name,
        'bioguide_id': bioguide_id,
        'title': title,
        'party': party,
        'state': state,
        'in_lookup': in_lookup,
        'image_count': image_count,
        'count_score': count_score,
        'quality_score': quality_score,
        'avg_quality': avg_quality,
        'min_quality': min_quality,
        'faces_detected': faces_detected,
        'faces_missing': faces_missing,
        'multiple_faces_count': multiple_faces,
        'undersized_images': undersized_images,
        'face_detection_rate': faces_detected / image_count if image_count > 0 else 0.0,
        'strength': strength,
        'category': category,
        'priority': priority,
        'issues': '; '.join(issues) if issues else '',
        'images': [img.name for img in sorted(images)],
        'image_paths': [str(img) for img in sorted(images)],
        'image_details': image_details,
    }


def generate_csv_report(
    member_stats: List[Dict],
    missing_from_db: List[Dict],
    output_dir: Path,
    db_dir: Path,
    db_name: str
) -> Path:
    """Generate CSV analysis report."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = output_dir / f"{db_name}_strength_analysis_{timestamp}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by strength (weakest first)
    sorted_members = sorted(member_stats, key=lambda x: x['strength'])

    # Write main CSV
    fieldnames = [
        'member', 'full_name', 'bioguide_id', 'title', 'party', 'state',
        'in_lookup', 'image_count', 'strength', 'category', 'priority',
        'avg_quality', 'min_quality', 'faces_detected', 'faces_missing',
        'multiple_faces_count', 'undersized_images', 'face_detection_rate',
        'issues', 'images'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for m in sorted_members:
            row = {**m}
            row['images'] = '|'.join(m['images'])  # Join image names with pipe
            writer.writerow(row)

    # Write missing members CSV if any
    if missing_from_db:
        missing_csv = output_dir / f"{db_name}_missing_members_{timestamp}.csv"
        missing_fields = ['bioguide_id', 'first_name', 'last_name', 'title', 'party', 'state', 'chamber', 'expected_filename']
        with open(missing_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=missing_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(missing_from_db)
        logging.info(f"✅ Missing members CSV: {missing_csv}")

    # Write problematic images CSV (no face detected or multiple faces)
    problem_images = []
    for m in member_stats:
        for detail in m.get('image_details', []):
            if detail.get('face_detected') is False or detail.get('multiple_faces'):
                problem_images.append({
                    'member': m['member'],
                    'filename': detail['filename'],
                    'path': detail['path'],
                    'face_count': detail.get('face_count', -1),
                    'face_detected': detail.get('face_detected'),
                    'multiple_faces': detail.get('multiple_faces'),
                    'detection_error': detail.get('detection_error', ''),
                    'width': detail.get('width', 0),
                    'height': detail.get('height', 0),
                    'quality_score': detail.get('quality_score', 0),
                })

    if problem_images:
        problems_csv = output_dir / f"{db_name}_problem_images_{timestamp}.csv"
        problem_fields = ['member', 'filename', 'path', 'face_count', 'face_detected',
                         'multiple_faces', 'detection_error', 'width', 'height', 'quality_score']
        with open(problems_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=problem_fields)
            writer.writeheader()
            writer.writerows(problem_images)
        logging.info(f"✅ Problem images CSV: {problems_csv}")

    logging.info(f"✅ Strength analysis CSV: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze member database strength and identify weak members'
    )
    parser.add_argument('--db', required=True, help='Face database directory')
    parser.add_argument(
        '--output-dir',
        default=str(DEFAULT_REPORT_DIR),
        help='Directory where reports will be saved'
    )
    parser.add_argument(
        '--lookup',
        default=str(DEFAULT_LOOKUP_FILE),
        help='Path to members_face_lookup.csv for cross-reference'
    )
    parser.add_argument(
        '--skip-face-detection',
        action='store_true',
        help='Skip face detection (faster but less accurate)'
    )

    args = parser.parse_args()

    db_dir = Path(args.db)
    if not db_dir.exists():
        logging.error(f"Database not found: {db_dir}")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    lookup_path = Path(args.lookup)
    run_face_detection = not args.skip_face_detection

    logging.info(f"Analyzing database: {db_dir}")

    if run_face_detection:
        if INSIGHTFACE_AVAILABLE:
            logging.info("Face detection: ENABLED")
        else:
            logging.info("Face detection: DISABLED (InsightFace not available)")
            run_face_detection = False
    else:
        logging.info("Face detection: SKIPPED (--skip-face-detection)")

    # Load lookup file
    logging.info(f"Loading lookup file: {lookup_path}")
    lookup_by_filename, all_bioguides = load_lookup_file(lookup_path)
    logging.info(f"  Found {len(lookup_by_filename)} members in lookup file")

    # Organize by member
    members = organize_by_member(db_dir)
    logging.info(f"Found {len(members)} members in database")

    # Find members in lookup but missing from DB
    db_member_set = set(members.keys())
    lookup_member_set = set(lookup_by_filename.keys())
    missing_from_db = []

    for base_name in lookup_member_set - db_member_set:
        info = lookup_by_filename[base_name]
        missing_from_db.append({
            'bioguide_id': info.get('bioguide_id', ''),
            'first_name': info.get('first_name', ''),
            'last_name': info.get('last_name', ''),
            'title': info.get('title', ''),
            'party': info.get('party', ''),
            'state': info.get('state', ''),
            'chamber': info.get('chamber', ''),
            'expected_filename': info.get('filename', ''),
        })

    if missing_from_db:
        logging.info(f"⚠️  {len(missing_from_db)} members in lookup but MISSING from face DB")

    # Find members in DB but not in lookup
    orphan_members = db_member_set - lookup_member_set
    if orphan_members:
        logging.info(f"⚠️  {len(orphan_members)} members in DB but NOT in lookup file")

    # Analyze each member
    logging.info("Analyzing member strength...")
    if run_face_detection:
        logging.info("  (Running face detection - this may take a while)")

    member_stats = []
    total = len(members)

    for i, (member, images) in enumerate(members.items(), 1):
        if i % 50 == 0 or i == total:
            logging.info(f"  Progress: {i}/{total} ({i*100//total}%)")

        lookup_info = lookup_by_filename.get(member)
        stats = calculate_member_strength(
            member, images, lookup_info,
            run_face_detection=run_face_detection
        )
        member_stats.append(stats)

    # Generate reports
    logging.info("Generating reports...")
    csv_path = generate_csv_report(
        member_stats,
        missing_from_db,
        output_dir,
        db_dir,
        db_dir.name
    )

    # Summary
    weak = [m for m in member_stats if m['category'] == 'Weak']
    fair = [m for m in member_stats if m['category'] == 'Fair']
    good = [m for m in member_stats if m['category'] == 'Good']
    strong = [m for m in member_stats if m['category'] == 'Strong']

    avg_strength = np.mean([m['strength'] for m in member_stats])

    total_faces_missing = sum(m['faces_missing'] for m in member_stats)
    total_multiple_faces = sum(m['multiple_faces_count'] for m in member_stats)
    total_undersized = sum(m['undersized_images'] for m in member_stats)

    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total members in DB: {len(member_stats)}")
    logging.info(f"  🔴 Weak:   {len(weak):4d} ({len(weak)*100//len(member_stats):2d}%)")
    logging.info(f"  🟡 Fair:   {len(fair):4d} ({len(fair)*100//len(member_stats):2d}%)")
    logging.info(f"  🟢 Good:   {len(good):4d} ({len(good)*100//len(member_stats):2d}%)")
    logging.info(f"  🔵 Strong: {len(strong):4d} ({len(strong)*100//len(member_stats):2d}%)")
    logging.info(f"Overall strength: {avg_strength:.3f}")

    if run_face_detection:
        logging.info("-" * 40)
        logging.info("Face Detection Issues:")
        logging.info(f"  Images with NO face detected: {total_faces_missing}")
        logging.info(f"  Images with MULTIPLE faces:   {total_multiple_faces}")
        logging.info(f"  Undersized images:            {total_undersized}")

    if missing_from_db:
        logging.info("-" * 40)
        logging.info(f"⚠️  {len(missing_from_db)} active members MISSING from face DB")

    logging.info("=" * 80)
    logging.info(f"\nReports saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
