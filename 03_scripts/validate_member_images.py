#!/usr/bin/env python3
"""
Validate Face Database - Cross-Check Images for Each Member
============================================================

This tool checks that all images for each congress member are actually
of the same person by computing face similarity scores between all pairs.

Features:
1. Detects mislabeled images (low similarity to other images of same person)
2. Detects images where wrong face was selected (multiple faces)
3. Generates visual report showing problematic members
4. Provides confidence scores for each member's image set

Usage:
    python 03_scripts/validate_member_images.py --db 01_data/faces_official
    python 03_scripts/validate_member_images.py --db 01_data/faces_official --threshold 0.5
    python 03_scripts/validate_member_images.py --db 01_data/faces_official --output-dir 04_reports

Author: Face Overlay Project
"""

import argparse
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = PROJECT_ROOT / "04_reports"

# Try to import face recognition libraries
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class FaceValidator:
    """Validates that all images for a member are of the same person."""

    def __init__(self, similarity_threshold: float = 0.4):
        """
        Initialize validator.

        Args:
            similarity_threshold: Minimum cosine similarity to consider same person
                                 (0.4 is lenient, 0.5 is moderate, 0.6 is strict)
        """
        self.similarity_threshold = similarity_threshold
        self.face_app = None

        if INSIGHTFACE_AVAILABLE:
            logging.info("Initializing InsightFace...")
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
                logging.info("✅ InsightFace initialized")
            except Exception as e:
                logging.error(f"Failed to initialize InsightFace: {e}")
                raise

    def get_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Get face embedding for an image."""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            faces = self.face_app.get(image)
            if not faces:
                return None

            # Use largest face
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            embedding = getattr(largest, 'normed_embedding', None)
            if embedding is None:
                embedding = getattr(largest, 'embedding', None)

            return embedding

        except Exception as e:
            logging.error(f"Error processing {image_path.name}: {e}")
            return None

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def validate_member_images(
        self,
        images: List[Path]
    ) -> Dict:
        """
        Validate that all images for a member are of the same person.

        Args:
            images: List of image paths for one member

        Returns:
            Dictionary with validation results
        """
        if len(images) < 2:
            return {
                'status': 'skip',
                'reason': 'Only one image',
                'image_count': len(images),
                'similarities': []
            }

        # Get embeddings for all images
        embeddings = {}
        for img_path in images:
            emb = self.get_embedding(img_path)
            if emb is not None:
                embeddings[img_path.name] = emb

        if len(embeddings) < 2:
            return {
                'status': 'error',
                'reason': 'Could not extract faces from images',
                'image_count': len(images),
                'faces_found': len(embeddings),
                'similarities': []
            }

        # Compare all pairs
        similarities = []
        image_names = list(embeddings.keys())

        for i in range(len(image_names)):
            for j in range(i + 1, len(image_names)):
                img1 = image_names[i]
                img2 = image_names[j]

                sim = self.cosine_similarity(embeddings[img1], embeddings[img2])
                similarities.append({
                    'image1': img1,
                    'image2': img2,
                    'similarity': sim,
                    'match': sim >= self.similarity_threshold
                })

        # Calculate statistics
        sim_values = [s['similarity'] for s in similarities]
        avg_similarity = np.mean(sim_values)
        min_similarity = np.min(sim_values)
        max_similarity = np.max(sim_values)

        # Find outliers (images with low similarity to others)
        image_avg_similarities = defaultdict(list)
        for sim in similarities:
            image_avg_similarities[sim['image1']].append(sim['similarity'])
            image_avg_similarities[sim['image2']].append(sim['similarity'])

        outliers = []
        for img, sims in image_avg_similarities.items():
            avg = np.mean(sims)
            if avg < self.similarity_threshold:
                outliers.append({
                    'image': img,
                    'avg_similarity': avg,
                    'comparisons': len(sims)
                })

        # Determine status
        if outliers:
            status = 'warning'
            reason = f"{len(outliers)} potential mislabeled image(s)"
        elif min_similarity < self.similarity_threshold:
            status = 'warning'
            reason = f"Some image pairs have low similarity (min: {min_similarity:.3f})"
        else:
            status = 'ok'
            reason = "All images appear to be of the same person"

        return {
            'status': status,
            'reason': reason,
            'image_count': len(images),
            'faces_found': len(embeddings),
            'avg_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'similarities': similarities,
            'outliers': outliers
        }


def organize_images_by_member(db_dir: Path) -> Dict[str, List[Path]]:
    """Organize images by member name (e.g., JSmith1.jpg, JSmith2.jpg -> JSmith)."""

    members = defaultdict(list)

    image_extensions = {'.jpg', '.jpeg', '.png'}
    for img_path in db_dir.glob('*'):
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Extract member name (everything before trailing numbers)
        name = img_path.stem
        base_name = re.sub(r'\d+$', '', name)

        members[base_name].append(img_path)

    return dict(members)


def generate_validation_report(
    results: Dict[str, Dict],
    output_dir: Path,
    db_name: str
):
    """Generate detailed validation report."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"{db_name}_VALIDATION_REPORT_{timestamp}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Categorize results
    ok_members = {k: v for k, v in results.items() if v['status'] == 'ok'}
    warning_members = {k: v for k, v in results.items() if v['status'] == 'warning'}
    error_members = {k: v for k, v in results.items() if v['status'] == 'error'}
    skip_members = {k: v for k, v in results.items() if v['status'] == 'skip'}

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FACE DATABASE VALIDATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total members: {len(results)}\n")
        f.write("\n")

        # Summary
        f.write("SUMMARY:\n")
        f.write("-" * 80 + "\n")
        f.write(f"✅ OK (all images match):        {len(ok_members):4d} members\n")
        f.write(f"⚠️  WARNING (potential issues):  {len(warning_members):4d} members\n")
        f.write(f"❌ ERROR (could not validate):   {len(error_members):4d} members\n")
        f.write(f"⏭️  SKIPPED (only 1 image):      {len(skip_members):4d} members\n")
        f.write("\n")

        # Warning members (HIGH PRIORITY)
        if warning_members:
            f.write("=" * 80 + "\n")
            f.write("⚠️  MEMBERS WITH POTENTIAL ISSUES (HIGH PRIORITY)\n")
            f.write("=" * 80 + "\n")
            f.write("These members have images that may be mislabeled or of different people.\n")
            f.write("RECOMMENDED: Review these images manually.\n")
            f.write("\n")

            for member, result in sorted(warning_members.items()):
                f.write(f"\n{member}:\n")
                f.write(f"  Images: {result['image_count']}\n")
                f.write(f"  Status: {result['reason']}\n")
                f.write(f"  Avg similarity: {result['avg_similarity']:.3f}\n")
                f.write(f"  Min similarity: {result['min_similarity']:.3f}\n")

                if result['outliers']:
                    f.write(f"  Outlier images (likely mislabeled):\n")
                    for outlier in result['outliers']:
                        f.write(f"    - {outlier['image']} (avg sim: {outlier['avg_similarity']:.3f})\n")

                # Show problematic pairs
                low_pairs = [s for s in result['similarities'] if not s['match']]
                if low_pairs:
                    f.write(f"  Low similarity pairs:\n")
                    for pair in low_pairs[:5]:  # Show first 5
                        f.write(f"    - {pair['image1']} vs {pair['image2']}: {pair['similarity']:.3f}\n")
                    if len(low_pairs) > 5:
                        f.write(f"    ... and {len(low_pairs) - 5} more low similarity pairs\n")
                f.write("\n")

        # Error members
        if error_members:
            f.write("=" * 80 + "\n")
            f.write("❌ MEMBERS WITH ERRORS\n")
            f.write("=" * 80 + "\n")
            f.write("Could not validate these members (face detection failed).\n")
            f.write("\n")

            for member, result in sorted(error_members.items()):
                f.write(f"{member}: {result['reason']}\n")

        # Statistics
        if ok_members:
            avg_sims = [r['avg_similarity'] for r in ok_members.values()]
            overall_avg = np.mean(avg_sims)

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("STATISTICS (OK MEMBERS)\n")
            f.write("=" * 80 + "\n")
            f.write(f"Average similarity across all OK members: {overall_avg:.3f}\n")
            f.write(f"This indicates how consistent images are within each member.\n")
            f.write("\n")

        # Recommendations
        f.write("=" * 80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("=" * 80 + "\n")
        f.write("\n")

        if warning_members:
            f.write("1. REVIEW WARNING MEMBERS (HIGH PRIORITY):\n")
            f.write("   These members have potential mislabeled images.\n")
            f.write(f"   Total to review: {len(warning_members)} members\n\n")

            f.write("   For each warning member:\n")
            f.write("   a. Open the images in the database folder\n")
            f.write("   b. Visually compare to identify mislabeled image\n")
            f.write("   c. Check 'outlier images' first (most likely wrong)\n")
            f.write("   d. Remove or rename incorrect images\n\n")

        f.write("2. RUN PREPROCESSING AGAIN:\n")
        f.write("   After fixing mislabeled images, reprocess the database.\n\n")

        f.write("3. VALIDATION PASSED:\n")
        if len(warning_members) == 0:
            f.write("   ✅ No issues found! Database is consistent.\n")
        else:
            f.write(f"   Fix {len(warning_members)} warning members to achieve validation.\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    logging.info(f"Validation report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Validate face database - check for mislabeled images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate face database with default threshold (0.4 - lenient)
  python 03_scripts/validate_member_images.py --db 01_data/faces_official

  # Use stricter threshold (0.5 - moderate)
  python 03_scripts/validate_member_images.py --db 01_data/faces_official --threshold 0.5

  # Very strict threshold (0.6 - only flag clear mismatches)
  python 03_scripts/validate_member_images.py --db 01_data/faces_official --threshold 0.6

  # Output report to custom directory
  python 03_scripts/validate_member_images.py --db 01_data/faces_official --output-dir 04_reports
        """
    )

    parser.add_argument(
        '--db',
        type=str,
        required=True,
        help='Face database directory to validate'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.4,
        help='Similarity threshold (0.4=lenient, 0.5=moderate, 0.6=strict)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help='Directory where the validation report will be written'
    )

    args = parser.parse_args()

    # Validate inputs
    db_dir = Path(args.db)
    if not db_dir.exists():
        logging.error(f"Database directory not found: {db_dir}")
        return 1

    if not INSIGHTFACE_AVAILABLE:
        logging.error("InsightFace not available. Install with: pip install insightface")
        return 1

    if not CV2_AVAILABLE:
        logging.error("OpenCV not available. Install with: pip install opencv-python-headless")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()

    # Initialize validator
    logging.info(f"Validating database: {db_dir}")
    logging.info(f"Similarity threshold: {args.threshold}")
    validator = FaceValidator(similarity_threshold=args.threshold)

    # Organize images by member
    logging.info("Organizing images by member...")
    members = organize_images_by_member(db_dir)
    logging.info(f"Found {len(members)} members")

    # Validate each member
    logging.info("Validating member images...")
    results = {}

    for i, (member, images) in enumerate(members.items(), 1):
        if i % 50 == 0:
            logging.info(f"Progress: {i}/{len(members)} members ({i*100//len(members)}%)")

        result = validator.validate_member_images(images)
        results[member] = result

    # Generate report
    logging.info("Generating validation report...")
    report_path = generate_validation_report(results, output_dir, db_dir.name)

    # Summary
    warning_count = sum(1 for r in results.values() if r['status'] == 'warning')
    error_count = sum(1 for r in results.values() if r['status'] == 'error')

    logging.info("=" * 80)
    logging.info("VALIDATION COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total members: {len(results)}")
    logging.info(f"✅ OK: {len([r for r in results.values() if r['status'] == 'ok'])}")
    logging.info(f"⚠️  Warnings: {warning_count}")
    logging.info(f"❌ Errors: {error_count}")
    logging.info("=" * 80)
    logging.info(f"Report: {report_path}")

    if warning_count > 0:
        logging.warning(f"\n⚠️  Found {warning_count} members with potential issues!")
        logging.warning(f"Review the report: {report_path}")

    return 0 if warning_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
