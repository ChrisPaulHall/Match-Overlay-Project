from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover
    from .config import MatcherConfig
    from . import overlay
except ImportError:  # pragma: no cover
    from config import MatcherConfig  # type: ignore
    import overlay  # type: ignore

# Optional dependencies ------------------------------------------------------
try:  # DeepFace (TensorFlow ArcFace)
    from deepface import DeepFace  # type: ignore
    DEEPFACE_OK = True
except Exception:
    DeepFace = None  # type: ignore
    DEEPFACE_OK = False

try:  # InsightFace (ONNX ArcFace + detection)
    from insightface.app import FaceAnalysis  # type: ignore
    INSIGHT_OK = True
except Exception:
    FaceAnalysis = None  # type: ignore
    INSIGHT_OK = False

try:  # dlib-based detector via face_recognition
    import face_recognition  # type: ignore
    FACEREC_OK = True
except Exception:
    face_recognition = None  # type: ignore
    FACEREC_OK = False


@dataclass
class FaceMatch:
    filename: str
    similarity: float
    bioguide_id: str = ""
    crop: Optional[np.ndarray] = None
    detection_area: float = 0.0


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def cosine_topk(query_vec: np.ndarray, emb: np.ndarray, k: int = 3) -> Tuple[List[int], List[float]]:
    v = query_vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = emb @ v
    idx = np.argsort(-sims)[:k]
    return idx.tolist(), sims[idx].astype(float).tolist()


def assess_face_quality(crop: np.ndarray) -> Tuple[float, bool]:
    """Assess face crop quality using blur detection and brightness analysis.
    
    Args:
        crop: Face image as numpy array (BGR format)
        
    Returns:
        Tuple of (quality_score, is_acceptable) where:
        - quality_score: 0.0 to 1.0, higher is better
        - is_acceptable: True if crop meets minimum quality standards
    """
    try:
        # Convert to grayscale if needed
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        h, w = gray.shape
        if h < 10 or w < 10:
            return 0.0, False
        
        # Blur detection using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(np.var(laplacian))
        # Threshold: typically >100 for sharp, <50 for very blurry
        blur_normalized = min(1.0, blur_score / 150.0)  # Normalize to [0, 1]
        
        # Brightness analysis (avoid over/under-exposed)
        mean_brightness = float(np.mean(gray))
        # Optimal range: 85-170 (avoid too dark <50 or too bright >230)
        if mean_brightness < 50:
            brightness_score = 0.0
        elif mean_brightness > 230:
            brightness_score = 0.3
        elif 85 <= mean_brightness <= 170:
            brightness_score = 1.0  # Optimal range
        else:
            # Gradual falloff outside optimal range
            if mean_brightness < 85:
                brightness_score = (mean_brightness - 50) / 35.0
            else:  # > 170
                brightness_score = 1.0 - ((mean_brightness - 170) / 60.0)
        
        # Combined quality score
        # Weight blur more heavily (70%) than brightness (30%)
        quality_score = 0.7 * blur_normalized + 0.3 * brightness_score

        # Acceptable threshold: must have some sharpness and reasonable brightness
        # Lowered from 0.3 to 0.15 for compressed video (OBS Virtual Camera)
        is_acceptable = quality_score > 0.15
        
        return quality_score, is_acceptable
    except Exception:
        return 0.0, False


def file_sig(path: str) -> str:
    st = os.stat(path)
    return f"{st.st_mtime_ns}:{st.st_size}"


def save_cache(cache_npz: str, cache_json: str, filenames: List[str], emb_matrix: np.ndarray,
               model_tag: str = "unknown", *, file_sigs: Optional[dict[str, str]] = None) -> None:
    np.savez_compressed(cache_npz, emb=emb_matrix.astype(np.float32))
    meta = {"files": filenames, "model_tag": model_tag}
    if file_sigs:
        meta["signatures"] = file_sigs
    with open(cache_json, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)


def load_cache(cache_npz: str, cache_json: str, expect_model_tag: Optional[str] = None):
    if not (os.path.exists(cache_npz) and os.path.exists(cache_json)):
        return None, None, None
    try:
        with open(cache_json, encoding="utf-8") as fh:
            meta = json.load(fh)
        if expect_model_tag and meta.get("model_tag") != expect_model_tag:
            return None, None, None
        data = np.load(cache_npz)
        return meta.get("files", []), data["emb"], meta
    except Exception:
        return None, None, None


class FacePipeline:
    """Handles DB warmup, embedding, and frame-level matching."""

    def __init__(self, config: MatcherConfig) -> None:
        self.config = config
        self.faces_dir = config.faces_db or os.path.join(config.data_dir, "faces_official")
        self.db_filenames: List[str] = []
        self.db_emb: Optional[np.ndarray] = None
        self.embed_dim: Optional[int] = None
        self.model_tag: Optional[str] = None
        self._insight_app = None
        self.last_diagnostics: List[Dict[str, Any]] = []

    def cleanup(self) -> None:
        """Release model resources to prevent memory leaks."""
        if self._insight_app is not None:
            try:
                logging.info("Cleaning up FacePipeline resources...")
                # Delete the InsightFace app to release GPU/CPU memory
                del self._insight_app
                # Force garbage collection to release memory immediately
                import gc
                gc.collect()
                logging.info("FacePipeline resources released")
            except Exception as e:
                logging.warning(f"Error during FacePipeline cleanup: {e}")
            finally:
                self._insight_app = None

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------
    def preload(self) -> None:
        if not os.path.isdir(self.faces_dir):
            logging.warning("Faces DB dir %s not found; face matching disabled", self.faces_dir)
            return
        if not (DEEPFACE_OK or INSIGHT_OK):
            logging.warning("No embedding backend available (DeepFace/InsightFace)")
            return

        logging.info("Initializing face database from %s", self.faces_dir)
        cache_npz = os.path.join(self.faces_dir, "_arcface_cache.npz")
        cache_json = os.path.join(self.faces_dir, "_arcface_cache.json")
        files = sorted(
            f for f in os.listdir(self.faces_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        )
        logging.info("Found %d face images in database", len(files))

        mode = str(self.config.embed_backend or "auto").lower()
        expect_tag: Optional[str]
        if mode == "deepface":
            expect_tag = "deepface_arcface" if DEEPFACE_OK else ("insightface_buffalo_l" if INSIGHT_OK else None)
        elif mode == "insightface":
            expect_tag = "insightface_buffalo_l" if INSIGHT_OK else ("deepface_arcface" if DEEPFACE_OK else None)
        else:
            expect_tag = "deepface_arcface" if DEEPFACE_OK else ("insightface_buffalo_l" if INSIGHT_OK else None)

        logging.info("Loading face embeddings cache (this may take 10-30 seconds)...")
        cached_files, cached_emb, cached_meta = load_cache(cache_npz, cache_json, expect_model_tag=expect_tag)
        if cached_emb is not None:
            logging.info("Cache loaded (%d embeddings)", len(cached_emb))
        else:
            logging.info("No valid cache found, will compute embeddings from scratch")

        logging.info("Verifying file signatures for %d faces...", len(files))
        cached_sigs = (cached_meta.get("signatures") if cached_meta else None) or {}
        cur_sigs = {fn: file_sig(os.path.join(self.faces_dir, fn)) for fn in files}
        logging.info("File signature verification complete")
        fast_only = bool(self.config.get("fast_cache_only", False))

        if cached_files is not None and cached_emb is not None:
            if cached_sigs and cached_files == files and all(cached_sigs.get(fn) == cur_sigs.get(fn) for fn in files):
                self._set_db(cached_files, cached_emb, expect_tag)
                logging.info("Loaded %d cached embeddings (dim=%d, model=%s)", self.db_emb.shape[0], self.embed_dim, self.model_tag)
                return
            if fast_only and not cached_sigs and cached_files == files:
                self._set_db(cached_files, cached_emb, expect_tag)
                logging.info("Loaded %d cached embeddings without signature validation due to --fast_cache_only.", self.db_emb.shape[0])
                return

        emb_map: dict[str, np.ndarray] = {}
        cache_sigs_out: dict[str, str] = {}

        if cached_files is not None and cached_emb is not None and cached_sigs:
            cached_index = {fn: i for i, fn in enumerate(cached_files)}
            for fn in files:
                idx = cached_index.get(fn)
                if idx is None:
                    continue
                if cached_sigs.get(fn) != cur_sigs.get(fn):
                    continue
                emb_map[fn] = np.array(cached_emb[idx], dtype=np.float32)
                cache_sigs_out[fn] = cur_sigs.get(fn, "")
            if emb_map:
                logging.info("Loaded %d cached embeddings; %d require recompute.", len(emb_map), len(files) - len(emb_map))
        elif cached_files is not None and cached_emb is not None and not cached_sigs:
            logging.info("Existing cache missing file signatures; recalculating embeddings to refresh cache metadata.")

        missing = [fn for fn in files if fn not in emb_map]
        if fast_only and missing:
            if emb_map:
                ordered = [fn for fn in files if fn in emb_map]
                if ordered:
                    stack = np.stack([emb_map[fn] for fn in ordered], axis=0)
                    self._set_db(ordered, stack, expect_tag)
                    logging.info("Loaded %d cached embeddings; skipping %d missing due to --fast_cache_only.", len(ordered), len(files) - len(ordered))
            else:
                logging.warning("--fast_cache_only requested but no reusable cached embeddings were found.")
            return

        sel = mode if mode != "auto" else ("deepface" if DEEPFACE_OK else ("insightface" if INSIGHT_OK else "none"))
        detail = (f"detector={self.config.retina_backend}" if sel == "deepface" else "detector=insightface")
        logging.info("Precomputing embeddings for %d DB faces (%s, %s)...", len(missing), sel, detail)

        for i, fn in enumerate(missing, 1):
            path = os.path.join(self.faces_dir, fn)
            img = cv2.imread(path)
            if img is None:
                continue
            emb = self.represent(img)
            if emb is None:
                continue
            emb_map[fn] = emb.astype(np.float32)
            cache_sigs_out[fn] = cur_sigs.get(fn, file_sig(path))
            if i % 50 == 0:
                logging.info("  … %d/%d", i, len(missing))

        if not emb_map:
            logging.warning("No embeddings produced from %s", self.faces_dir)
            return

        ordered = [fn for fn in files if fn in emb_map]
        if not ordered:
            logging.warning("No embeddings aligned with current face list; aborting cache save.")
            return

        stack = np.stack([emb_map[fn] for fn in ordered], axis=0)
        self._set_db(ordered, stack, expect_tag)
        sig_payload = {
            fn: cache_sigs_out.get(fn) or cur_sigs.get(fn) or file_sig(os.path.join(self.faces_dir, fn))
            for fn in ordered
        }
        save_cache(cache_npz, cache_json, self.db_filenames, self.db_emb, model_tag=self.model_tag or "unknown", file_sigs=sig_payload)
        logging.info("Face DB ready with %d embeddings (dim=%d, model=%s). Cache saved.", self.db_emb.shape[0], self.embed_dim, self.model_tag)

    def _set_db(self, filenames: List[str], emb_matrix: np.ndarray, tag: Optional[str]) -> None:
        self.db_filenames = filenames
        self.db_emb = l2norm(emb_matrix.astype(np.float32)) if emb_matrix is not None else None
        self.embed_dim = None if self.db_emb is None else int(self.db_emb.shape[1])
        self.model_tag = tag or self.model_tag or "unknown"

    # ------------------------------------------------------------------
    # Frame-level matching
    # ------------------------------------------------------------------
    def match_frame(self, frame: np.ndarray) -> List[FaceMatch]:
        if self.db_emb is None or not (DEEPFACE_OK or INSIGHT_OK):
            return []

        self.last_diagnostics = []
        crops, crop_meta = self._detect_faces(frame)
        if not crops:
            return []

        # Add minimum similarity threshold to prevent forced matching
        # This is critical for reducing false positives and bias
        min_similarity = float(self.config.get("face_match_min_similarity", 0.40))

        frame_area = float(frame.shape[0] * frame.shape[1]) if frame is not None else 0.0
        min_area_frac_raw = getattr(self.config, "face_min_area_for_match", 0.004)
        try:
            min_area_frac = float(min_area_frac_raw)
        except (TypeError, ValueError):
            min_area_frac = 0.0
        if not np.isfinite(min_area_frac):
            min_area_frac = 0.0
        min_area_frac = max(0.0, min(1.0, min_area_frac))

        matches: List[FaceMatch] = []
        crop_idx = -1
        for idx, crop in enumerate(crops):
            meta_info = crop_meta[idx] if idx < len(crop_meta) else {}
            crop_idx += 1
            diag_entry: Dict[str, Any] = {
                "crop_index": crop_idx,
                "frame_area": frame_area,
            }
            if frame_area > 0:
                crop_area_frac = float((crop.shape[0] * crop.shape[1]) / frame_area)
            else:
                crop_area_frac = 0.0
            diag_entry["crop_area_frac"] = crop_area_frac
            diag_entry["min_area_frac"] = min_area_frac

            if min_area_frac > 0.0 and crop_area_frac < min_area_frac:
                logging.debug(
                    "Skipping face crop below area threshold: %.5f < %.5f",
                    crop_area_frac,
                    min_area_frac,
                )
                diag_entry.update({
                    "status": "area_rejected",
                })
                self.last_diagnostics.append(diag_entry)
                continue

            # Assess face quality before processing - reject blurred/poorly lit faces
            quality_score, is_acceptable = assess_face_quality(crop)
            diag_entry["quality_score"] = float(quality_score)
            diag_entry["quality_ok"] = bool(is_acceptable)
            if not is_acceptable:
                logging.debug(
                    "Skipping low-quality face crop (quality=%.2f, blur detected or poor lighting)",
                    quality_score
                )
                diag_entry.update({
                    "status": "quality_rejected",
                })
                self.last_diagnostics.append(diag_entry)
                continue

            emb = self.represent(crop, detected_face=meta_info.get("insight_face"))
            if emb is None:
                diag_entry.update({
                    "status": "embedding_failed",
                })
                self.last_diagnostics.append(diag_entry)
                continue
            idxs, sims = cosine_topk(emb, self.db_emb, k=int(self.config.face_topk))
            max_sim = float(max(sims)) if len(sims) > 0 else 0.0
            diag_entry["max_similarity"] = max_sim
            diag_entry["similarity_threshold"] = float(min_similarity)
            accepted: List[Tuple[int, float]] = [
                (idx, sim) for idx, sim in zip(idxs, sims) if sim >= min_similarity
            ]
            diag_entry["accepted_matches"] = len(accepted)
            if not accepted:
                diag_entry.update({
                    "status": "below_similarity",
                })
                self.last_diagnostics.append(diag_entry)
                continue

            for idx, sim in zip(idxs, sims):
                # Only include matches above minimum similarity threshold
                # This prevents low-quality matches that disproportionately affect people of color
                if sim >= min_similarity:
                    filename = self.db_filenames[idx]
                    bioguide = overlay.get_bioguide_for_filename(filename) if overlay else ""
                    matches.append(
                        FaceMatch(
                            filename,
                            float(sim),
                            bioguide,
                            crop=crop.copy(),
                            detection_area=crop_area_frac,
                        )
                    )
            top_match = None
            if accepted:
                top_idx, top_sim = accepted[0]
                top_match = {
                    "filename": self.db_filenames[top_idx],
                    "similarity": float(top_sim),
                    "bioguide": overlay.get_bioguide_for_filename(self.db_filenames[top_idx]) if overlay else "",
                }
            diag_entry.update({
                "status": "matched",
                "matched_top": top_match,
            })
            self.last_diagnostics.append(diag_entry)
        
        # Memory optimization: only keep crop for top match to prevent memory leaks
        if matches:
            top_match = max(matches, key=lambda m: m.similarity)
            for m in matches:
                if m != top_match:
                    m.crop = None  # Release memory immediately
            
        return matches

    def represent(self, bgr_img: np.ndarray, *, detected_face: Any = None) -> Optional[np.ndarray]:
        mode = str(self.config.embed_backend or "auto").lower()
        # prefer DeepFace when requested/available
        if mode in {"auto", "deepface"}:
            emb = self._deepface_embed(bgr_img, skip_detection=detected_face is not None)
            if emb is not None:
                self.model_tag = "deepface_arcface"
                return emb
            if mode == "deepface":
                return self._insight_embed(bgr_img, detected_face=detected_face)
        else:  # insightface requested
            emb = self._insight_embed(bgr_img, detected_face=detected_face)
            if emb is not None:
                self.model_tag = "insightface_buffalo_l"
                return emb
            return self._deepface_embed(bgr_img, skip_detection=detected_face is not None)
        # fallback: try insight when auto but deepface failed
        fallback = self._insight_embed(bgr_img, detected_face=detected_face)
        if fallback is not None:
            self.model_tag = "insightface_buffalo_l"
        return fallback

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------
    def _detect_faces(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        crops: List[np.ndarray] = []
        meta: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []
        used_insight = False

        # Prefer InsightFace (GPU/ONNX) for primary detections when available
        if self._ensure_insightface_ready():
            faces = []
            try:
                faces = self._insight_app.get(frame)
            except Exception:
                faces = []
            if faces:
                used_insight = True
                # Convert to (top, right, bottom, left), add small padding like before
                for f in faces:
                    x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                    mh = int(0.05 * (y2 - y1))
                    mw = int(0.05 * (x2 - x1))
                    t, b = max(0, y1 - mh), min(frame.shape[0], y2 + mh)
                    l, r = max(0, x1 - mw), min(frame.shape[1], x2 + mw)
                    candidates.append({
                        "box": (t, r, b, l),
                        "source": "insight",
                        "insight_face": f,
                    })

        # Fallback to dlib/HOG if InsightFace is unavailable or produced no boxes
        if (not candidates) and FACEREC_OK and face_recognition is not None:
            try:
                hog_locs = face_recognition.face_locations(frame, model="hog")
            except Exception:
                hog_locs = []
            for box in hog_locs:
                candidates.append({
                    "box": box,
                    "source": "hog",
                    "insight_face": None,
                })

        if not candidates:
            return [], []

        # Optionally filter and rank faces by centrality or size
        mode = str(self.config.get("face_priority", "auto")).lower()
        topn_default = len(candidates) if (mode == "auto" and not used_insight) else 1 if (mode == "auto" and used_insight) else int(self.config.get("face_max_detections", 1))
        topn = max(1, min(int(topn_default), len(candidates)))

        def _scores_for_loc(box: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
            t, r, b, l = box
            h, w = frame.shape[0], frame.shape[1]
            bw, bh = max(1, r - l), max(1, b - t)
            area = float(bw * bh)
            area_norm = area / float(max(1, w * h))
            cx, cy = (l + r) * 0.5, (t + b) * 0.5
            fx, fy = w * 0.5, h * 0.5
            d = np.hypot(cx - fx, cy - fy)
            diag_half = np.hypot(fx, fy)
            centrality = 1.0 - float(min(1.0, d / max(1e-6, diag_half)))
            # Return also a blended score for hybrid
            w_center = float(self.config.get("face_center_weight", 0.6))
            w_center = float(np.clip(w_center, 0.0, 1.0))
            hybrid = w_center * centrality + (1.0 - w_center) * area_norm
            return area_norm, centrality, hybrid

        # Precompute scores and apply optional gating to drop tiny or very off-center faces
        min_area = float(self.config.get("face_min_area_frac", 0.0) or 0.0)
        min_area = float(np.clip(min_area, 0.0, 1.0))
        min_center = float(self.config.get("face_min_centrality", 0.0) or 0.0)
        min_center = float(np.clip(min_center, 0.0, 1.0))

        scored_all = []
        for entry in candidates:
            box = entry["box"]
            area_norm, centrality, hybrid = _scores_for_loc(box)
            scored_all.append((area_norm, centrality, hybrid, entry))

        if min_area > 0.0 or min_center > 0.0:
            filtered = [
                (a, c, h, data)
                for (a, c, h, data) in scored_all
                if (a >= min_area) and (c >= min_center)
            ]
            # Ensure we don't drop everything; if all filtered, keep originals
            if filtered:
                scored_all = filtered
                candidates = [data for (_, _, _, data) in scored_all]
                topn = max(1, min(topn, len(candidates)))

        if mode in {"largest", "central", "hybrid"}:
            scored = []
            for area_norm, centrality, hyb, data in scored_all:
                if mode == "largest":
                    key = area_norm
                elif mode == "central":
                    key = centrality
                else:
                    key = hyb
                scored.append((key, data))
            # Sort descending by score, keep top-N
            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = [data for _, data in scored[:topn]]
        else:
            # auto: preserve prior behavior — all face_recognition locs, or only the largest InsightFace face
            if used_insight and len(candidates) > 1:
                # pick largest like before
                candidates = [max(candidates, key=lambda data: (data["box"][2] - data["box"][0]) * (data["box"][1] - data["box"][3]))]

        # Produce crops in the ranked order
        for entry in candidates:
            top, right, bottom, left = entry["box"]
            crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
            if crop.size:
                crops.append(crop.copy())  # CRITICAL: Break reference to frame to prevent memory leak
                meta.append({
                    "source": entry.get("source", ""),
                    "insight_face": entry.get("insight_face"),
                })
        if not crops:
            return [], []
        return crops, meta

    def _ensure_insightface_ready(self) -> bool:
        if not INSIGHT_OK or FaceAnalysis is None:
            return False
        if self._insight_app is not None:
            return True
        try:
            self._insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            det_size = (640, 640)
            try:
                sz = str(self.config.get("insight_det_size", "640,640"))
                parts = [int(x.strip()) for x in sz.replace("x", ",").split(",") if x.strip()]
                if len(parts) == 2 and parts[0] > 0 and parts[1] > 0:
                    det_size = (parts[0], parts[1])
            except Exception:
                det_size = (640, 640)
            det_thresh = float(self.config.get("insight_det_thresh", 0.45))
            self._insight_app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
            return True
        except Exception as exc:
            logging.warning("InsightFace init failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Embedding backends
    # ------------------------------------------------------------------
    def _deepface_embed(self, bgr_img: np.ndarray, *, skip_detection: bool = False) -> Optional[np.ndarray]:
        if not DEEPFACE_OK or DeepFace is None:
            return None
        try:
            rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            reps = DeepFace.represent(
                img_path=rgb,
                model_name="ArcFace",
                detector_backend="skip" if skip_detection else self.config.retina_backend,
                enforce_detection=False,
            )
        except Exception:
            return None
        if not reps:
            return None
        return np.asarray(reps[0]["embedding"], dtype=np.float32)

    def _insight_embed(self, bgr_img: np.ndarray, *, detected_face: Any = None) -> Optional[np.ndarray]:
        if detected_face is not None:
            emb = getattr(detected_face, "normed_embedding", None)
            if emb is None:
                emb = getattr(detected_face, "embedding", None)
            if emb is not None:
                return np.asarray(emb, dtype=np.float32)
        if not self._ensure_insightface_ready():
            return None
        try:
            faces = self._insight_app.get(bgr_img)
        except Exception:
            faces = []
        if not faces:
            return None
        areas = [max(1, int((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))) for f in faces]
        f = faces[int(np.argmax(areas))]
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)
