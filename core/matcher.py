from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, NamedTuple, TextIO

import cv2
import numpy as np

OCR_ENABLED = False  # Hard-disable OCR across the pipeline

# Optional xxhash for faster, better hashing
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

try:  # pragma: no cover - allow running as script without package context
    from .config import MatcherConfig, parse_config
    from .faces import FaceMatch, FacePipeline
    from .ocr import OCRProcessor, OCRResult
    from . import overlay
except ImportError:  # pragma: no cover
    from config import MatcherConfig, parse_config  # type: ignore
    from faces import FaceMatch, FacePipeline  # type: ignore
    from ocr import OCRProcessor, OCRResult  # type: ignore
    import overlay  # type: ignore


# ----------------------------------------------------------------------------
# Async Match Logger
# ----------------------------------------------------------------------------

class AsyncMatchLogger:
    """Background thread for async match log writing."""

    def __init__(self, log_path: str, max_queue_size: int = 1000):
        self.log_path = log_path
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.worker = threading.Thread(target=self._worker, daemon=True, name="MatchLogger")
        self.worker.start()
        self._file_handle: Optional[TextIO] = None
        self._shutdown = False

    def _worker(self):
        """Background worker that writes entries to disk."""
        try:
            while True:
                try:
                    entry = self.queue.get(timeout=1.0)
                except queue.Empty:
                    if self._shutdown:
                        break
                    continue

                if entry is None:  # Shutdown signal
                    break

                # Write entry to disk
                try:
                    if not self._file_handle:
                        self._file_handle = open(self.log_path, "a", encoding="utf-8", buffering=8192)

                    self._file_handle.write(json.dumps(entry) + "\n")
                    self._file_handle.flush()
                except Exception as e:
                    logging.error(f"Async match log write failed: {e}")
        finally:
            # Cleanup on shutdown - always runs regardless of how loop exits
            if self._file_handle:
                try:
                    self._file_handle.close()
                except Exception:
                    pass

    def log(self, entry: Dict[str, Any]) -> bool:
        """Queue entry for async writing. Returns False if queue is full."""
        try:
            self.queue.put_nowait(entry)
            return True
        except queue.Full:
            logging.warning("Match log queue full, dropping entry")
            return False

    def shutdown(self, timeout: float = 5.0):
        """Gracefully shutdown the logger."""
        self._shutdown = True
        self.queue.put(None)  # Shutdown signal
        self.worker.join(timeout=timeout)


# ----------------------------------------------------------------------------
# Frame caching for performance
# ----------------------------------------------------------------------------

class FrameCache:
    """Cache OCR results for identical or similar frames.

    Uses xxhash (if available) or SHA-256 for collision-resistant hashing.
    MD5 is NOT used due to known vulnerabilities and high collision probability.
    """
    def __init__(self, max_size: int = 30):
        self.cache: Dict[str, OCRResult] = {}
        self.frame_hashes: deque[str] = deque(maxlen=max_size)
        self.collision_count: int = 0
        self._hash_algo = "xxhash64" if XXHASH_AVAILABLE else "sha256"
        self._lock = threading.Lock()

        if XXHASH_AVAILABLE:
            logging.info("FrameCache: Using xxhash64 for fast, collision-resistant hashing")
        else:
            logging.info("FrameCache: Using SHA-256 (install xxhash for 3-5x speedup: pip install xxhash)")

    def get_hash(self, frame: np.ndarray) -> str:
        """Fast hash of frame using downsampled version.

        Returns 16-character hex hash for good collision resistance.
        Collision probability with 16 hex chars (64 bits):
        - After 1 billion frames: ~0.000027% chance
        - After 1 million frames: ~0.0000027% chance

        This is vastly better than 12-char MD5 (~0.003% at 1M frames).
        """
        if frame is None or frame.size == 0:
            return ""

        # Downsample by 8x for speed (same as before)
        downsampled = frame[::8, ::8].tobytes()

        if XXHASH_AVAILABLE:
            # xxhash is 3-5x faster than SHA-256 and has excellent distribution
            h = xxhash.xxh64(downsampled).hexdigest()[:16]
        else:
            # SHA-256 fallback - cryptographically secure, slower
            h = hashlib.sha256(downsampled).hexdigest()[:16]

        return h

    def get(self, frame: np.ndarray) -> Optional[OCRResult]:
        """Get cached OCR result if frame is similar."""
        h = self.get_hash(frame)
        if not h:
            return None

        with self._lock:
            result = self.cache.get(h)
            if result:
                logging.debug(f"FrameCache hit: {h[:8]}...")
            return result

    def put(self, frame: np.ndarray, result: OCRResult) -> None:
        """Store OCR result for this frame."""
        h = self.get_hash(frame)
        if not h:
            return

        with self._lock:
            # Handle collision: try to make hash unique
            if h in self.cache:
                self.collision_count += 1
                # Try adding frame dimensions to resolve collision
                h_with_dims = f"{h}_{frame.shape[0]}x{frame.shape[1]}"
                if h_with_dims not in self.cache:
                    h = h_with_dims
                else:
                    # Still colliding, use full hash as final fallback
                    downsampled = frame[::8, ::8].tobytes()
                    if XXHASH_AVAILABLE:
                        h = xxhash.xxh64(downsampled).hexdigest()  # Full 16-char hash
                    else:
                        h = hashlib.sha256(downsampled).hexdigest()[:32]  # 32-char hash

                # Log warning on first collision and every 100th thereafter
                if self.collision_count == 1 or self.collision_count % 100 == 0:
                    logging.warning(
                        f"FrameCache collision resolved (#{self.collision_count}): "
                        f"Using extended hash. Algorithm: {self._hash_algo}"
                    )

            # Evict oldest entry if cache is full
            if len(self.frame_hashes) >= self.frame_hashes.maxlen:
                oldest = self.frame_hashes.popleft()
                if oldest:
                    self.cache.pop(oldest, None)

            self.frame_hashes.append(h)
            self.cache[h] = result
            logging.debug(f"FrameCache stored: {h[:8]}... (size={len(self.cache)})")


# ----------------------------------------------------------------------------
# Runtime configuration hot-reload
# ----------------------------------------------------------------------------

class RuntimeConfigLoader:
    """Periodically reload runtime configuration from file."""
    def __init__(self, config_path: str, check_interval: int = 60):
        self.config_path = config_path
        self.check_interval = check_interval  # frames between checks
        self.last_mtime: float = 0.0
        self.last_check_frame: int = 0
        self.runtime_config: Dict = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._load()
    
    def _load(self) -> None:
        """Load configuration from file."""
        # Using contextmanager pattern to handle lock in both locked and unlocked states
        with self._lock:
            if not os.path.exists(self.config_path):
                return
            
            try:
                mtime = os.path.getmtime(self.config_path)
                if mtime <= self.last_mtime:
                    return  # No changes
                
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if loaded:
                        self.runtime_config = loaded
                        self.last_mtime = mtime
                        logging.info(f"Loaded runtime config: {list(loaded.keys())}")
            except Exception as e:
                logging.warning(f"Failed to load runtime config: {e}")
    
    def maybe_reload(self, frame_count: int) -> bool:
        """Check if config should be reloaded. Returns True if reloaded."""
        with self._lock:
            if frame_count - self.last_check_frame < self.check_interval:
                return False
            
            self.last_check_frame = frame_count
            old_mtime = self.last_mtime
            self._load()
            return self.last_mtime > old_mtime
    
    def get(self, key: str, default=None):
        """Get a config value."""
        with self._lock:
            return self.runtime_config.get(key, default)
    
    def get_float(self, key: str, default: float) -> float:
        """Get a float config value."""
        with self._lock:
            val = self.runtime_config.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
            return default

    def get_int(self, key: str, default: Optional[int]) -> Optional[int]:
        """Get an int config value."""
        with self._lock:
            val = self.runtime_config.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return default


class OperatorCommandQueue:
    """Load operator-issued commands (e.g., manual rejects) from disk."""

    def __init__(self, command_path: str):
        self.command_path = command_path
        self.last_mtime: float = 0.0
        self._pending: deque[Dict[str, Any]] = deque()
        self._seen_ids: deque[str] = deque(maxlen=256)
        self._seen_lookup: set[str] = set()

    def poll(self) -> List[Dict[str, Any]]:
        """Return newly observed commands since the previous poll."""
        self._reload()
        if not self._pending:
            return []
        items = list(self._pending)
        self._pending.clear()
        return items

    def _reload(self) -> None:
        if not self.command_path or not os.path.exists(self.command_path):
            return
        try:
            mtime = os.path.getmtime(self.command_path)
        except Exception:
            return
        if mtime <= self.last_mtime:
            return
        try:
            with open(self.command_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            logging.warning("Failed to load operator commands: %s", exc)
            self.last_mtime = mtime
            return

        commands = payload.get("commands", []) if isinstance(payload, dict) else []
        for entry in commands:
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("id") or "").strip()
            if not cid or self._is_seen(cid):
                continue
            self._register_seen(cid)
            self._pending.append(entry)
        self.last_mtime = mtime

    def _is_seen(self, cid: str) -> bool:
        return cid in self._seen_lookup

    def _register_seen(self, cid: str) -> None:
        if cid in self._seen_lookup:
            return
        if len(self._seen_ids) >= self._seen_ids.maxlen:
            old = self._seen_ids.popleft()
            if old in self._seen_lookup:
                self._seen_lookup.remove(old)
        self._seen_ids.append(cid)
        self._seen_lookup.add(cid)


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------

def combined_score(face_sim: float, full_sim: float, last_sim: float,
                   stable_bonus: float = 0.0, meta_bonus: float = 0.0,
                   w_face: float = 0.55, w_full: float = 0.30,
                   w_last: float = 0.10, w_bonus: float = 0.05) -> float:
    f = (w_face * max(0.0, min(1.0, face_sim))
         + w_full * max(0.0, min(1.0, full_sim))
         + w_last * max(0.0, min(1.0, last_sim))
         + w_bonus * (stable_bonus + meta_bonus))
    return float(f)


class PersonKey(NamedTuple):
    bioguide: str
    filename: str

    @property
    def label(self) -> str:
        return self.bioguide or os.path.basename(self.filename) or "(unknown)"

    @property
    def display_name(self) -> str:
        try:
            info = overlay.info_for_person(self.bioguide, self.filename)
        except Exception:
            info = {}
        first = str(info.get("first_name") or "").strip()
        last = str(info.get("last_name") or "").strip()
        if first and last:
            return f"{first[0].upper()}. {last}"
        if last:
            return last
        if first:
            return first
        return self.label

class MatcherRunner:
    def __init__(self, config: MatcherConfig) -> None:
        self.config = config
        overlay.init(config)
        self.face_db = FacePipeline(config)
        self.face_db.preload()
        self.ocr = OCRProcessor(config, overlay.score_ocr_text) if OCR_ENABLED else None

        self.id_window: deque[Optional[PersonKey]] = deque(maxlen=max(5, config.id_window))
        self.last_match: Optional[PersonKey] = None
        self.last_match_frame: int = -config.cooldown
        self.last_overlay_payload: Optional[dict] = None
        self.last_member_role: Optional[str] = None

        # No-match ticker state
        self.no_match_frames: int = 0
        self.in_ticker_mode: bool = False
        self.ticker_threshold_frames: int = 12

        # Performance: Frame caching
        self.frame_cache = FrameCache() if OCR_ENABLED else None

        # Hot-reload: Runtime configuration
        runtime_config_path = os.path.join(config.output_dir, "runtime_config.json")
        self.runtime_config = RuntimeConfigLoader(runtime_config_path, check_interval=60)

        # Operator command channel (manual overrides)
        command_path = os.path.join(config.output_dir, "matcher_commands.json")
        self.command_queue = OperatorCommandQueue(command_path)
        self._suppress_bioguide: Dict[str, int] = {}
        self._suppress_filename: Dict[str, int] = {}
        self._suppress_display: Dict[str, int] = {}

        # Match logging - async for performance
        self.match_log_path = self.config.match_log_path or ""
        self.async_logger: Optional[AsyncMatchLogger] = None
        self._init_match_logger()

        # Camera resource management
        self.video: Optional[cv2.VideoCapture] = None
        self._is_running = False

        # Apply initial runtime config if available
        self._apply_runtime_config()
    
    def _apply_runtime_config(self) -> None:
        """Apply runtime configuration overrides."""
        rc = self.runtime_config
        
        # Face detection thresholds
        face_thresh = rc.get_float("face_threshold", None)
        if face_thresh is not None:
            self.config.face_min = face_thresh
        
        face_area_min = rc.get_float("face_min_area_for_match", None)
        if face_area_min is not None:
            self.config.face_min_area_for_match = face_area_min
        
        # OCR weight
        ocr_weight = rc.get_float("ocr_weight", None)
        if ocr_weight is not None:
            self.config.ocr_weight = ocr_weight
        
        # Beam minimum
        beam_min = rc.get_float("beam_min", None)
        if beam_min is not None:
            self.config.beam_min = beam_min
        
        # Switch margin
        switch_margin = rc.get_float("switch_margin", None)
        if switch_margin is not None:
            self.config.switch_margin = switch_margin
        
        # OCR min face threshold
        ocr_min_face = rc.get_float("ocr_min_face_for_weight", None)
        if ocr_min_face is not None:
            self.config.ocr_min_face_for_weight = ocr_min_face
        
        # Frame intervals
        cooldown = rc.get_int("cooldown_frames", None)
        if cooldown is not None:
            self.config.cooldown = cooldown
        
        face_interval = rc.get_int("face_interval", None)
        if face_interval is not None:
            self.config.face_interval = face_interval
        
        frame_interval = rc.get_int("frame_interval", None)
        if frame_interval is not None:
            self.config.frame_interval = frame_interval

    def _process_operator_commands(self, frame_count: int) -> None:
        """Apply any manual override commands (e.g., reject current identity)."""
        # Drop expired suppressions
        for mapping in (self._suppress_bioguide, self._suppress_filename, self._suppress_display):
            expired_keys = [key for key, expiry in mapping.items() if expiry <= frame_count]
            for key in expired_keys:
                mapping.pop(key, None)

        commands = self.command_queue.poll()
        if not commands:
            return

        for command in commands:
            ctype = str(command.get("type") or "").strip().lower()
            if ctype != "reject":
                logging.debug("Ignoring unsupported operator command type: %s", ctype or "(unknown)")
                continue

            ttl_frames = int(command.get("ttl_frames") or 0)
            ttl_frames = max(15, min(2000, ttl_frames))
            expiry = frame_count + ttl_frames

            bioguide = str(command.get("bioguide") or "").strip()
            filename = os.path.basename(str(command.get("filename") or "").strip())
            display_name = str(command.get("display_name") or "").strip()

            label = display_name or bioguide or filename or "unknown"
            logging.info(
                "Operator reject %s: suppressing %s for %d frames",
                command.get("id"),
                label,
                ttl_frames,
            )

            if bioguide:
                self._suppress_bioguide[bioguide] = expiry
            if filename:
                self._suppress_filename[filename.lower()] = expiry
            if display_name:
                self._suppress_display[display_name.lower()] = expiry

            if self.last_match:
                same_bioguide = bioguide and self.last_match.bioguide == bioguide
                same_filename = filename and os.path.basename(self.last_match.filename or "").lower() == filename.lower()
                same_display = display_name and self.last_match.display_name.lower() == display_name.lower()
                if same_bioguide or same_filename or same_display:
                    logging.info("Clearing current match %s due to operator reject", self.last_match.display_name)
                    self.last_match = None
                    self.last_match_frame = frame_count
                    self.id_window.append(None)

    def _init_match_logger(self) -> None:
        """Initialize async match logger."""
        if not self.match_log_path:
            return
        try:
            log_path = os.path.abspath(os.path.expanduser(self.match_log_path))
            parent = os.path.dirname(log_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            self.match_log_path = log_path
            self.async_logger = AsyncMatchLogger(log_path)
            logging.info("Match log (async) will write to: %s", log_path)
        except Exception:
            logging.exception("Failed to initialize match log at %s", self.match_log_path)
            self.match_log_path = ""
            self.async_logger = None

    # ------------------------------------------------------------------
    # OCR candidate resolution
    # ------------------------------------------------------------------
    def _extract_ocr_candidates(self, text: str) -> Tuple[Dict[PersonKey, Dict[str, float]], Optional[str], Optional[str]]:
        words = overlay.NAME_WORD_RE.findall(text or "")
        ocr_candidates: Dict[PersonKey, Dict[str, float]] = {}
        matched_full = None
        matched_last = None

        def _ensure_entry(key: PersonKey) -> Dict[str, float]:
            entry = ocr_candidates.get(key)
            if entry is None:
                entry = {"full_sim": 0.0, "last_sim": 0.0}
                ocr_candidates[key] = entry
            return entry

        def _key_from_bioguide(bid: str) -> PersonKey:
            filename = overlay.get_primary_filename(bid) or ""
            return PersonKey(bid, os.path.basename(filename) if filename else "")

        def _key_from_filename(fn: str) -> PersonKey:
            fn = os.path.basename(fn or "")
            bid = overlay.get_bioguide_for_filename(fn)
            if bid:
                return PersonKey(bid, fn)
            return PersonKey("", fn)

        # Tri-grams / bi-grams
        for n in (3, 2):
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                tokens = words[i:i+n]
                if not all(overlay.is_valid_token(tok) for tok in tokens):
                    continue
                slug = overlay.slugify(" ".join(tokens))
                if slug not in overlay.FULL_SLUG_SET:
                    continue
                bid = overlay.slug_to_bioguide(slug)
                if bid:
                    key = _key_from_bioguide(bid)
                else:
                    fn = overlay.resolve_full_slug(slug)
                    if not fn:
                        continue
                    key = _key_from_filename(fn)
                entry = _ensure_entry(key)
                entry["full_sim"] = max(entry["full_sim"], 1.0)
                matched_full = " ".join(tokens)

        # Last-name fallback
        if not matched_full:
            for word in words:
                if not overlay.is_valid_token(word):
                    continue
                slug = overlay.slugify(word)
                if slug in overlay.NAME_SUFFIX_SLUGS:
                    continue
                if slug in overlay.LAST_SLUG_SET:
                    bids = overlay.last_slug_bioguide_candidates(slug)
                    if bids:
                        for bid in bids:
                            key = _key_from_bioguide(bid)
                            entry = _ensure_entry(key)
                            entry["last_sim"] = max(entry["last_sim"], 0.9)
                            matched_last = word
                    else:
                        for fn in overlay.resolve_last_slug(slug):
                            key = _key_from_filename(fn)
                            entry = _ensure_entry(key)
                            entry["last_sim"] = max(entry["last_sim"], 0.9)
                            matched_last = word

        # Apply thresholds
        cut_full = max(0, min(100, int(self.config.ocr_cut_full)))
        cut_last = max(0, min(100, int(self.config.ocr_cut_last)))
        filtered: Dict[PersonKey, Dict[str, float]] = {}
        for key, sims in ocr_candidates.items():
            full_score = sims.get("full_sim", 0.0) * 100.0
            last_score = sims.get("last_sim", 0.0) * 100.0
            keep = False
            if full_score >= cut_full:
                keep = True
            if last_score >= cut_last:
                keep = True
            if keep:
                filtered[key] = {
                    "full_sim": sims.get("full_sim", 0.0),
                    "last_sim": sims.get("last_sim", 0.0)
                }
        return filtered, matched_full, matched_last

    # ------------------------------------------------------------------
    # Face candidates
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_face_matches(matches: List[FaceMatch]) -> Dict[PersonKey, FaceMatch]:
        agg: Dict[PersonKey, FaceMatch] = {}
        for match in matches:
            if match.bioguide_id:
                primary = overlay.get_primary_filename(match.bioguide_id) or match.filename
                key = PersonKey(match.bioguide_id, os.path.basename(primary or ""))
            else:
                key = PersonKey("", os.path.basename(match.filename))
            prev = agg.get(key)
            if prev is None or match.similarity > prev.similarity:
                agg[key] = match
        return agg

    # ------------------------------------------------------------------
    # Temporal stability
    # ------------------------------------------------------------------
    def _temporal_stability(self, person: Optional[PersonKey]) -> float:
        if not person or not self.id_window:
            return 0.0
        cnt = sum(1 for x in self.id_window if x == person)
        return cnt / len(self.id_window)

    def _is_suppressed(self, key: Optional[PersonKey], frame_count: int) -> bool:
        if not key:
            return False
        if key.bioguide and self._suppress_bioguide.get(key.bioguide, 0) > frame_count:
            return True
        filename = os.path.basename(key.filename or "")
        if filename and self._suppress_filename.get(filename.lower(), 0) > frame_count:
            return True
        display = key.display_name.lower()
        if display and self._suppress_display.get(display, 0) > frame_count:
            return True
        return False

    # ------------------------------------------------------------------
    # Resource management (context manager support)
    # ------------------------------------------------------------------
    def __enter__(self):
        """Context manager entry - initialize camera."""
        self._open_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - guaranteed cleanup."""
        self._cleanup()
        return False  # Don't suppress exceptions

    def _open_camera(self) -> None:
        """Open camera with validation."""
        if self.video is not None:
            logging.warning("Camera already opened, releasing first")
            self._release_camera()

        logging.info(f"Opening camera index {self.config.cam_index}...")
        self.video = cv2.VideoCapture(self.config.cam_index)

        if not self.video.isOpened():
            self.video = None
            raise RuntimeError(f"Failed to open camera index {self.config.cam_index}")

        # Log camera properties
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        logging.info(f"Camera opened: index={self.config.cam_index}, resolution={width}x{height}, fps={fps:.1f}")

    def _release_camera(self) -> None:
        """Safely release camera resource."""
        if self.video is not None:
            try:
                self.video.release()
                logging.info("Camera released")
            except Exception as e:
                logging.warning(f"Error releasing camera: {e}")
            finally:
                self.video = None

    def _cleanup(self) -> None:
        """Clean up all resources."""
        self._is_running = False
        self._release_camera()

        # Release FacePipeline resources (models)
        if hasattr(self, 'face_db') and self.face_db:
            logging.info("Cleaning up FacePipeline...")
            self.face_db.cleanup()

        # Shutdown async logger
        if self.async_logger:
            logging.info("Shutting down async logger...")
            self.async_logger.shutdown()
            self.async_logger = None

        # Ensure trade cache is flushed before exit
        if hasattr(overlay, "flush_trades_cache"):
            try:
                overlay.flush_trades_cache(force=True)
            except Exception:
                logging.debug("Failed to flush trade cache", exc_info=True)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Main processing loop with robust error handling and retry logic."""
        # Maximum consecutive failures before giving up
        MAX_CONSECUTIVE_FAILURES = 10
        INITIAL_BACKOFF_SECONDS = 0.5
        MAX_BACKOFF_SECONDS = 5.0

        consecutive_failures = 0
        backoff_seconds = INITIAL_BACKOFF_SECONDS
        frame_count = 0

        # Open camera if not already opened (e.g., when using context manager)
        if self.video is None:
            self._open_camera()

        self._is_running = True

        try:
            while self._is_running:
                ok, frame = self.video.read()

                if not ok or frame is None:
                    consecutive_failures += 1
                    logging.error(
                        f"Failed to read from camera ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                    )

                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logging.critical(
                            f"Camera read failed {consecutive_failures} consecutive times. Giving up."
                        )
                        break

                    # Exponential backoff with cap
                    import time
                    time.sleep(backoff_seconds)
                    backoff_seconds = min(backoff_seconds * 2, MAX_BACKOFF_SECONDS)
                    continue

                # Reset failure tracking on successful read
                if consecutive_failures > 0:
                    logging.info(f"Camera recovered after {consecutive_failures} failures")
                    consecutive_failures = 0
                    backoff_seconds = INITIAL_BACKOFF_SECONDS

                frame_count += 1

                # Hot-reload runtime configuration periodically (only at frame boundaries)
                if frame_count % max(1, self.config.frame_interval) == 0:
                    if self.runtime_config.maybe_reload(frame_count):
                        logging.info("Runtime configuration reloaded!")
                self._apply_runtime_config()

                if frame_count % max(1, self.config.frame_interval) != 0:
                    continue

                self._process_operator_commands(frame_count)

                ocr_result = self._run_ocr(frame) if OCR_ENABLED else OCRResult(text="", thresholded=None, roi=None)
                ocr_candidates, matched_full, matched_last = self._extract_ocr_candidates(ocr_result.text)

                face_candidates: Dict[PersonKey, Tuple[float, str]] = {}
                if self._should_run_face(frame_count):
                    matches = self.face_db.match_frame(frame)
                    face_candidates = self._aggregate_face_matches(matches)

                best_key, best_score, score_map = self._pick_candidate(
                    frame_count,
                    ocr_candidates,
                    face_candidates,
                    text=ocr_result.text,
                )

                # Apply hysteresis/anti-jitter: prefer current identity unless new one wins by a margin.
                final_key = best_key
                final_score = best_score
                current_key = self.last_match
                current_score = score_map.get(current_key, 0.0) if current_key else 0.0

                if best_key and current_key and best_key != current_key:
                    within_cooldown = (frame_count - self.last_match_frame) < int(self.config.cooldown)
                    base_margin = float(self.config.switch_margin)
                    override_margin = float(self.config.switch_override_margin)
                    required_lead = override_margin if within_cooldown else base_margin
                    if not (best_score >= current_score + required_lead):
                        # Stick with current identity to avoid a brief flip
                        final_key = current_key
                        final_score = current_score

                emitted = False
                if final_key:
                    if final_score >= self.config.beam_min or self._face_override(final_key, face_candidates, ocr_candidates):
                        self._emit_overlay(final_key, face_candidates, ocr_candidates, final_score, frame, ocr_result.text, frame_count)
                        self.last_match = final_key
                        self.last_match_frame = frame_count
                        emitted = True
                    else:
                        self._maybe_blank(face_candidates, ocr_candidates, final_key, final_score)
                else:
                    self._maybe_blank({}, {}, None, 0.0)

                # Manage ticker mode based on consecutive no-match frames
                if emitted:
                    # Reset ticker mode and counter on successful emission
                    self.no_match_frames = 0
                    if self.in_ticker_mode:
                        self.in_ticker_mode = False
                else:
                    self.no_match_frames += 1
                    if (not self.in_ticker_mode) and self.no_match_frames >= int(self.ticker_threshold_frames):
                        try:
                            ticker_payload = overlay.build_networth_ticker_payload(self.last_member_role)
                            self._write_overlay_payload(ticker_payload)
                            self.in_ticker_mode = True
                        except Exception:
                            logging.exception("Failed to emit net worth ticker payload")

                self.id_window.append(best_key)
                overlay.write_heartbeat(frame_count, self.face_db.model_tag or "")

        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down gracefully...")
        except Exception:
            logging.exception("Fatal error in main processing loop")
            raise
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    def _run_ocr(self, frame: np.ndarray) -> OCRResult:
        # OCR disabled via OCR_ENABLED toggle; keep signature for callers
        return OCRResult(text="", thresholded=None, roi=None)

        # Check cache first for performance
        cached = self.frame_cache.get(frame) if self.frame_cache else None
        if cached is not None:
            return cached

        try:
            result = self.ocr.process_frame(frame)
            
            # Cache the result
            if self.frame_cache:
                self.frame_cache.put(frame, result)
            
            if self.config.dump_ocr_debug and result.thresholded is not None and result.roi is not None:
                try:
                    cv2.imwrite(os.path.join(self.config.output_dir, "ocr_latest_raw.png"), result.roi)
                    cv2.imwrite(os.path.join(self.config.output_dir, "ocr_latest.png"), result.thresholded)
                except Exception:
                    pass
            if self.config.loglevel == "DEBUG":
                logging.debug("OCR text: %s", (result.text or "").strip().replace("\n", " | "))
            return result
        except Exception:
            logging.exception("Error during OCR pass")
            return OCRResult(text="", thresholded=None, roi=None)

    def _should_run_face(self, frame_count: int) -> bool:
        try:
            return (self.config.face_interval <= 1) or ((frame_count // max(1, self.config.frame_interval)) % max(1, self.config.face_interval) == 0)
        except Exception:
            return True

    def _score_single(self, key: PersonKey, text: str,
                      ocr_candidates: Dict[PersonKey, Dict[str, float]],
                      face_candidates: Dict[PersonKey, FaceMatch]) -> float:
        o = ocr_candidates.get(key, {"full_sim": 0.0, "last_sim": 0.0})
        full_sim = float(o.get("full_sim", 0.0))
        last_sim = float(o.get("last_sim", 0.0))
        match = face_candidates.get(key)
        face_sim = float(match.similarity) if match else 0.0
        stable = self._temporal_stability(key)
        meta_bonus = 0.05 if ("(" in text and ")" in text and "-" in text) else 0.0

        # OCR contributes only when we have a corresponding face match at/above threshold
        has_ocr = (full_sim > 0.05) or (last_sim > 0.05)
        min_face_for_ocr = float(self.config.ocr_min_face_for_weight)
        ocr_allowed = has_ocr and (face_sim >= min_face_for_ocr)

        if ocr_allowed and face_sim < 0.35:
            w_face, w_full, w_last, w_bonus = 0.30, 0.50, 0.15, 0.05
        elif not ocr_allowed:
            w_face, w_full, w_last, w_bonus = 0.80, 0.15, 0.05, 0.05
        else:
            w_face, w_full, w_last, w_bonus = 0.55, 0.30, 0.10, 0.05

        if ocr_allowed and face_sim < 0.25:
            w_face = 0.0

        if ocr_allowed:
            ow = float(np.clip(self.config.ocr_weight, 0.25, 4.0))
            w_full *= ow
            w_last *= ow
            s = w_face + w_full + w_last + w_bonus
            if s > 0:
                w_face, w_full, w_last, w_bonus = [w / s for w in (w_face, w_full, w_last, w_bonus)]
        else:
            # Prevent OCR from contributing to the score when disallowed
            full_sim = 0.0
            last_sim = 0.0

        return combined_score(
            face_sim, full_sim, last_sim,
            stable_bonus=stable, meta_bonus=meta_bonus,
            w_face=w_face, w_full=w_full, w_last=w_last, w_bonus=w_bonus,
        )

    def _pick_candidate(self, frame_count: int,
                        ocr_candidates: Dict[PersonKey, Dict[str, float]],
                        face_candidates: Dict[PersonKey, FaceMatch], *,
                        text: str) -> Tuple[Optional[PersonKey], float, Dict[PersonKey, float]]:
        beam = set(ocr_candidates.keys()) | set(face_candidates.keys())
        if not beam:
            if (frame_count - self.last_match_frame) > self.config.cooldown:
                logging.info("No match this frame.")
            return None, -1.0, {}

        best_key: Optional[PersonKey] = None
        best_score = -1.0
        score_map: Dict[PersonKey, float] = {}
        for key in beam:
            if self._is_suppressed(key, frame_count):
                logging.debug("Operator suppress: skipping candidate %s", key.display_name)
                continue
            score = self._score_single(key, text, ocr_candidates, face_candidates)
            score_map[key] = score
            if score > best_score:
                best_score = score
                best_key = key

        if best_key:
            match_meta = face_candidates.get(best_key)
            face_for_log = match_meta.similarity if match_meta else 0.0
            area_for_log = (match_meta.detection_area * 100.0) if (match_meta and match_meta.detection_area) else 0.0
            o_entry = ocr_candidates.get(best_key, {})
            o_full_raw = float(o_entry.get("full_sim", 0.0))
            o_last_raw = float(o_entry.get("last_sim", 0.0))
            has_ocr = (o_full_raw > 0.0) or (o_last_raw > 0.0)
            ocr_allowed = has_ocr and (float(face_for_log) >= float(self.config.ocr_min_face_for_weight))
            o_full_used = o_full_raw if ocr_allowed else 0.0
            o_last_used = o_last_raw if ocr_allowed else 0.0
            stable = self._temporal_stability(best_key)

            # Adaptive logging format based on whether OCR is contributing
            if has_ocr and ocr_allowed:
                # Full format when OCR is active
                logging.info(
                    "Beam pick: %s (score=%.2f) [face=%.2f full=%.2f last=%.2f stable=%.2f area=%.2f%%]",
                    best_key.display_name, best_score,
                    face_for_log, o_full_used, o_last_used, stable, area_for_log
                )
            else:
                # Compact format for face-only matching
                num_candidates = len(face_candidates)
                margin_to_min = face_for_log - float(self.config.face_min)
                logging.info(
                    "Match: %s → face=%.2f (score=%.2f, margin=+%.2f, stable=%.1f%%, n=%d, area=%.2f%%)",
                    best_key.display_name,
                    face_for_log,
                    best_score,
                    margin_to_min,
                    stable * 100,
                    num_candidates,
                    area_for_log
                )
        return best_key, best_score, score_map

    def _face_override(self, key: PersonKey,
                       face_candidates: Dict[PersonKey, FaceMatch],
                       ocr_candidates: Dict[PersonKey, Dict[str, float]]) -> bool:
        match = face_candidates.get(key)
        face_only_sim = float(match.similarity) if match else 0.0
        min_area = float(getattr(self.config, "face_min_area_for_match", 0.0) or 0.0)
        area = float(match.detection_area) if match else 0.0
        o_full = float(ocr_candidates.get(key, {}).get("full_sim", 0.0))
        o_last = float(ocr_candidates.get(key, {}).get("last_sim", 0.0))
        has_ocr = (o_full > 0.0) or (o_last > 0.0)
        if (not has_ocr) and face_only_sim >= float(self.config.face_min):
            if min_area > 0.0 and area < min_area:
                return False
            margin = face_only_sim - float(self.config.face_min)
            logging.info(
                "Face-only override: %s → %.2f (min=%.2f, margin=+%.2f, area=%.2f%%)",
                key.display_name,
                face_only_sim,
                self.config.face_min,
                margin,
                area * 100.0,
            )
            return True
        return False

    def _write_overlay_payload(self, payload: dict) -> bool:
        """Write overlay JSON only when the payload changed."""
        if payload == self.last_overlay_payload:
            return False
        overlay.write_overlay(payload)
        self.last_overlay_payload = payload
        return True

    def _emit_overlay(self, key: PersonKey,
                      face_candidates: Dict[PersonKey, FaceMatch],
                      ocr_candidates: Dict[PersonKey, Dict[str, float]],
                      best_score: float,
                      frame: np.ndarray,
                      ocr_text: str,
                      frame_count: int) -> None:
        match = face_candidates.get(key)
        face_sim = match.similarity if match else 0.0
        candidate_filename = (match.filename if match else key.filename) or overlay.get_primary_filename(key.bioguide) or ""
        face_crop = match.crop if match and match.crop is not None else None
        # Reflect OCR gating in the overlay payload for transparency
        o_entry = ocr_candidates.get(key, {})
        o_full_raw = float(o_entry.get("full_sim", 0.0))
        o_last_raw = float(o_entry.get("last_sim", 0.0))
        has_ocr = (o_full_raw > 0.0) or (o_last_raw > 0.0)
        ocr_allowed = has_ocr and (float(face_sim) >= float(self.config.ocr_min_face_for_weight))
        o_full_used = o_full_raw if ocr_allowed else 0.0
        o_last_used = o_last_raw if ocr_allowed else 0.0
        payload = overlay.build_overlay_payload(
            key.bioguide,
            filename=candidate_filename,
            face_score=face_sim,
            full_sim=o_full_used,
            last_sim=o_last_used,
            combined_score=best_score,
        )
        self._write_overlay_payload(payload)
        role = self._infer_member_role(key, candidate_filename)
        if role:
            self.last_member_role = role
        self._log_match(
            key=key,
            payload=payload,
            candidate_filename=candidate_filename,
            face_sim=face_sim,
            best_score=best_score,
            o_full_raw=o_full_raw,
            o_last_raw=o_last_raw,
            o_full_used=o_full_used,
            o_last_used=o_last_used,
            frame_count=frame_count,
            ocr_text=ocr_text,
            role=role,
        )
        if key.bioguide and face_crop is not None:
            # Save diverse face crops for under-represented members
            overlay.save_pending_review(
                face_crop,
                candidate_filename or key.display_name,
                key.bioguide,
                face_sim,
                best_score,
            )
        elif face_crop is not None:
            # Save unassigned faces (non-congress members like Trump) if function exists
            if hasattr(overlay, 'save_unassigned_face'):
                try:
                    overlay.save_unassigned_face(face_crop,
                                                 face_score=face_sim,
                                                 combined_score=best_score,
                                                 ocr_text=ocr_text,
                                                 source_filename=candidate_filename)
                except Exception as e:
                    logging.debug(f"Could not save unassigned face: {e}")
            else:
                # Function not implemented - log for debugging
                logging.debug(
                    f"Detected non-congress member: {key.display_name} "
                    f"(face={face_sim:.2f}, score={best_score:.2f}, file={candidate_filename})"
                )

    def _log_match(
        self,
        *,
        key: PersonKey,
        payload: Dict[str, Any],
        candidate_filename: str,
        face_sim: float,
        best_score: float,
        o_full_raw: float,
        o_last_raw: float,
        o_full_used: float,
        o_last_used: float,
        frame_count: int,
        ocr_text: str,
        role: Optional[str],
    ) -> None:
        """Write match log entry using atomic file operations."""
        if not self.match_log_path:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "frame": frame_count,
            "bioguide_id": key.bioguide,
            "display_name": key.display_name,
            "filename": candidate_filename,
            "face_sim": round(float(face_sim), 4),
            "combined_score": round(float(best_score), 4),
            "ocr": {
                "full_raw": round(float(o_full_raw), 4),
                "last_raw": round(float(o_last_raw), 4),
                "full_used": round(float(o_full_used), 4),
                "last_used": round(float(o_last_used), 4),
            },
            "ocr_text": (ocr_text or "").replace("\n", " ").strip()[:500],
            "payload": payload,
        }
        if role:
            entry["role"] = role

        # Use async logger (non-blocking)
        if self.async_logger:
            self.async_logger.log(entry)

    def _maybe_blank(self, face_candidates: Dict[PersonKey, FaceMatch],
                     ocr_candidates: Dict[PersonKey, Dict[str, float]],
                     best_key: Optional[PersonKey], best_score: float) -> None:
        # If ticker mode is active, avoid writing a blank overlay that would replace the ticker
        if getattr(self, "in_ticker_mode", False):
            return
        if not self.config.blank_on_unstable:
            return
        if best_key and ocr_candidates.get(best_key, {}).get("full_sim", 0.0) > 0:
            return
        if best_key and ocr_candidates.get(best_key, {}).get("last_sim", 0.0) > 0:
            if self._temporal_stability(best_key) >= float(self.config.stable_min_last_only):
                return
        if best_key and face_candidates.get(best_key, None) and face_candidates[best_key].similarity >= float(self.config.face_min):
            return
        face_sim = face_candidates[best_key].similarity if best_key and face_candidates.get(best_key) else 0.0
        o_data = ocr_candidates.get(best_key, {}) if best_key else {}
        placeholder = {
            "name": self.config.blank_message,
            "image": "",
            "bioguide_id": "",
            "dob": "",
            "age": "",
            "tenure_pretty": "",
            "committees": [],
            "latest_trades": [],
            "quiver_url": "",
            "networth_current": "",
            "top_donors": [],
            "top_industries": [],
            "periods": {},
            "source_url": "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scores": {
                "face": round(float(face_sim), 3),
                "full": round(float(o_data.get("full_sim", 0.0)), 3),
                "last": round(float(o_data.get("last_sim", 0.0)), 3),
                "combined": round(float(best_score), 3),
            }
        }
        self._write_overlay_payload(placeholder)

    def _infer_member_role(self, key: PersonKey, filename: str) -> Optional[str]:
        """Return 'senate' or 'house' if the person is a congress member."""
        if not key:
            return None
        lookup_filename = filename or key.filename or ""
        try:
            info = overlay.info_for_person(key.bioguide, lookup_filename)
        except Exception:
            info = {}
        title = str(info.get("title") or "").strip().lower()
        if title.startswith("sen"):
            return "senate"
        if title.startswith("rep") or title.startswith("del"):
            return "house"
        return None


def main():
    config = parse_config()
    # Use context manager for guaranteed resource cleanup
    with MatcherRunner(config) as runner:
        runner.run()


if __name__ == "__main__":
    main()
