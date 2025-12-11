from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .settings import load_settings
except ImportError:  # pragma: no cover - allow running as script
    _settings_path = Path(__file__).resolve().with_name("settings.py")
    _module_name = "settings"
    _spec = importlib.util.spec_from_file_location(_module_name, _settings_path)
    if _spec and _spec.loader:
        _settings_mod = importlib.util.module_from_spec(_spec)
        import sys

        sys.modules[_module_name] = _settings_mod
        _spec.loader.exec_module(_settings_mod)
        load_settings = _settings_mod.load_settings  # type: ignore[attr-defined]
    else:  # pragma: no cover
        raise


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s:%(message)s"


@dataclass
class MatcherConfig:
    """Strongly-typed configuration for the live matcher."""

    loglevel: str = field(default_factory=lambda: load_settings().log_level)
    cam_index: int = 1
    frame_interval: int = field(default_factory=lambda: load_settings().runtime_defaults.frame_interval)
    face_interval: int = field(default_factory=lambda: load_settings().runtime_defaults.face_interval)
    face_topk: int = 3
    beam_min: float = field(default_factory=lambda: load_settings().runtime_defaults.beam_min)
    face_min: float = field(default_factory=lambda: load_settings().runtime_defaults.face_threshold)
    face_min_area_for_match: float = 0.001
    cooldown: int = field(default_factory=lambda: load_settings().runtime_defaults.cooldown_frames)
    id_window: int = 10
    blank_on_unstable: bool = False
    blank_message: str = "Identifying..."
    stable_min: float = 0.25
    stable_min_last_only: float = 0.30
    ocr_weight: float = field(default_factory=lambda: load_settings().runtime_defaults.ocr_weight)
    # Minimum face similarity required for OCR to contribute to the score.
    # Set to 0.0 to allow OCR contribution without any face match.
    ocr_min_face_for_weight: float = field(
        default_factory=lambda: load_settings().runtime_defaults.ocr_min_face_for_weight
    )
    ocr_method: str = "adaptive"
    ocr_cut_full: int = 80
    ocr_cut_last: int = 70
    ocr_last_min_len: int = 4
    retina_backend: str = "retinaface"
    embed_backend: str = "auto"
    faces_db: Optional[str] = None
    trades_source: str = "auto"

    # Identity switching hysteresis
    switch_margin: float = field(default_factory=lambda: load_settings().runtime_defaults.switch_margin)  # lead required to switch away from current identity
    switch_override_margin: float = 0.15  # lead required to switch during cooldown window

    # Derived paths
    script_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    project_root: str = field(init=False)
    data_dir: str = field(init=False)
    output_dir: str = field(init=False)
    pending_review_dir: str = field(init=False)
    match_log_path: Optional[str] = None

    extras: Dict[str, Any] = field(default_factory=dict, repr=False)
    _raw: Optional[argparse.Namespace] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        settings = load_settings()
        paths = settings.paths

        self.project_root = str(paths.project_root)
        self.data_dir = str(paths.data_dir)
        self.output_dir = str(paths.output_dir)
        self.pending_review_dir = str(paths.pending_review_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.pending_review_dir, exist_ok=True)

        # Match logging is disabled by default (empty string = disabled)
        # Set --match_log_path to enable, or pass path to enable logging
        if self.match_log_path is None:
            # Default: disabled (empty string)
            self.match_log_path = ""
        else:
            candidate = str(self.match_log_path).strip()
            if candidate:
                # User provided a path - use it
                self.match_log_path = os.path.abspath(os.path.expanduser(candidate))
            else:
                # User explicitly set empty string - disable logging
                self.match_log_path = ""

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.__dict__:
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        if self._raw is not None and hasattr(self._raw, key):
            return getattr(self._raw, key)
        return default

    def __getattr__(self, item: str) -> Any:
        if item in self.extras:
            return self.extras[item]
        if self._raw is not None and hasattr(self._raw, item):
            return getattr(self._raw, item)
        raise AttributeError(item)


def build_argument_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    runtime_defaults = settings.runtime_defaults

    p = argparse.ArgumentParser(description="Live Face Matcher Overlay")
    p.add_argument("--loglevel", default=settings.log_level, choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--cam_index", type=int, default=1, help="Camera index")
    p.add_argument("--frame_interval", type=int, default=runtime_defaults.frame_interval, help="Process every Nth frame")
    p.add_argument("--face_interval", type=int, default=runtime_defaults.face_interval, help="Only run face matching every N processed frames (throttle)")
    p.add_argument("--face_topk", type=int, default=2, help="Top-K face candidates to consider")

    # OCR controls
    p.add_argument("--ocr_psm", type=int, default=7, help="Tesseract PSM (7=single line, 6=block)")
    p.add_argument("--ocr_invert", action="store_true", help="Invert OCR ROI (white text on dark background)")
    p.add_argument("--ocr_dilate", action="store_true", help="Dilate OCR ROI slightly to thicken strokes")
    p.add_argument("--ocr_y0", type=float, default=0.60, help="OCR ROI start fraction (0.0=top, 1.0=bottom)")
    p.add_argument("--ocr_height", type=float, default=0.40, help="OCR ROI height fraction (0..1)")
    p.add_argument("--ocr_x0", type=float, default=0.30, help="OCR ROI start fraction from left (0.0=left, 1.0=right)")
    p.add_argument("--ocr_width", type=float, default=0.40, help="OCR ROI width fraction (0..1)")
    p.add_argument("--ocr_scale", type=float, default=1.0, help="Upscale factor for OCR ROI (e.g., 2.0)")
    p.add_argument("--ocr_sharpen", action="store_true", help="Apply unsharp mask before thresholding")
    p.add_argument("--ocr_auto_invert", action="store_true", help="Try both inverted/non-inverted images and keep best text")
    p.add_argument("--ocr_method", type=str, default="adaptive", choices=["adaptive", "otsu"], help="Thresholding method")
    p.add_argument("--ocr_cut_full", type=int, default=80, help="OCR full-name score cutoff (0-100)")
    p.add_argument("--ocr_cut_last", type=int, default=70, help="OCR last-name score cutoff (0-100)")
    p.add_argument("--ocr_last_min_len", type=int, default=4, help="Minimum OCR token length for last-name matching")
    p.add_argument("--ocr_lang", type=str, default="eng", help="OCR language(s)")
    p.add_argument("--ocr_whitelist", type=str, default="", help="Optional Tesseract whitelist")
    p.add_argument("--ocr_letters_only", action="store_true", help="Strip digits/punctuation from OCR text")
    p.add_argument("--dump_ocr_debug", action="store_true", help="Write OCR debug PNGs each processed frame")
    p.add_argument("--ocr_bilateral", action="store_true", help="Use bilateral denoise before deskew")
    p.add_argument("--ocr_use_sauvola", action="store_true", help="Use Sauvola threshold when available")
    p.add_argument("--ocr_engine", type=str, default="auto", choices=["tesseract", "paddle", "easyocr", "auto"], help="OCR engine strategy")

    # Fusion / gating
    p.add_argument("--beam_min", type=float, default=runtime_defaults.beam_min, help="Minimum combined score to update overlay")
    p.add_argument("--cooldown", type=int, default=runtime_defaults.cooldown_frames, help="Frames before allowing another update")
    p.add_argument("--id_window", type=int, default=10, help="Temporal smoothing window length")
    p.add_argument("--ocr_weight", type=float, default=runtime_defaults.ocr_weight, help="Multiplier to emphasize OCR vs face")
    p.add_argument(
        "--ocr_min_face_for_weight",
        type=float,
        default=runtime_defaults.ocr_min_face_for_weight,
        help="Minimum face similarity (0..1) required for OCR to add weight; set 0 to disable gating",
    )
    p.add_argument("--face_min", type=float, default=0.50, help="Allow update by face-only when >= this")
    p.add_argument(
        "--face_min_area_for_match",
        type=float,
        default=0.004,
        help=(
            "Minimum face area (as fraction of frame) required for a detection to be considered. "
            "Helps avoid matching distant or tiny faces in wide-room shots."
        ),
    )
    p.add_argument(
        "--face_match_min_similarity",
        type=float,
        default=0.40,
        help=(
            "Minimum cosine similarity (0..1) for a face match to be considered valid. "
            "Prevents forced matching to poor-quality detections. "
            "Recommended: 0.35-0.45. Higher = fewer false positives, more missed matches."
        ),
    )
    p.add_argument("--blank_on_unstable", action="store_true", help="Blank overlay when unstable & no OCR")
    p.add_argument("--stable_min", type=float, default=0.25, help="Min temporal stability when blank_on_unstable")
    p.add_argument("--stable_min_last_only", type=float, default=0.30, help="Extra stability for last-name-only matches")
    p.add_argument("--blank_message", type=str, default="Identifying...", help="Placeholder when blanked")

    # Hysteresis / anti-jitter tuning
    p.add_argument("--switch_margin", type=float, default=0.06, help="Required score lead to switch identities")
    p.add_argument("--switch_override_margin", type=float, default=0.15, help="Required lead to switch during cooldown")

    # Backends / paths
    p.add_argument("--retina_backend", default="retinaface", help="DeepFace detector backend")
    p.add_argument("--faces_db", type=str, default=None, help="Faces DB dir (default=01_data/faces_official)")
    p.add_argument("--embed_backend", type=str, default="auto", choices=["auto", "deepface", "insightface"], help="Embedding backend to use")
    p.add_argument("--insight_det_size", type=str, default="640,640", help="InsightFace detector input size WxH")
    p.add_argument("--insight_det_thresh", type=float, default=0.45, help="InsightFace detection threshold")
    p.add_argument("--match_log_path", type=str, default=None, help="Write JSONL match log to this path (empty string to disable)")

    # Face selection priority
    p.add_argument(
        "--face_priority",
        type=str,
        default="hybrid",
        choices=["auto", "central", "largest", "hybrid"],
        help=(
            "When multiple faces are detected, how to prioritize. "
            "'auto' preserves previous behavior (all faces from face_recognition; largest face from InsightFace). "
            "'central' prefers the face closest to frame center; 'largest' prefers the biggest face; "
            "'hybrid' blends centrality and size (see --face_center_weight)."
        ),
    )
    p.add_argument(
        "--face_max_detections",
        type=int,
        default=1,
        help=(
            "After ranking faces by --face_priority, keep at most this many faces per frame. "
            "Only applied when --face_priority != auto."
        ),
    )
    p.add_argument(
        "--face_center_weight",
        type=float,
        default=0.75,
        help=(
            "Weight for center vs size in 'hybrid' mode. "
            "score = w*centrality + (1-w)*size, where centrality and size are normalized [0,1]."
        ),
    )
    p.add_argument(
        "--face_min_area_frac",
        type=float,
        default=0.002,
        help=(
            "Ignore faces smaller than this fraction of frame area (0..1). "
            "Use 0 to disable size filtering."
        ),
    )
    p.add_argument(
        "--face_min_centrality",
        type=float,
        default=0.30,
        help=(
            "Ignore faces with normalized centrality below this threshold (0..1). "
            "Centrality=1 is center; ~0 is at corners. Use 0 to disable."
        ),
    )

    # Quiver API
    p.add_argument("--quiver_token", type=str, default="", help="Quiver API token")
    p.add_argument("--trades_source", choices=["auto", "live", "csv"], default="auto", help="Where to fetch latest trades")

    # Save diverse face crops for DB augmentation (pending review)
    p.add_argument("--save_face_hits", action="store_true", default=True, help="Save face crops for under-represented members to pending_review/")
    p.add_argument("--save_face_min_confidence", type=float, default=0.70, help="Minimum face similarity to save (0..1)")
    p.add_argument("--save_face_max_confidence", type=float, default=0.85, help="Maximum face similarity to save (captures diversity, not duplicates)")
    p.add_argument("--save_face_max_db_images", type=int, default=10, help="Only save if member has fewer than this many DB images")
    p.add_argument("--save_face_max_per_person", type=int, default=5, help="Maximum pending images to keep per person")
    p.add_argument("--save_face_dir", type=str, default="", help="Override output dir for saved matches")

    # Performance
    p.add_argument("--fast_cache_only", action="store_true", help="Load cached embeddings only; do not compute missing")

    # Smoke test / debug
    p.add_argument("--smoke", action="store_true", help="Run lightweight smoke-check and exit (no camera)")
    p.add_argument("--smoke_name", type=str, default="", help="Person to target for overlay smoke")
    p.add_argument("--smoke_write_overlay", action="store_true", help="Write overlay_data.json during smoke mode")
    p.add_argument("--smoke_ocr_text", type=str, default="", help="Optional OCR text sample for scoring")

    return p


def parse_config(argv: Optional[list[str]] = None) -> MatcherConfig:
    parser = build_argument_parser()
    ns = parser.parse_args(argv)
    kwargs: Dict[str, Any] = {}
    for field in fields(MatcherConfig):
        if field.init and field.name not in {"extras", "_raw"}:
            if hasattr(ns, field.name):
                kwargs[field.name] = getattr(ns, field.name)
    config = MatcherConfig(**kwargs)
    config.extras = {k: v for k, v in vars(ns).items() if k not in kwargs}
    config._raw = ns
    logging.basicConfig(level=getattr(logging, config.loglevel), format=DEFAULT_LOG_FORMAT)
    logging.getLogger("deepface").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return config
