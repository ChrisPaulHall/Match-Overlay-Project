from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type


ENV_PREFIX = "FACE_OVERLAY_"


def _coerce_value(expected_type: Type[Any], raw: str) -> Optional[Any]:
    try:
        if expected_type is bool:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if expected_type is int:
            return int(raw)
        if expected_type is float:
            return float(raw)
        return raw
    except ValueError:
        return None


def _env_lookup(*candidates: str) -> Optional[str]:
    for key in candidates:
        if key in os.environ:
            return os.environ[key]
    return None


@dataclass
class RuntimeDefaults:
    face_threshold: float = 0.50
    ocr_weight: float = 1.2
    beam_min: float = 0.50
    switch_margin: float = 0.06
    ocr_min_face_for_weight: float = 0.20
    cooldown_frames: int = 20
    face_interval: int = 3
    frame_interval: int = 20

    @classmethod
    def from_env(cls) -> "RuntimeDefaults":
        overrides: Dict[str, Any] = {}
        base = cls()
        for f in fields(base):
            env_key = f"{ENV_PREFIX}RUNTIME_{f.name}".upper()
            fallback_key = f"{ENV_PREFIX}{f.name}".upper()
            raw = _env_lookup(env_key, fallback_key)
            if raw is None:
                continue
            coerced = _coerce_value(f.type, raw)
            if coerced is not None:
                overrides[f.name] = coerced
        if overrides:
            return cls(**overrides)
        return base

    def as_dict(self) -> Dict[str, Any]:
        return {
            "face_threshold": self.face_threshold,
            "ocr_weight": self.ocr_weight,
            "beam_min": self.beam_min,
            "switch_margin": self.switch_margin,
            "ocr_min_face_for_weight": self.ocr_min_face_for_weight,
            "cooldown_frames": self.cooldown_frames,
            "face_interval": self.face_interval,
            "frame_interval": self.frame_interval,
        }


@dataclass
class Paths:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    pending_review_dir: Path = field(init=False)
    runtime_config_path: Path = field(init=False)
    overlay_json_path: Path = field(init=False)
    faces_db_default: Path = field(init=False)
    faces_db_variants: List[Path] = field(init=False)

    def __post_init__(self) -> None:
        data_override = _env_lookup(f"{ENV_PREFIX}DATA_DIR", "FACE_DATA_DIR")
        self.data_dir = Path(data_override) if data_override else self.project_root / "01_data"

        output_override = _env_lookup("OVERLAY_OUTPUT_DIR", f"{ENV_PREFIX}OUTPUT_DIR")
        self.output_dir = Path(output_override) if output_override else self.project_root / "02_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pending_override = _env_lookup(f"{ENV_PREFIX}PENDING_REVIEW_DIR")
        self.pending_review_dir = Path(pending_override) if pending_override else self.data_dir / "pending_review"
        self.pending_review_dir.mkdir(parents=True, exist_ok=True)

        runtime_override = _env_lookup("RUNTIME_CONFIG_PATH", f"{ENV_PREFIX}RUNTIME_CONFIG_PATH")
        self.runtime_config_path = Path(runtime_override) if runtime_override else self.output_dir / "runtime_config.json"

        overlay_override = _env_lookup("OVERLAY_JSON_PATH", f"{ENV_PREFIX}OVERLAY_JSON_PATH")
        self.overlay_json_path = Path(overlay_override) if overlay_override else self.output_dir / "overlay_data.json"

        faces_db_override = _env_lookup(f"{ENV_PREFIX}FACES_DB", f"{ENV_PREFIX}FACES_DIR")
        self.faces_db_default = Path(faces_db_override) if faces_db_override else self.data_dir / "faces_official"

        extra_faces_dir = self.data_dir / "faces_preprocessed"
        self.faces_db_variants = [self.faces_db_default]
        if extra_faces_dir != self.faces_db_default:
            self.faces_db_variants.append(extra_faces_dir)


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    runtime_defaults: RuntimeDefaults = field(default_factory=RuntimeDefaults.from_env)
    log_level: str = field(default_factory=lambda: os.getenv(f"{ENV_PREFIX}LOG_LEVEL", "INFO"))
    overlay_port: int = field(
        default_factory=lambda: int(
            _env_lookup("OVERLAY_PORT", f"{ENV_PREFIX}PORT") or "5021"
        )
    )

    def runtime_defaults_dict(self) -> Dict[str, Any]:
        return dict(self.runtime_defaults.as_dict())


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
