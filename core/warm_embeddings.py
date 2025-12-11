from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
import numpy as np, cv2


def file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime_ns)}:{int(st.st_size)}"
    except FileNotFoundError:
        return ""
    except Exception:
        return ""


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def save_cache(
    cache_npz: str,
    cache_json: str,
    filenames: List[str],
    emb_matrix: np.ndarray,
    model_tag: str,
    *,
    file_sigs: Dict[str, str] | None = None,
):
    np.savez_compressed(cache_npz, emb=emb_matrix.astype(np.float32))
    meta = {"files": filenames, "model_tag": model_tag}
    if file_sigs:
        meta["signatures"] = file_sigs
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f)


def load_cache(cache_npz: str, cache_json: str) -> Tuple[List[str], np.ndarray | None, str, Dict[str, str]]:
    if not (os.path.exists(cache_npz) and os.path.exists(cache_json)):
        return [], None, "", {}
    try:
        with open(cache_json, encoding="utf-8") as f:
            meta = json.load(f)
        data = np.load(cache_npz)
        files = meta.get("files", [])
        model_tag = meta.get("model_tag", "")
        emb = data["emb"]
        sigs = meta.get("signatures", {}) or {}
        return files, emb, model_tag, sigs
    except Exception:
        return [], None, "", {}


def get_deepface_embedder(backend: str):
    try:
        from deepface import DeepFace  # type: ignore
    except Exception:
        return None

    def _embed(img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            reps = DeepFace.represent(
                img_path=rgb,
                model_name="ArcFace",
                detector_backend=backend,
                enforce_detection=False,
            )
        except Exception:
            return None
        if not reps:
            return None
        return np.array(reps[0]["embedding"], dtype=np.float32)

    return _embed, "deepface_arcface"


def get_insightface_embedder(det_size: tuple[int,int] | None = None, det_thresh: float | None = None):
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        return None

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])  # CPU OK
    try:
        if det_size is None:
            det_size = (640, 640)
        if det_thresh is None:
            det_thresh = 0.45
        app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
    except Exception:
        return None

    def _embed(img_bgr: np.ndarray):
        try:
            faces = app.get(img_bgr)
        except Exception:
            faces = []
        if not faces:
            return None
        # largest face
        areas = [max(1, int((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))) for f in faces]
        f = faces[int(np.argmax(areas))]
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            emb = getattr(f, "embedding", None)
        if emb is None:
            return None
        return np.asarray(emb, dtype=np.float32)

    return _embed, "insightface_buffalo_l"


def build_or_update_cache(
    faces_dir: str,
    backend: str = "skip",
    update_only: bool = True,
    force: bool = False,
    embed_backend: str = "auto",
    insight_det_size: tuple[int,int] | None = None,
    insight_det_thresh: float | None = None,
):
    logging.info("Faces dir: %s", faces_dir)
    cache_npz = os.path.join(faces_dir, "_arcface_cache.npz")
    cache_json = os.path.join(faces_dir, "_arcface_cache.json")

    # Discover files
    files = sorted([f for f in os.listdir(faces_dir) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))])
    logging.info("Found %d face images", len(files))

    # Try to load existing cache
    cur_sigs = {fn: file_sig(os.path.join(faces_dir, fn)) for fn in files}

    cached_files, cached_emb, cached_tag, cached_sigs = load_cache(cache_npz, cache_json)
    if cached_files and cached_emb is not None and not cached_sigs:
        logging.info("Existing cache lacks file signatures; treating as stale and rebuilding from scratch")
        cached = {}
        cached_files = []
        cached_emb = None
    else:
        cached = {}
        if cached_files and cached_emb is not None and cached_sigs:
            for i, fn in enumerate(cached_files):
                if cached_sigs.get(fn) and cached_sigs.get(fn) == cur_sigs.get(fn):
                    cached[fn] = i

    # Choose embedder per requested backend
    embed_backend = (embed_backend or "auto").lower()
    embedder = None
    model_tag = ""
    # Only initialize the requested primary backend; defer fallback initialization until needed
    deep = None if embed_backend == "insightface" else get_deepface_embedder(backend)
    ins  = None if embed_backend == "deepface" else get_insightface_embedder(insight_det_size, insight_det_thresh)
    if embed_backend == "deepface":
        if deep is None:
            # try to initialize InsightFace on-demand for fallback
            if ins is None:
                ins = get_insightface_embedder(insight_det_size, insight_det_thresh)
            if ins is None:
                raise RuntimeError("No embedding backend available (DeepFace/InsightFace)")
            logging.warning("DeepFace unavailable; falling back to InsightFace for cache build")
            embedder, model_tag = ins
            logging.info("Using InsightFace (buffalo_l)")
        else:
            embedder, model_tag = deep
            logging.info("Using DeepFace ArcFace (backend=%s)", backend)
    elif embed_backend == "insightface":
        if ins is None:
            # Try to fallback to DeepFace only if available
            if deep is None:
                deep = get_deepface_embedder(backend)
            if deep is None:
                raise RuntimeError("No embedding backend available (InsightFace requested; DeepFace fallback unavailable)")
            logging.warning("InsightFace unavailable; falling back to DeepFace for cache build")
            embedder, model_tag = deep
            logging.info("Using DeepFace ArcFace (backend=%s)", backend)
        else:
            embedder, model_tag = ins
            logging.info("Using InsightFace (buffalo_l)")
    else:  # auto
        if deep is not None:
            embedder, model_tag = deep
            logging.info("Using DeepFace ArcFace (backend=%s)", backend)
        elif ins is not None:
            embedder, model_tag = ins
            logging.info("Using InsightFace (buffalo_l)")
        else:
            raise RuntimeError("Neither DeepFace nor InsightFace available for embeddings")

    # If force or model changed, rebuild from scratch
    if force or (cached and cached_tag and cached_tag != model_tag):
        logging.info("Rebuilding cache (force or model changed: %s -> %s)", cached_tag, model_tag)
        cached = {}
        cached_files = []
        cached_emb = None

    # Prepare output buffers
    out_files: List[str] = []
    out_embeds: List[np.ndarray] = []
    out_sigs: Dict[str, str] = {}

    # If updating and cache exists, start with cached entries for files that still exist
    if cached and update_only:
        for fn in files:
            if fn in cached:
                idx = cached[fn]
                out_files.append(fn)
                out_embeds.append(np.array(cached_emb[idx], dtype=np.float32))
                out_sigs[fn] = cur_sigs.get(fn, "")

    # Compute missing
    missing = [fn for fn in files if fn not in set(out_files)]
    logging.info("Computing embeddings for %d new images (of %d total)", len(missing), len(files))
    for i, fn in enumerate(missing, 1):
        path = os.path.join(faces_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        emb = embedder(img)
        if emb is None:
            continue
        out_files.append(fn)
        out_embeds.append(emb.astype(np.float32))
        out_sigs[fn] = cur_sigs.get(fn, file_sig(path))
        if i % 50 == 0:
            logging.info("  … %d/%d new", i, len(missing))

    if not out_embeds:
        logging.warning("No embeddings produced; cache not written.")
        return

    mat = np.stack(out_embeds, axis=0)
    # Save raw (matcher will l2-normalize on load)
    save_cache(cache_npz, cache_json, out_files, mat, model_tag=model_tag, file_sigs=out_sigs)
    logging.info("Cache saved: %s (%d vectors, dim=%d, model=%s)", cache_npz, mat.shape[0], mat.shape[1], model_tag)


def main():
    ap = argparse.ArgumentParser(description="Warm or update ArcFace embeddings cache for a faces directory without importing matcher")
    ap.add_argument("--faces_db", required=True, help="Directory containing face images")
    ap.add_argument("--backend", default="skip", help="DeepFace detector backend (skip, retinaface, mtcnn, ssd, opencv, etc.)")
    ap.add_argument("--force", action="store_true", help="Rebuild the entire cache from scratch")
    ap.add_argument("--embed_backend", default="auto", choices=["auto","deepface","insightface"],
                    help="Embedding backend to use (auto prefers DeepFace)")
    ap.add_argument("--no_update", action="store_true", help="Disable incremental update; recompute all (unless --force)")
    ap.add_argument("--insight_det_size", type=str, default="640,640", help="InsightFace detector size WxH, e.g., 640,640 or 1024,1024")
    ap.add_argument("--insight_det_thresh", type=float, default=0.45, help="InsightFace detection threshold (lower -> more faces)")
    ap.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel), format="%(asctime)s %(levelname)s:%(message)s")

    # Parse det_size
    det_size = (640, 640)
    try:
        sz = str(getattr(args, "insight_det_size", "640,640"))
        parts = [int(x.strip()) for x in sz.replace("x", ",").split(",") if x.strip()]
        if len(parts) == 2 and parts[0] > 0 and parts[1] > 0:
            det_size = (parts[0], parts[1])
    except Exception:
        det_size = (640, 640)
    det_thresh = float(getattr(args, "insight_det_thresh", 0.45))

    build_or_update_cache(
        faces_dir=args.faces_db,
        backend=args.backend,
        update_only=not args.no_update,
        force=bool(args.force),
        embed_backend=str(getattr(args, "embed_backend", "auto")).lower(),
        insight_det_size=det_size,
        insight_det_thresh=det_thresh,
    )


if __name__ == "__main__":
    main()
