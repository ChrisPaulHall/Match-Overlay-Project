from __future__ import annotations

import logging
import os
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import pytesseract

try:  # pragma: no cover
    from .config import MatcherConfig
except ImportError:  # pragma: no cover
    from config import MatcherConfig  # type: ignore


# Timeout context manager for OCR operations -------------------------------
@contextmanager
def timeout_context(seconds: int):
    """Context manager to timeout operations using SIGALRM (Unix only)."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"OCR operation timed out after {seconds}s")

    # Set alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore handler


# Optional OCR backends -----------------------------------------------------
PADDLE_OK = False
_EASY_OCR_OK = False
_PADDLE = None
_EASYOCR_READER = None
PaddleOCR = None  # type: ignore
easyocr = None  # type: ignore

if str(os.environ.get("DISABLE_PADDLE", "")).lower() not in {"1", "true", "yes"}:
    try:
        from paddleocr import PaddleOCR  # type: ignore
        PADDLE_OK = True
    except Exception:  # pragma: no cover
        PADDLE_OK = False

if str(os.environ.get("DISABLE_EASYOCR", "")).lower() not in {"1", "true", "yes"}:
    try:
        import easyocr  # type: ignore
        _EASY_OCR_OK = True
    except Exception:  # pragma: no cover
        _EASY_OCR_OK = False


@dataclass
class OCRResult:
    text: str
    thresholded: Optional[np.ndarray]
    roi: Optional[np.ndarray]


class OCRProcessor:
    """Encapsulates ROI extraction and OCR engine orchestration."""

    def __init__(self, config: MatcherConfig, score_fn: Callable[[str], float]) -> None:
        self.config = config
        self.score_fn = score_fn
        self._ensure_tesseract_cmd()

    def process_frame(self, frame: np.ndarray) -> OCRResult:
        thr, roi = self._extract_roi(frame)
        text = self._run_ocr(thr)
        return OCRResult(text=text or "", thresholded=thr, roi=roi)

    # ------------------------------------------------------------------
    def _extract_roi(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame.shape[:2]
        y0_frac = float(self.config.get("ocr_y0", 0.60))
        h_frac = float(self.config.get("ocr_height", 0.40))
        x0_frac = float(self.config.get("ocr_x0", 0.30))
        w_frac = float(self.config.get("ocr_width", 0.75))

        y0 = int(h * max(0.0, min(1.0, y0_frac)))
        y1 = int(min(h, y0 + h * max(0.05, min(1.0, h_frac))))
        x0 = int(w * max(0.0, min(1.0, x0_frac)))
        x1 = int(min(w, x0 + w * max(0.05, min(1.0, w_frac))))

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clahe_clip = float(self.config.get("ocr_clahe_clip", 2.0))
        clahe_tiles = int(max(1, self.config.get("ocr_clahe_tiles", 8)))
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tiles, clahe_tiles))
        gray = clahe.apply(gray)

        if self.config.get("ocr_bilateral", False):
            gray = cv2.bilateralFilter(gray, 5, 25, 25)
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        gray = self._deskew(gray)

        if self.config.get("ocr_sharpen", False):
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
            gray = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

        thr = self._threshold(gray)

        kernel = np.ones((2, 2), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.config.get("ocr_dilate", False):
            thr = cv2.dilate(thr, kernel, iterations=1)

        if self.config.get("ocr_auto_invert", False):
            white = float((thr == 255).sum())
            black = float(thr.size) - white
            if white < black:
                thr = cv2.bitwise_not(thr)
        elif self.config.get("ocr_invert", False):
            thr = cv2.bitwise_not(thr)

        scale = max(1.0, float(self.config.get("ocr_scale", 3.0)))
        thr = cv2.resize(thr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return thr, roi

    def _threshold(self, gray: np.ndarray) -> np.ndarray:
        method = str(self.config.get("ocr_method", "adaptive"))
        if method == "adaptive":
            block_size = int(self.config.get("ocr_block_size", 31) or 31)
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(3, block_size)
            c = int(self.config.get("ocr_block_c", 9) or 9)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, block_size, c)
        if method == "otsu":
            g2 = cv2.GaussianBlur(gray, (3, 3), 0)
            return cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return self._binarize(gray)

    def _run_ocr(self, thr: np.ndarray) -> str:
        engine = str(self.config.get("ocr_engine", "tesseract")).lower()
        candidates: list[str] = []
        if engine == "auto":
            candidates.extend(["tesseract", "paddle" if PADDLE_OK else "easyocr" if _EASY_OCR_OK else "tesseract"])
        else:
            candidates.append(engine)

        seen = set()
        engines: list[str] = []
        for name in candidates:
            if name in seen:
                continue
            seen.add(name)
            engines.append(name)
            if len(engines) >= 2:
                break

        best_text = ""
        best_score = -1.0

        for name in engines:
            if name == "tesseract":
                text = self._ocr_tesseract(thr)
            elif name == "paddle":
                text = self._ocr_paddle(thr)
            elif name == "easyocr":
                text = self._ocr_easyocr(thr)
            else:
                logging.debug("Unknown OCR engine '%s'", name)
                text = ""
            text = (text or "").strip()
            if not text:
                continue
            score = self.score_fn(text)
            if score > best_score or (score == best_score and len(text) > len(best_text)):
                best_text, best_score = text, score

        return best_text

    def _ocr_tesseract(self, thr: np.ndarray) -> str:
        texts = []
        psm_list = [int(self.config.get("ocr_psm", 11))]
        for p in (6, 7, 11, 13):
            if p not in psm_list:
                psm_list.append(p)

        inv_variants = [thr]
        if self.config.get("ocr_auto_invert", False):
            inv_variants.append(cv2.bitwise_not(thr))

        whitelist = (self.config.get("ocr_whitelist", "") or "").strip()
        lang = str(self.config.get("ocr_lang", "eng"))

        for psm in psm_list:
            cfg = f"--psm {int(psm)}"
            if whitelist:
                cfg += f" -c tessedit_char_whitelist={whitelist}"
            for vimg in inv_variants:
                try:
                    # Add timeout to prevent hanging (5 seconds per OCR attempt)
                    txt = pytesseract.image_to_string(vimg, lang=lang, config=cfg, timeout=5)
                    if self.config.get("ocr_letters_only", False):
                        txt = self._letters_only(txt)
                    if txt:
                        texts.append(txt)
                except Exception as exc:  # pragma: no cover
                    logging.debug("Tesseract failed or timeout (psm=%s): %s", psm, exc)

        return max(texts, key=self.score_fn, default="")

    def _ocr_paddle(self, img: np.ndarray) -> str:
        if not self._ensure_paddle():
            return ""
        try:
            with timeout_context(5):  # 5 second timeout
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = _PADDLE.ocr(rgb, cls=True) or []
        except TimeoutError:
            logging.warning("PaddleOCR timeout after 5 seconds")
            return ""
        except Exception as exc:  # pragma: no cover
            logging.debug("PaddleOCR failed: %s", exc)
            return ""

        words = []
        for page in res:
            for item in (page or []):
                try:
                    (_box, (txt, conf)) = item
                    if conf >= 0.60 and txt and any(ch.isalpha() for ch in txt):
                        words.append(txt)
                except Exception:
                    continue
        return " ".join(words).strip()

    def _ocr_easyocr(self, img: np.ndarray) -> str:
        if not self._ensure_easyocr():
            return ""
        try:
            with timeout_context(5):  # 5 second timeout
                src = img
                if len(src.shape) == 3 and src.shape[2] == 3:
                    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                results = _EASYOCR_READER.readtext(src)
        except TimeoutError:
            logging.warning("EasyOCR timeout after 5 seconds")
            return ""
        except Exception as exc:  # pragma: no cover
            logging.debug("EasyOCR failed: %s", exc)
            return ""

        texts = []
        for item in results:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    text = item[1]
                elif isinstance(item, dict):
                    text = item.get("text", "")
                else:
                    text = ""
                text = (text or "").strip()
                if text:
                    texts.append(text)
            except Exception:
                continue
        joined = " ".join(texts).strip()
        return self._letters_only(joined) if self.config.get("ocr_letters_only", False) else joined

    # ------------------------------------------------------------------
    def _ensure_tesseract_cmd(self) -> None:
        try:
            if pytesseract.pytesseract.tesseract_cmd and os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                return
        except Exception:
            pass
        import shutil
        import platform

        candidates = []
        if platform.system() == "Darwin":
            candidates += ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
        candidates += ["tesseract"]
        for candidate in candidates:
            exe = shutil.which(candidate)
            if exe:
                pytesseract.pytesseract.tesseract_cmd = exe
                logging.info("Tesseract at: %s", exe)
                return
        logging.warning("Tesseract not found in PATH; OCR may fail.")

    def _ensure_easyocr(self) -> bool:
        global _EASYOCR_READER
        if not _EASY_OCR_OK:
            return False
        if _EASYOCR_READER is not None:
            return True
        try:
            lang = str(self.config.get("ocr_lang", "eng"))
            lang = "en" if lang.lower().startswith("eng") else lang
            _EASYOCR_READER = easyocr.Reader([lang], gpu=False, verbose=False)
            return True
        except Exception as exc:  # pragma: no cover
            logging.debug("EasyOCR init failed: %s", exc)
            return False

    def _ensure_paddle(self) -> bool:
        global _PADDLE
        if not PADDLE_OK:
            return False
        if _PADDLE is not None:
            return True
        try:
            _PADDLE = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            return True
        except Exception as exc:  # pragma: no cover
            logging.debug("PaddleOCR init failed: %s", exc)
            return False

    @staticmethod
    def _letters_only(text: str) -> str:
        import re
        cleaned = re.sub(r"[^A-Za-z'\-\s]+", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        try:
            g = cv2.GaussianBlur(gray, (3, 3), 0)
            thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            thr = cv2.bitwise_not(thr)
            coords = np.column_stack(np.where(thr > 0))
            if coords.size == 0:
                return gray
            pts = coords[:, ::-1].astype(np.float32)
            angle = cv2.minAreaRect(pts)[-1]
            if angle < -45:
                angle = 90 + angle
            h, w = gray.shape[:2]
            matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
            return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return gray

    def _binarize(self, gray: np.ndarray) -> np.ndarray:
        if self.config.get("ocr_use_sauvola", False):
            try:
                from skimage.filters import threshold_sauvola  # type: ignore

                base = max(15, min(51, int(min(gray.shape[:2]) / 24)))
                win = base if base % 2 == 1 else base + 1
                T = threshold_sauvola(gray, window_size=win, k=0.2)
                return (gray > T).astype(np.uint8) * 255
            except Exception:
                pass
        g2 = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
