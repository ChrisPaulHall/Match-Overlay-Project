# Quick Start

This file shows the minimal steps to get the project running.

## Prerequisites
- Python 3.10 or 3.11
- (Optional) `brew install tesseract` for OCR

## From repo root

### 1) Download the face DB (separate, ~1.6 GB)

```bash
# Download from Google Drive and unzip into 01_data/
# https://drive.google.com/drive/folders/1zMTi7956xKNUfFbspdEaMTFKekDscwv9?usp=drive_link
unzip faces_official.zip -d 01_data/
# verify you have: 01_data/faces_official/
```

### 2) Create and activate a venv

```bash
# From repo root (macOS/Linux, zsh)
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
```

### 3) Install dependencies (staged to avoid building OpenCV)

```bash
# Core dependencies
pip install Flask==3.0.3 flask-cors==5.0.0 beautifulsoup4==4.12.3 numpy==1.26.4 \
    requests==2.32.3 pandas==2.2.2 pillow==10.4.0 PyYAML==6.0.2 xxhash==3.4.1 \
    pytesseract==0.3.13

# ML packages (use headless OpenCV to avoid build)
pip install opencv-python-headless==4.10.0.84 tensorflow==2.15.0 \
    tensorflow-estimator==2.15.0 "onnx>=1.14,<1.17" onnxruntime==1.18.1 \
    scikit-image==0.25.2 quiverquant==0.2.2

# Face recognition and insightface
pip install --no-deps deepface==0.0.93
pip install insightface==0.7.3

# Install the small missing deepface deps without triggering OpenCV rebuild
pip install --no-deps fire gdown gunicorn mtcnn retina-face
```

### Quick verification

```bash
# Should print the success message (some harmless warnings are OK)
python -c "import cv2, tensorflow, flask, insightface, deepface; print('All imports successful')"
```

### 4) Warm the embedding cache (first run only)

```bash
# This will generate embeddings for the faces DB
python core/warm_embeddings.py --faces_db 01_data/faces_official --embed_backend auto
# If you prefer explicit backends use --embed_backend insightface or deepface
```

**Important:** The backend you choose here is locked into the cache. When running matcher.py later, use the same `--embed_backend` flag. If they don't match, embeddings will be recomputed at runtime (slow).

### 5) Start the overlay server (new shell)

```bash
source venv/bin/activate
python core/overlay_server_5021.py
# Server runs at http://localhost:5021/  (routes: /, /data.json, /dashboard)
```

### 6) Start the matcher (new shell)

```bash
source venv/bin/activate
# default cam_index=0; adjust as needed
# IMPORTANT: use the same --embed_backend as step 4 (default: auto)
python core/matcher.py --cam_index 0
# matcher writes overlay JSON to 02_outputs/overlay_data.json
# OCR is disabled by default; matcher runs face-only unless you enable OCR deps
```

### 7) Add the overlay to OBS

**Video Source (1920x1080):**
- Add a **Browser Source** in OBS
- Set URL to any video with Congress members (live stream, member social media, YouTube, etc.):
  - Examples: https://live.house.gov/, https://www.senate.gov/legislative/floor_activity_pail.htm

**Overlay Source (576x1080):**
- Add another **Browser Source** in OBS
- Set URL to `http://localhost:5021/` (try `/1` or `/2` for different layouts)
- Refresh the source once `overlay_server_5021.py` is running

**Virtual Camera (feeds matcher.py):**
- In OBS, go to **Controls → Start Virtual Camera**.
- Make sure the **Output Type** is set to “Source” and **Output Selection** is your video source (1920x1080).
- Leave the virtual camera running; `matcher.py` will read from it (`--cam_index 0` by default—adjust if needed).
