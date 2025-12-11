# Quick Start

Quick guide to getting the program running in ~15 minutes. For detailed setup, see [README.md](README.md).

## Prerequisites

- **Python 3.10 or 3.11** (3.9 too old, 3.13 not yet supported)
- **OBS Studio** with Virtual Camera capability
- **GPU** (optional) — macOS CoreML support for lower CPU usage


## Step 1: Download the Face Database (~1.6 GB)

Download from Google Drive and extract to `01_data/`:

**[Download faces_official.zip](https://drive.google.com/drive/folders/1VjMNSBHbMNhX-oLgdK1u1NF8ttAcBxpC?usp=drive_link)**

```bash
unzip faces_official.zip -d 01_data/
ls 01_data/faces_official/  # Should show ~3,600 face images
```

## Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
```

## Step 3: Install Dependencies (Staged)

Install in stages to avoid OpenCV building from source:

```bash
# Core dependencies
pip install Flask beautifulsoup4 numpy requests pillow PyYAML xxhash pytesseract

# ML packages (headless OpenCV avoids GUI/build issues)
pip install opencv-python-headless tensorflow "onnx>=1.14,<1.17" onnxruntime \
    scikit-image quiverquant

# Face recognition (install deepface without deps to avoid opencv conflicts)
pip install --no-deps deepface
pip install insightface

# DeepFace sub-dependencies
pip install --no-deps fire gdown gunicorn mtcnn retina-face
```

**Verify installation:**
```bash
python -c "import cv2, tensorflow, flask, insightface, deepface; print('All imports OK')"
```

### Apple Silicon (M1/M2/M3) Install

Prefer the arm64 wheels to avoid x86 fallbacks:

```bash
# Core dependencies
pip install Flask beautifulsoup4 requests pillow PyYAML xxhash scikit-image quiverquant

# Pinned arm64 wheels
pip install "numpy==1.26.4" "opencv-python-headless==4.12.0.88"
pip install "onnx==1.16.2" "onnxruntime-silicon==1.16.3"
pip install tensorflow-macos tensorflow-metal

# Face recognition
pip install --no-deps deepface
pip install insightface
pip install --no-deps fire gdown gunicorn mtcnn retina-face
```

Verify:
```bash
python -c "import cv2, tensorflow, flask, insightface, deepface; print('All imports OK')"
```

**Or use the automated script** (from repo root, venv activated):
```bash
bash 03_scripts/install_m1.sh
```

## Step 4: Generate Face Embeddings (First Run Only)

```bash
python core/warm_embeddings.py --faces_db 01_data/faces_official --embed_backend auto
```

This can take 5-15 minutes. The `--embed_backend` you choose here must match what you use with matcher.py.

## Step 5: Start the System (Two Terminals)

**Terminal 1 - Overlay Server:**
```bash
source venv/bin/activate
python core/overlay_server_5021.py
```
Server runs at http://localhost:5021/

**Terminal 2 - Face Matcher:**
```bash
source venv/bin/activate
python core/matcher.py --cam_index 0
```
Adjust `--cam_index` if OBS Virtual Camera isn't on index 0.

## Step 6: Configure OBS

1. **Add Video Source** (Browser Source, 1920x1080):
   - URL: Any Congress video stream (e.g., https://live.house.gov/)

2. **Add Overlay Source** (Browser Source, 576x1080):
   - URL: `http://localhost:5021/` (try `/1` or `/2` for layouts)

3. **Start Virtual Camera**:
   - Controls → Start Virtual Camera
   - Output Type: "Source" → your video source

The matcher reads from the virtual camera and writes overlay data; the browser source displays it.

## Step 7: Configure (Optional)

Open `http://localhost:5021/config` in your browser to tune settings:

| Setting | Description | Default |
|---------|-------------|---------|
| **Use GPU** | Enable CoreML acceleration (macOS) | On |
| **Detection Size** | Face detection resolution | 640 |
| **Face Threshold** | Minimum similarity for matches | 0.50 |
| **Face Interval** | Detect faces every N frames | 5 |

**Note:** GPU and Detection Size changes require restarting the matcher.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Activate venv: `source venv/bin/activate` |
| ml_dtypes error | `pip install "onnx>=1.14,<1.17"` |
| OpenCV building from source | Use staged install above |
| No faces detected | Check `python 03_scripts/list_cameras.py` for correct cam index |
| High CPU usage | Enable GPU and reduce Detection Size at `http://localhost:5021/config` |
| `mutex lock failed` (Apple Silicon) | Set env vars: `OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 python ...` |

See [README.md](README.md) for full documentation.
