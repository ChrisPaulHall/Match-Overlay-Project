# Face Overlay Project

Real-time face recognition system that identifies U.S. Congress members on a live video feed and renders an overlay for OBS streaming.

## What it does
- Captures video from OBS Virtual Camera
- Detects faces using InsightFace (ONNX) or DeepFace (TensorFlow)
- Matches against a database of 500+ Congress member faces (`01_data/faces_official`)
- Displays member info: name, title, net worth, committee assignments, recent stock trades
- Writes structured overlay data to `02_outputs/overlay_data.json`
- Serves a browser overlay at `http://localhost:5021/` for OBS

## Prerequisites
- **Python 3.10 or 3.11** (required; 3.9 will fail because of dependency minimums; 3.13 has compatibility issues)
- **Tesseract OCR** (optional, only for text/ticker OCR)
  - Install with `brew install tesseract` (macOS) or `apt-get install tesseract-ocr` (Linux)
  - Face recognition works without it; OCR features require the binary on PATH
- **OBS Studio** with Browser Source capability
- macOS/Linux recommended (OCR timeouts rely on `SIGALRM`)

## Setup

### 1. Download the Face Database
The face database (~1.6 GB) is hosted separately due to size:

**[Download faces_official.zip from Google Drive](https://drive.google.com/drive/folders/1zMTi7956xKNUfFbspdEaMTFKekDscwv9?usp=drive_link)**

Extract to `01_data/`:
```bash
unzip faces_official.zip -d 01_data/
```
You should end up with `01_data/faces_official/` containing ~3,600 face images.

### 2. Install Dependencies

**Do not use `pip install -r requirements.txt`** — it skips the critical staging steps needed to avoid long OpenCV builds. Follow the staged approach below instead.

```bash
# From repo root
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
```

**Important:** Install in stages to avoid OpenCV compilation:
```bash
# Core dependencies first
pip install Flask==3.0.3 flask-cors==5.0.0 beautifulsoup4==4.12.3 numpy==1.26.4 \
    requests==2.32.3 pandas==2.2.2 pillow==10.4.0 PyYAML==6.0.2 xxhash==3.4.1 \
    pytesseract==0.3.13

# ML packages
pip install opencv-python-headless==4.10.0.84 tensorflow==2.15.0 \
    tensorflow-estimator==2.15.0 "onnx>=1.14,<1.17" onnxruntime==1.18.1 \
    scikit-image==0.25.2 quiverquant==0.2.2

# Face recognition (install deepface without deps to avoid opencv rebuild)
pip install --no-deps deepface==0.0.93
pip install insightface==0.7.3

# Missing deepface dependencies (install without rebuilding opencv-python)
pip install --no-deps fire gdown gunicorn mtcnn retina-face

# Verify all imports
python -c "import cv2, tensorflow, flask, insightface, deepface; print('✓ All imports successful')"
```

### 3. Warm the Embedding Cache (first run only)
```bash
python core/warm_embeddings.py --faces_db 01_data/faces_official --embed_backend auto
```
This generates face embeddings for fast matching (~2-5 minutes). Use `--embed_backend insightface` or `--embed_backend deepface` to force a specific backend.

**Important:** If you specify a backend here, use the same `--embed_backend` flag when running `matcher.py` later. If the backends don't match, the matcher will recompute embeddings at runtime (slow).

### Key Dependencies
| Package | Version | Notes |
|---------|---------|-------|
| tensorflow | 2.15.0 | ML backend for DeepFace |
| insightface | 0.7.3 | Primary face detection/embedding |
| onnx | >=1.14,<1.17 | **Must be <1.17** (ml_dtypes compat) |
| opencv-python-headless | 4.10.x | Single OpenCV variant (no GUI) |
| deepface | 0.0.93 | Face recognition framework |

## Running the system
You typically run two processes: the matcher (produces overlay JSON) and the web overlay server (renders it).

1) Start the matcher (camera index 0 by default):
```bash
source venv/bin/activate
python core/matcher.py --cam_index 0
```
Useful flags:
- `--embed_backend {auto,deepface,insightface}` (default `auto`) — **must match the backend used for `warm_embeddings.py`**
- `--ocr_engine {auto,tesseract}` (Paddle/EasyOCR are disabled unless installed)
- `--fast_cache_only` to refuse computing embeddings if the cache is missing
- `--save_face_hits` (on by default) saves diverse face crops to `02_outputs/pending_review/`
- `--quiver_token <TOKEN>` to fetch live trades from Quiver (optional; otherwise uses cached data)

2) In another shell, start the overlay server:
```bash
source venv/bin/activate
python core/overlay_server_5021.py
```
Endpoints:
- Overlay (use in OBS): `http://localhost:5021/` (default), `/1`, or `/2` for different layouts
- Data feed: `http://localhost:5021/data.json`
- Runtime config (GET/POST): `http://localhost:5021/config`
- Dashboard views: `/dashboard`, `/dashboard/compact`
- Health check: `/health`

## Using with OBS
1. Add a **Browser Source** for your video (1920x1080):
   - Set URL to any video with Congress members (live stream, member social media, YouTube, etc.)
   - Examples: https://live.house.gov/, https://www.senate.gov/legislative/floor_activity_pail.htm
2. Add another **Browser Source** for the overlay (576x1080):
   - Set URL to `http://localhost:5021/` (try `/1` or `/2` for different layouts)
   - Enable transparency and refresh once `overlay_server_5021.py` is running

## Face Database Augmentation

The matcher automatically captures diverse face images for under-represented members:

**How it works:**
- Saves faces with confidence scores between 0.70-0.85 (diverse but confident)
- Only saves if the member has fewer than 10 images in the database
- Images go to `02_outputs/pending_review/` for human verification

**Review pending faces:**
```bash
python 03_scripts/review_pending_faces.py
```

Commands during review:
- `[a]` Approve - move to faces database
- `[r]` Reject - delete the image
- `[s]` Skip - leave for later
- `[q]` Quit

**Tuning parameters:**
```bash
--save_face_min_confidence 0.70   # Floor (avoid false positives)
--save_face_max_confidence 0.85   # Ceiling (avoid duplicates)
--save_face_max_db_images 10      # Only save if member has fewer
--save_face_max_per_person 5      # Max pending images per person
```

After approving faces, regenerate embeddings:
```bash
python core/warm_embeddings.py --faces_db 01_data/faces_official
```

## Quick Start

For a faster overview of the setup and running process, see [QUICKSTART.md](QUICKSTART.md).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **`ModuleNotFoundError: No module named 'cv2'` or `'flask'`** | Dependencies not installed. Activate venv then `pip install -r requirements.txt` |
| **`AttributeError: cannot import name 'float4_e2m1fn' from 'ml_dtypes'`** | onnx version too high. Run: `pip install \"onnx>=1.14,<1.17\"` |
| **`ERROR: No matching distribution found for scikit-image==0.25.2`** | Using Python 3.9 (too old). Install Python 3.10 or 3.11 and recreate venv |
| **opencv-python building from source (takes forever)** | Cancel and install with staged approach above; install deepface with `--no-deps` |
| **No matches / ticker mode only** | Check InsightFace loads: `python -c \"from insightface.app import FaceAnalysis\"`. If error, fix onnx version |
| **Broken venv shebang** | Recreate env: `rm -rf venv && python3.11 -m venv venv` |
| **Tesseract not found** | OCR is optional. Install if needed: `brew install tesseract` (macOS) |
| **Empty overlay** | Check `02_outputs/overlay_data.json` and matcher logs; lower `--beam_min` or `--face_min` |
| **High CPU** | Increase `--frame_interval` and `--face_interval`; use `--embed_backend insightface` |
| **Python 3.13 errors** | Use Python 3.10/3.11 instead |

## Data layout
- `01_data/faces_official/` : reference face crops (**download from** https://drive.google.com/drive/folders/1zMTi7956xKNUfFbspdEaMTFKekDscwv9?usp=drive_link)
- `01_data/*.csv`, `01_data/*.yaml` : member data (included in repo)
- `02_outputs/` : runtime outputs (overlay JSON, runtime config, tickers cache)
- Environment overrides (optional):
  - `FACE_OVERLAY_DATA_DIR` or `FACE_DATA_DIR` to point to a custom `01_data`
  - `OVERLAY_OUTPUT_DIR` or `FACE_OVERLAY_OUTPUT_DIR` to move `02_outputs`
  - `FACE_OVERLAY_PORT` or `OVERLAY_PORT` to change the Flask port (default 5021)

## Key files
- `requirements.txt` — runtime dependencies
- `core/matcher.py` — main loop: camera capture, face/OCR matching, overlay write
- `core/overlay_server_5021.py` — Flask server for the browser overlay
- `core/ocr.py` — OCR pipeline and ROI extraction
- `core/faces.py` — embedding backends (InsightFace/DeepFace) and face DB handling
- `03_scripts/review_pending_faces.py` — review and approve captured face crops
- `03_scripts/scraper_OS.py` — scrape donor data from OpenSecrets (see Data Scrapers)
- `03_scripts/scraper_quiver.py` — fetch net worth and holdings from QuiverQuant (see Data Scrapers)
- `03_scripts/compare_donor_files.py` — validate scraped donor data before deployment
- `03_scripts/compare_holdings_files.py` — validate scraped holdings data before deployment
- `03_scripts/list_cameras.py` — list available cameras to find OBS virtual camera index
- `03_scripts/update_trades_snapshot.py` — append latest trades to local CSV (requires Quiver token)
- `03_scripts/analyze_member_database_strength.py` — analyze face DB quality and find weak members

## Data Scrapers

The overlay displays financial data for Congress members sourced from external APIs. Two scraper scripts populate the CSV files used by the matcher.

### Data Flow

| Scraper | Output File | Production File | Overlay Data |
|---------|-------------|-----------------|--------------|
| `scraper_OS.py` | `00members_donor_summary.csv` | `members_donor_summary.csv` | Top donors, top industries |
| `scraper_quiver.py` | `00members_holdings_and_sectors.csv` | `members_holdings_and_sectors.csv` | Net worth, holdings, traded sectors |

The scrapers write to `00*` prefixed files. After validating with the comparison scripts, rename to production files.

### OpenSecrets Scraper (`03_scripts/scraper_OS.py`)

Scrapes campaign finance data directly from [OpenSecrets.org](https://www.opensecrets.org/) public pages (no API key required):
- Top Contributors (organizations donating to the member)
- Top Industries (industry sectors contributing)
- Contribution time periods

```bash
python 03_scripts/scraper_OS.py --input 01_data/members_donor_summary.csv --output 01_data/00members_donor_summary.csv
```

Options:
- `--delay 1.5` — seconds between requests (default 1.5, be respectful)
- `--top-n 5` — number of contributors/industries to keep
- `--resume` — skip already-processed rows
- `--cache-dir .cache_pages` — cache HTML locally to avoid repeated requests

### QuiverQuant Scraper (`03_scripts/scraper_quiver.py`)

Fetches financial disclosure data from [QuiverQuant](https://www.quiverquant.com/):
- Net worth estimates and top stock holdings (via authenticated API)
- Top traded sectors (scraped from public pages)

**Requires API token:** Set `QUIVER_API_TOKEN` env var or create `quiver_token.txt` in the project root.

```bash
python 03_scripts/scraper_quiver.py --resume
```

Options:
- `--cache-dir .cache_quiver` — cache responses locally
- `--resume` — skip already-processed members
- `--test N` — process only first N members (for testing)

### Validating Scraped Data

Before deploying scraped data to production, compare the new files against existing data:

```bash
# Compare donor data (OpenSecrets)
python 03_scripts/compare_donor_files.py

# Compare holdings data (QuiverQuant)
python 03_scripts/compare_holdings_files.py
```

These scripts check for:
- Missing members (data loss)
- Empty fields that previously had data
- Significant value changes
- Period/date mismatches

If no critical issues, deploy to production:
```bash
mv 01_data/00members_donor_summary.csv 01_data/members_donor_summary.csv
mv 01_data/00members_holdings_and_sectors.csv 01_data/members_holdings_and_sectors.csv
```

## Utility Scripts

### List Cameras (`03_scripts/list_cameras.py`)

Find the correct camera index for OBS Virtual Camera:
```bash
python 03_scripts/list_cameras.py
```
Use the index shown for `--cam_index` when running the matcher.

### Update Trades Snapshot (`03_scripts/update_trades_snapshot.py`)

Append latest congressional trades to the local CSV (requires Quiver token):
```bash
python 03_scripts/update_trades_snapshot.py
python 03_scripts/update_trades_snapshot.py --dry-run  # preview without writing
```

### Analyze Face Database (`03_scripts/analyze_member_database_strength.py`)

Identify members with weak face representation (few images, poor quality):
```bash
python 03_scripts/analyze_member_database_strength.py --db 01_data/faces_official
```

## Data Sources & Attribution

This project aggregates publicly available data from multiple sources:

| Source | Data Used | Website |
|--------|-----------|---------|
| **OpenSecrets** | Campaign contributions, top donors, industry funding | [opensecrets.org](https://www.opensecrets.org/) |
| **QuiverQuant** | Net worth estimates, stock holdings, traded sectors | [quiverquant.com](https://www.quiverquant.com/) |
| **Congress.gov** | Member biographies, committee assignments | [congress.gov](https://www.congress.gov/) |

**OpenSecrets** is a project of the Center for Responsive Politics, a nonpartisan, nonprofit research organization tracking money in U.S. politics. Data is sourced from FEC filings and other public disclosures.

**QuiverQuant** provides financial data on congressional stock trading and net worth based on periodic financial disclosure reports required by the STOCK Act.

Please review each provider's terms of service before running the scrapers. The cached CSV files in `01_data/` are provided for convenience; regenerate them with the scraper scripts for the latest data.

## Notes
- Default port is 5021; override with `OVERLAY_PORT` or `FACE_OVERLAY_PORT`
- Project favors InsightFace+ONNXRuntime for speed; DeepFace+TensorFlow is still supported via `--embed_backend deepface`

## License
This project's source code is licensed under the [MIT License](LICENSE) - free for personal and commercial use.

**Data Attribution:** The financial and biographical data displayed by this application is sourced from third-party providers (OpenSecrets, QuiverQuant, Congress.gov). Please review their respective terms of service for data usage requirements. See [Data Sources & Attribution](#data-sources--attribution) for details.

---
*Last updated: 2025-12-10*
