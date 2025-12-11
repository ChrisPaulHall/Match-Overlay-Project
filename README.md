# Congress Face Match & Overlay Project

Real-time face recognition system that identifies U.S. Congress members on a live video feed and renders an overlay for OBS streaming.

> **[QUICKSTART.md](QUICKSTART.md)** for the fastest path to running the demo.

## What it does
- Captures video from OBS Virtual Camera
- Detects faces using InsightFace (ONNX) or DeepFace (TensorFlow)
- Matches against a database of 500+ Congress member faces (`01_data/faces_official`)
- Displays member info: name, title, net worth, committee assignments, recent stock trades
- Writes structured overlay data to `02_outputs/overlay_data.json`
- Serves a browser overlay at `http://localhost:5021/` for OBS browser source

## Prerequisites
- **Python 3.10 or 3.11** (required; 3.9 will fail because of dependency minimums; 3.13 has compatibility issues)
- **OBS Studio** with Browser Source capability
- **GPU** (optional but recommended) - macOS with CoreML support for GPU acceleration

## Setup

### 1. Download the Face Database
The face database (~1.6 GB) is hosted separately due to size:

**[Download faces_official.zip from Google Drive](https://drive.google.com/drive/folders/1VjMNSBHbMNhX-oLgdK1u1NF8ttAcBxpC?usp=drive_link)**

Extract to `01_data/`:
```bash
unzip faces_official.zip -d 01_data/
```
You should end up with `01_data/faces_official/` containing ~3,600 face images.

### 2. Install Dependencies

**Recommended:** use the staged install below to avoid long OpenCV builds. Direct `pip install -r requirements.txt` can work but may compile OpenCV from source (slow).

```bash
# From repo root
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
```

**Important:** Install in stages to avoid OpenCV compilation:
```bash
# Core dependencies first
pip install Flask beautifulsoup4 numpy requests pillow PyYAML xxhash pytesseract

# ML packages (headless OpenCV avoids GUI/build issues)
pip install opencv-python-headless tensorflow "onnx>=1.14,<1.17" onnxruntime \
    scikit-image quiverquant

# Face recognition (install deepface without deps to avoid opencv rebuild)
pip install --no-deps deepface
pip install insightface

# DeepFace sub-dependencies (install without rebuilding opencv-python)
pip install --no-deps fire gdown gunicorn mtcnn retina-face

# Verify all imports
python -c "import cv2, tensorflow, flask, insightface, deepface; print('All imports OK')"
```

### 3. Warm the Embedding Cache (first run only)
```bash
python core/warm_embeddings.py --faces_db 01_data/faces_official --embed_backend auto
```
This generates face embeddings for fast matching (~5-15 minutes). Use `--embed_backend insightface` or `--embed_backend deepface` to force a specific backend.

**Important:** If you specify a backend here, use the same `--embed_backend` flag when running `matcher.py` later. If the backends don't match, the matcher will recompute embeddings at runtime (slow).

### Key Dependencies
| Package | Version | Notes |
|---------|---------|-------|
| tensorflow | >=2.15 | ML backend for DeepFace |
| insightface | >=0.7 | Primary face detection/embedding |
| onnx | >=1.14,<1.17 | **Must be <1.17** (ml_dtypes compat) |
| opencv-python-headless | >=4.8 | Single OpenCV variant (no GUI) |
| deepface | >=0.0.90 | Face recognition framework |

### Apple Silicon (M1/M2/M3) Notes

- Use the arm64 wheels: `tensorflow-macos` + `tensorflow-metal` (instead of `tensorflow`), and `onnxruntime-silicon` (instead of `onnxruntime`).
- Keep `onnx<1.17` (ml_dtypes compat) and avoid `numpy>=2.3` when paired with OpenCV 4.12.x.
- If you see resolver warnings, explicitly pin the compatible trio used in testing:
  ```bash
  pip install "numpy==1.26.4" "opencv-python-headless==4.12.0.88"
  pip install "onnx==1.16.2" "onnxruntime-silicon==1.16.3"
  pip install tensorflow-macos tensorflow-metal
  pip install --no-deps deepface
  pip install insightface
  pip install --no-deps fire gdown gunicorn mtcnn retina-face
  ```
- Verify imports: `python -c "import cv2, tensorflow, flask, insightface, deepface; print('All imports OK')"`

**Quick install script (Apple Silicon):** For automated setup, use the provided script:
```bash
source venv/bin/activate
bash 03_scripts/install_m1.sh
```

## Running the system
You typically run two processes: the matcher (produces overlay JSON) and the web overlay server (renders it).

1) Start the matcher (camera index 0 by default):
```bash
source venv/bin/activate
python core/matcher.py --cam_index 0
```
Useful flags:
- `--embed_backend {auto,deepface,insightface}` (default `auto`) — **must match the backend used for `warm_embeddings.py`**
- `--insight_det_size 640,640` — detection resolution (480=fast, 640=balanced, 960=accurate)
- `--insight_det_thresh 0.5` — face detection confidence threshold
- `--fast_cache_only` to refuse computing embeddings if the cache is missing
- `--save_face_hits` (on by default) saves diverse face crops to `01_data/pending_review/`
- `--quiver_token <TOKEN>` to fetch live trades from Quiver (optional; otherwise uses cached data)

2) In another shell, start the overlay server:
```bash
source venv/bin/activate
python core/overlay_server_5021.py
```
Endpoints:
- Overlay (use in OBS browser window): `http://localhost:5021/` (default), `/1`, or `/2` for different layouts
- Data feed: `http://localhost:5021/data.json`
- Runtime config (GET/POST): `http://localhost:5021/config`
- Health check: `/health`

## Using with OBS
1. Add a **Browser Source** for your video (1920x1080):
   - Set URL to any video with Congress members (live stream, member social media, YouTube, etc.)
   - Examples: https://live.house.gov/, https://www.senate.gov/legislative/floor_activity_pail.htm
2. Add another **Browser Source** for the overlay (576x1080):
   - Set URL to `http://localhost:5021/` (try `/1` or `/2` for different layouts)
   - Refresh once `overlay_server_5021.py` is running
3. Start the OBS **Virtual Camera** (feeds matcher.py):
   - In OBS, go to **Controls → Start Virtual Camera**.
   - Set **Output Type** to “Source” and **Output Selection** to your video source (1920x1080).
   - Leave the virtual camera running; `matcher.py` reads from it (`--cam_index 0` by default—adjust if needed).

## Face Database Augmentation

The matcher automatically captures diverse face images for under-represented members:

**How it works:**
- Saves faces with confidence scores between 0.70-0.85 (diverse but confident)
- Only saves if the member has fewer than 10 images in the database
- Images go to `01_data/pending_review/` for human verification

**Review and import pending faces:**
```bash
python 03_scripts/import_pending_faces.py
```

This opens each image in Preview for review:
- `[a]` or `Enter` = Approve (move to `faces_official/`)
- `[r]` = Reject (delete)
- `[s]` = Skip (leave for later)
- `[q]` = Quit

Use `--auto` to import all without review, or `--dry-run` to preview.

**Tuning parameters:**
```bash
--save_face_min_confidence 0.70   # Floor (avoid false positives)
--save_face_max_confidence 0.85   # Ceiling (avoid duplicates)
--save_face_max_db_images 10      # Only save if member has fewer
--save_face_max_per_person 5      # Max pending images per person
```

Pending review saves are grouped by member (bioguide), so variants of the same person share the same per-person limits and session diversity checks.

After approving faces, regenerate embeddings:
```bash
python core/warm_embeddings.py --faces_db 01_data/faces_official
```

## GPU Acceleration (macOS)

On macOS, the matcher can use CoreML to offload face detection to the GPU, reducing CPU usage significantly.

**Check if GPU is available:**
```bash
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
```

**GPU is enabled by default.** Configure via the web UI at `http://localhost:5021/config`:

| Setting | Description | Default |
|---------|-------------|---------|
| **Use GPU** | Enable CoreML/GPU acceleration | On |
| **Detection Size** | Face detection resolution (320-960) | 640 |

Or via environment variables (requires restart):
```bash
export FACE_OVERLAY_USE_GPU=true
export FACE_OVERLAY_DET_SIZE=640
```

**Performance comparison:**

| Configuration | Typical CPU Usage |
|---------------|-------------------|
| CPU only, det_size=960 | ~300-400% |
| CPU only, det_size=640 | ~100-150% |
| GPU (CoreML), det_size=640 | ~30-50% |

**Note:** GPU/det_size changes require restarting the matcher to take effect.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **`ModuleNotFoundError: No module named 'cv2'` or `'flask'`** | Dependencies not installed. Activate venv then run the staged install commands (or `pip install -r requirements.txt` if you accept possible OpenCV builds) |
| **`AttributeError: cannot import name 'float4_e2m1fn' from 'ml_dtypes'`** | onnx version too high. Run: `pip install \"onnx>=1.14,<1.17\"` |
| **`ERROR: No matching distribution found for scikit-image==0.25.2`** | Using Python 3.9 (too old). Install Python 3.10 or 3.11 and recreate venv |
| **opencv-python building from source (takes forever)** | Cancel and install with staged approach above; install deepface with `--no-deps` |
| **No matches / ticker mode only** | Check InsightFace loads: `python -c \"from insightface.app import FaceAnalysis\"`. If error, fix onnx version |
| **Broken venv shebang** | Recreate env: `rm -rf venv && python3.11 -m venv venv` |
| **Tesseract not found** | OCR is optional. Install if needed: `brew install tesseract` (macOS) |
| **Empty overlay** | Check `02_outputs/overlay_data.json` and matcher logs; lower `--beam_min` or `--face_min` |
| **High CPU** | Enable GPU at `http://localhost:5021/config`; reduce detection size to 480; increase `face_interval` and `frame_interval` |
| **Python 3.13 errors** | Use Python 3.10/3.11 instead |
| **CoreML not available** | Check `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` |
| **`mutex lock failed` on Apple Silicon** | Set threading env vars before running: `OMP_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1 python ...` |

## Project Structure

```
face_overlay_proj/
├── core/                    # Main application code
│   ├── matcher.py           # Face recognition main loop
│   ├── overlay_server_5021.py  # Flask server for browser overlay
│   ├── faces.py             # Face embedding backends
│   ├── overlay.py           # Overlay data generation
│   └── warm_embeddings.py   # Pre-compute face embeddings
├── 01_data/                 # Data files
│   ├── faces_official/      # Face database (download separately)
│   ├── pending_review/      # Captured faces awaiting review
│   ├── *.csv                # Member data, trades, donors
│   └── *.yaml               # Congress/committee metadata
├── 02_outputs/              # Runtime outputs (overlay JSON, caches)
├── 03_scripts/              # Utility and scraper scripts
├── 04_reports/              # Validation reports
├── requirements.txt         # Python dependencies
├── QUICKSTART.md            # Fast setup guide
└── README.md                # This file
```

## Data layout
- `01_data/faces_official/` : reference face crops (**download from** https://drive.google.com/drive/folders/1VjMNSBHbMNhX-oLgdK1u1NF8ttAcBxpC?usp=drive_link)
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
- `03_scripts/import_pending_faces.py` — interactive review and import of captured faces
- `03_scripts/analyze_pending_faces.py` — calculate impact score for pending images
- `03_scripts/validate_member_images.py` — cross-check face DB for mislabeled images
- `03_scripts/scraper_OS.py` — scrape donor data from OpenSecrets (see Data Scrapers)
- `03_scripts/scraper_quiver.py` — fetch net worth and holdings from QuiverQuant (see Data Scrapers)
- `03_scripts/compare_donor_files.py` — validate and deploy scraped donor data
- `03_scripts/compare_holdings_files.py` — validate and deploy scraped holdings data
- `03_scripts/list_cameras.py` — list available cameras to find OBS virtual camera index
- `03_scripts/update_trades_snapshot.py` — append latest trades to local CSV (requires Quiver token)
- `03_scripts/analyze_member_database_strength.py` — analyze face DB quality and find weak members
- `03_scripts/fetch_company_logos.py` — download company logos for stock tickers
- `03_scripts/install_m1.sh` — automated Apple Silicon install script

## Data Scrapers

The overlay displays financial data for Congress members sourced from external APIs. Two scraper scripts populate the CSV files used by the matcher.

### Data Flow

| Scraper | Output File | Production File | Overlay Data |
|---------|-------------|-----------------|--------------|
| `scraper_OS.py` | `01_data/00members_donor_summary.csv` | `01_data/members_donor_summary.csv` | Top donors, top industries |
| `scraper_quiver.py` | `01_data/00members_holdings_and_sectors.csv` | `01_data/members_holdings_and_sectors.csv` | Net worth, holdings, traded sectors |

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
- Net worth estimates and top stock holdings (via JSON API)
- Top traded sectors (scraped from public pages)
- **Strategy returns vs SPY** — portfolio performance comparison (e.g., "Pelosi: +800%, SPY: +200% since 5/16/2014")

**API token optional:** Set `QUIVER_API_TOKEN` env var or create `quiver_token.txt` for authenticated requests (higher rate limits). Works without token but may be slower.

```bash
python 03_scripts/scraper_quiver.py --resume
```

Options:
- `--cache-dir .cache_quiver` — cache responses locally
- `--resume` — skip already-processed members
- `--test N` — process only first N members (for testing)
- `--bioguide ID [ID ...]` — process specific member(s) by bioguide ID
- `--delay 1.5` — seconds between requests (default 1.5)
- `-v, --verbose` — verbose logging

**Output fields:**
| Field | Description | Example |
|-------|-------------|---------|
| `net_worth_estimate` | Raw net worth value | `8200000.00` |
| `normalized_net_worth` | Formatted net worth | `$8,200,000` |
| `top_holdings` | Top 5 holdings by asset type | `Stocks: $5.2M; Real Estate: $2.1M` |
| `top_traded_sectors` | Top 5 traded sectors | `Technology: 45; Healthcare: 23` |
| `strategy_return` | Portfolio return percentage | `800.25%` |
| `spy_return` | SPY benchmark return | `200.15%` |
| `strategy_start_date` | First trade date | `2014-05-16` |

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

If no critical issues, the script prompts to deploy automatically:
- Backs up existing file as `*_OLD.csv`
- Renames new file to production name

Or deploy manually:
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
python 03_scripts/update_trades_snapshot.py                    # live endpoint (last ~1000 trades)
python 03_scripts/update_trades_snapshot.py --bulk             # full history (paginated)
python 03_scripts/update_trades_snapshot.py --v2               # V2 API format (more fields)
python 03_scripts/update_trades_snapshot.py --dry-run          # preview without writing
python 03_scripts/update_trades_snapshot.py --bioguide P000197 # filter by member
python 03_scripts/update_trades_snapshot.py --ticker AAPL      # filter by ticker
```

Options:
- `--bulk` — use bulk endpoint for full trade history (paginated)
- `--v2` — use V2 API format with extra fields (company, status, district, etc.)
- `--bioguide ID` — filter by BioGuide ID
- `--ticker SYMBOL` — filter by ticker symbol
- `--nonstock` — include non-stock transactions
- `--page-size N` — items per page for bulk endpoint (default: 10000)
- `--max-pages N` — max pages to fetch (0 = unlimited)
- `--dry-run` — preview without writing

### Analyze Face Database (`03_scripts/analyze_member_database_strength.py`)

Identify members with weak face representation (few images, poor quality):
```bash
python 03_scripts/analyze_member_database_strength.py --db 01_data/faces_official
```

### Analyze Pending Faces (`03_scripts/analyze_pending_faces.py`)

Calculate the impact of adding pending images to the face database:
```bash
python 03_scripts/analyze_pending_faces.py
python 03_scripts/analyze_pending_faces.py --show-all  # Include low-impact images
```

The script analyzes each pending image and scores it based on:
- **Diversity** — how different from existing images
- **Consistency** — confirms it's the same person
- **Coverage** — members with fewer images get higher scores
- **Quality** — face detection confidence

Output categories:
- 🟢 **HIGH** — recommend approve (adds diversity)
- 🟡 **MEDIUM** — consider (moderate value)
- 🔴 **SUSPICIOUS** — low similarity (might be wrong person)
- ⚪ **LOW** — similar to existing or member well-covered

Reports are saved to `04_reports/pending_analysis_TIMESTAMP.txt`.

### Validate Face Database (`03_scripts/validate_member_images.py`)

Cross-check all images for each member to detect mislabeled photos:
```bash
python 03_scripts/validate_member_images.py --db 01_data/faces_official
```

Options:
- `--threshold 0.4` — similarity threshold (0.4=lenient, 0.5=moderate, 0.6=strict)
- `--output-dir 04_reports` — where to save validation reports

Reports are saved to `04_reports/` with details on problematic members.

### Fetch Company Logos (`03_scripts/fetch_company_logos.py`)

Download company logos for stock tickers in the trades data:
```bash
python 03_scripts/fetch_company_logos.py              # fetch all missing logos
python 03_scripts/fetch_company_logos.py --test 50    # test with first 50 tickers
python 03_scripts/fetch_company_logos.py --force      # re-fetch even if cached
python 03_scripts/fetch_company_logos.py --dry-run    # preview without downloading
```

Output:
- `01_data/logos/{TICKER}.png` — Cached logo images
- `01_data/ticker_logos.csv` — Index mapping tickers to logo paths

Uses [Logo.dev](https://logo.dev) API. Set `LOGO_DEV_TOKEN` env var to use your own token.

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

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

For bug reports and feature requests, please open an issue on GitHub.

---
*Last updated: 2025-12-11* (Apple Silicon support, strategy returns, bulk trades API, fetch_company_logos)
