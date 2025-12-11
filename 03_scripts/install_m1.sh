#!/usr/bin/env bash
set -euo pipefail

# Apple Silicon (M1/M2/M3) install helper for this project.
# Run from repo root with an active Python 3.10/3.11 virtualenv.
# Example:
#   python3.11 -m venv venv
#   source venv/bin/activate
#   bash 03_scripts/install_m1.sh

echo "==> Installing pinned Apple Silicon wheels"

echo "[1/6] Pin numpy and OpenCV (arm64)"
pip install "numpy==1.26.4" "opencv-python-headless==4.12.0.88"

echo "[2/6] Install ONNX and ONNX Runtime (silicon build, keep onnx<1.17)"
pip install "onnx==1.16.2" "onnxruntime-silicon==1.16.3"

echo "[3/6] TensorFlow arm64 + Metal acceleration"
pip install tensorflow-macos tensorflow-metal

echo "[4/6] Core deps"
pip install Flask beautifulsoup4 requests pillow PyYAML xxhash pytesseract quiverquant scikit-image

echo "[5/6] Face pipelines"
pip install --no-deps deepface
pip install insightface
pip install --no-deps fire gdown gunicorn mtcnn retina-face

echo "[6/6] Verify critical imports"
# Set threading env vars to avoid mutex crash when tensorflow + insightface are imported together
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1

python - <<'PY'
import sys
print("Python:", sys.version)

errors = []
def check(name, fn):
    try:
        fn()
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        print(f"✗ {name}: {exc}")
    else:
        print(f"✓ {name}: OK")

check("cv2", lambda: __import__("cv2"))
check("tensorflow", lambda: __import__("tensorflow"))
check("insightface", lambda: __import__("insightface"))
check("deepface", lambda: __import__("deepface"))
check("onnxruntime", lambda: __import__("onnxruntime"))
check("flask", lambda: __import__("flask"))

if errors:
    print("\nOne or more imports failed. See errors above.")
    sys.exit(1)
else:
    print("\nAll critical imports succeeded.")
PY

echo "==> Done. If you need embeddings, run:"
echo "    python core/warm_embeddings.py --faces_db 01_data/faces_official --embed_backend auto"
