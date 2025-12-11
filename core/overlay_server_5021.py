from flask import Flask, render_template_string, jsonify, make_response, request, url_for
import contextlib
import importlib.util
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

try:
    from .settings import load_settings
except ImportError:  # pragma: no cover - allow script execution
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

SETTINGS = load_settings()
PATHS = SETTINGS.paths

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = Path(SCRIPT_DIR) / "static"
PROJECT_ROOT = str(PATHS.project_root)
OUTPUT_DIR   = str(PATHS.output_dir)
OVERLAY_JSON = str(PATHS.overlay_json_path)
CONFIG_JSON  = str(PATHS.runtime_config_path)
COMMANDS_JSON = str(Path(OUTPUT_DIR) / "matcher_commands.json")
PORT = SETTINGS.overlay_port

COMPANY_TICKERS_PATH = Path(OUTPUT_DIR) / "company_tickers.json"
_TICKER_NAME_CACHE: Dict[str, str] = {}
_TICKER_CACHE_MTIME: float = 0.0

# Runtime configuration that can be hot-reloaded
RUNTIME_CONFIG: Dict[str, Any] = SETTINGS.runtime_defaults_dict()

DEFAULT_PAYLOAD: Dict[str, Any] = {
    "name": "No speaker detected",
    "dob": "",
    "age": "",
    "tenure_pretty": "",
    "bio_card": {"items": []},
    "committees": [],
    "latest_trades": [],
    "networth_current": "",
    "normalized_net_worth": "",
    "top_holdings": [],
    "top_donors": [],
    "top_industries": [],
    "periods": {},
    "source_url": "",
    "quiver_url": ""
}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# Disable Flask's static file caching in development
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def _clean_company_title(title: str) -> str:
    """Normalize company titles for display."""
    if not title:
        return ""
    temp = re.sub(r"[.,]+", " ", title)
    temp = re.sub(r"\s+", " ", temp).strip()
    tokens = [tok for tok in temp.split(" ") if tok]
    if not tokens:
        return title.strip()

    suffixes = {
        "inc", "incorporated", "corp", "corporation", "company", "co",
        "holdings", "group", "plc", "llc", "ltd", "lp", "partners", "sa",
        "spa", "ag", "nv", "se", "ab", "pty", "limited", "inc.", "corp.",
        "co.", "l.p."
    }
    while tokens and tokens[-1].lower().rstrip(".") in suffixes:
        tokens.pop()
    if tokens and tokens[-1].lower().rstrip(".") == "com":
        tokens.pop()

    result = " ".join(tokens).strip() or title.strip()

    def _smart_title(word: str) -> str:
        if word.isupper() or word.isnumeric():
            return word
        if len(word) <= 3 and word.isalpha():
            return word.upper()
        return word.capitalize()

    pretty = " ".join(_smart_title(w) for w in result.split(" "))
    return pretty.strip()


def _load_company_tickers() -> Dict[str, str]:
    """Load and cache ticker -> company name mapping."""
    global _TICKER_CACHE_MTIME
    try:
        mtime = COMPANY_TICKERS_PATH.stat().st_mtime  # type: ignore[union-attr]
    except FileNotFoundError:
        _TICKER_NAME_CACHE.clear()
        _TICKER_CACHE_MTIME = 0.0
        return _TICKER_NAME_CACHE
    except (OSError, IOError) as exc:
        app.logger.warning(f"Failed to stat company tickers file: {exc}")
        return _TICKER_NAME_CACHE

    if _TICKER_NAME_CACHE and mtime == _TICKER_CACHE_MTIME:
        return _TICKER_NAME_CACHE

    try:
        with COMPANY_TICKERS_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (IOError, OSError, json.JSONDecodeError, ValueError) as exc:
        app.logger.warning(f"Failed to load company tickers: {exc}")
        return _TICKER_NAME_CACHE

    entries = raw.values() if isinstance(raw, dict) else raw
    mapping: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ticker = str(entry.get("ticker") or "").strip().upper()
        title = str(entry.get("title") or "").strip()
        if not ticker or not title:
            continue
        mapping[ticker] = _clean_company_title(title)

    _TICKER_NAME_CACHE.clear()
    _TICKER_NAME_CACHE.update(mapping)
    _TICKER_CACHE_MTIME = mtime
    return _TICKER_NAME_CACHE


@app.route("/company_tickers.json")
def company_tickers():
    """Expose the ticker -> company mapping for the front-end."""
    mapping = _load_company_tickers()
    resp = make_response(jsonify(mapping))
    # Weak ETag to help OBS browser source notice changes
    resp.headers["ETag"] = str(int(_TICKER_CACHE_MTIME or time.time()))
    return resp


def _enrich_trade_records(payload: Dict[str, Any]) -> None:
    """Attach company names to trade entries for display."""
    trades = payload.get("latest_trades")
    if not isinstance(trades, list) or not trades:
        return
    ticker_names = _load_company_tickers()
    if not ticker_names:
        return

    for trade in trades:
        if not isinstance(trade, dict):
            continue
        ticker = str(
            trade.get("ticker")
            or trade.get("Ticker")
            or trade.get("symbol")
            or ""
        ).strip().upper()
        if not ticker:
            continue
        company_name = ticker_names.get(ticker)
        if not company_name:
            continue

        existing_name = str(
            trade.get("asset")
            or trade.get("company")
            or trade.get("description")
            or ""
        ).strip()
        if not existing_name or existing_name.upper() == ticker:
            trade["asset"] = company_name

        if not str(trade.get("company_clean") or "").strip():
            trade["company_clean"] = company_name


def _load_runtime_config():
    """Load runtime configuration from file if it exists."""
    global RUNTIME_CONFIG
    if os.path.exists(CONFIG_JSON):
        try:
            with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                RUNTIME_CONFIG.update(loaded)
                app.logger.info(f"Loaded runtime config from {CONFIG_JSON}")
        except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
            app.logger.warning(f"Failed to load runtime config: {e}", exc_info=True)

def _save_runtime_config():
    """Save runtime configuration to file."""
    try:
        with open(CONFIG_JSON, "w", encoding="utf-8") as f:
            json.dump(RUNTIME_CONFIG, f, indent=2)
        app.logger.info(f"Saved runtime config to {CONFIG_JSON}")
    except (IOError, OSError, TypeError) as e:
        app.logger.error(f"Failed to save runtime config: {e}", exc_info=True)


def _load_command_payload() -> Dict[str, Any]:
    """Return commands file contents; always at least {'commands': []}."""
    if not os.path.exists(COMMANDS_JSON):
        return {"commands": []}
    try:
        with open(COMMANDS_JSON, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
            if isinstance(raw, dict):
                commands = raw.get("commands")
                if isinstance(commands, list):
                    return {"commands": commands}
    except (IOError, OSError, json.JSONDecodeError, ValueError) as exc:
        app.logger.warning(f"Failed to read commands file {COMMANDS_JSON}: {exc}")
    return {"commands": []}


def _persist_commands(commands: List[Dict[str, Any]]) -> None:
    """Atomically persist command list, keeping only the most recent entries."""
    payload = {"commands": commands[-50:]}
    tmp_path = COMMANDS_JSON + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, COMMANDS_JSON)
    except (IOError, OSError, TypeError) as exc:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp_path)
        app.logger.error(f"Failed to persist commands: {exc}", exc_info=True)


def _append_command(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Append a command entry and persist it."""
    data = _load_command_payload()
    commands = data.get("commands", [])
    now_ts = time.time()
    # Prune stale entries older than 10 minutes to avoid unbounded growth.
    pruned = [
        cmd for cmd in commands
        if isinstance(cmd, dict) and float(cmd.get("ts", now_ts)) >= now_ts - 600.0
    ]
    pruned.append(entry)
    _persist_commands(pruned)
    return entry


def _current_overlay_identity() -> Dict[str, Any]:
    """Best-effort snapshot of the currently displayed identity."""
    payload = DEFAULT_PAYLOAD
    if os.path.exists(OVERLAY_JSON):
        try:
            with open(OVERLAY_JSON, "r", encoding="utf-8") as fh:
                file_payload = json.load(fh)
            payload = _merge_defaults(file_payload)
        except (IOError, OSError, json.JSONDecodeError, ValueError) as exc:
            app.logger.warning(f"Failed to inspect overlay payload: {exc}")
    ident = {
        "bioguide": str(payload.get("bioguide_id") or "").strip(),
        "display_name": str(payload.get("name") or "").strip(),
        "image": str(payload.get("image") or "").strip(),
    }
    scores = payload.get("scores") or {}
    if isinstance(scores, dict):
        ident["scores"] = {
            "combined": float(scores.get("combined", 0.0)),
            "face": float(scores.get("face", 0.0)),
        }
    return ident


# Prime the runtime configuration so defaults or existing overrides are immediately available.
_load_runtime_config()


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />
  <link rel="stylesheet" href="{{ css_href }}">
</head>
<body>
  <div class="overlay overlay--detail" id="overlay">
    <div class="overlay-header">
      <div class="identity-block">
        <div class="identity-headline">
          <div class="identity-headline__body">
            <h2 id="speaker-name">Loading...</h2>
          </div>
        </div>
        <div id="networth-section" class="identity-networth">
          <div class="identity-networth__label">Estimated net worth</div>
          <div class="kpi identity-networth__value" id="networth">—</div>
        </div>
      </div>
    </div>

    <!-- Ticker mode container (shown when data.ticker is present) -->
    <div id="ticker-mode" class="section">
      <div class="section-eyebrow" id="ticker-title"></div>
      <div class="ticker-wrapper">
        <ul id="ticker-list" class="list flat"></ul>
      </div>
    </div>

    <div class="overlay-grid" id="card-container">
      <div class="column column--secondary">
        <div id="bio-section" class="section" data-has-content="false">
          <div class="section-eyebrow">Biographical Information</div>
          <ul id="bio-list" class="list"></ul>
        </div>

        <div id="committees-section" class="section" data-has-content="false">
          <div class="section-eyebrow">Committees &amp; Subcommittees</div>
          <ul id="committee-list" class="list"></ul>
        </div>
      </div>

      <div class="column column--primary">
        <div id="holdings-section" class="section" data-has-content="false">
          <div class="section-eyebrow">Top holdings</div>
          <ul id="holdings-list" class="list list--kv"></ul>
        </div>

        <div id="sectors-section" class="section" data-has-content="false">
          <div class="section-eyebrow">Top traded sectors</div>
          <ul id="sectors-list" class="list"></ul>
        </div>

        <div id="trades-section" class="section" data-has-content="false">
          <div class="section-eyebrow">Latest trades</div>
          <div id="trades-list" class="trade-list"></div>
        </div>

        <div id="donors-section" class="section" data-has-content="false">
          <div class="section-eyebrow" id="donors-eyebrow">Top Campaign Contributors</div>
          <div id="donors-period" class="subtle"></div>
          <ul id="donor-list" class="list list--kv"></ul>
        </div>

        <div id="industries-section" class="section" data-has-content="false">
          <div class="section-eyebrow" id="industries-eyebrow">Top Donor Industries</div>
          <div id="industries-period" class="subtle"></div>
          <ul id="industry-list" class="list list--kv"></ul>
        </div>
      </div>
    </div>
  </div>

  <script src="{{ overlay_js_url }}"></script>
</body>
</html>

"""

def _merge_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge: ensure all expected keys exist; prefer file values."""
    merged = DEFAULT_PAYLOAD.copy()
    for k, v in (payload or {}).items():
        merged[k] = v
    # Normalize some fields
    if not isinstance(merged.get("committees"), list):
        merged["committees"] = []
    if not isinstance(merged.get("latest_trades"), list):
        merged["latest_trades"] = []
    if not isinstance(merged.get("top_donors"), list):
        merged["top_donors"] = []
    if not isinstance(merged.get("top_industries"), list):
        merged["top_industries"] = []
    if not isinstance(merged.get("top_holdings"), list):
        merged["top_holdings"] = []
    if not isinstance(merged.get("periods"), dict):
        merged["periods"] = {}
    return merged

@app.after_request
def add_no_cache_headers(resp):
    # strong no-cache for both HTML and JSON
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

@app.route("/")
@app.route("/3")
def overlay():
    timestamp = int(time.time())
    overlay_js_url = url_for("static", filename="js/overlay.js", v=timestamp)
    css_href = url_for("static", filename="css/overlay3.css", v=timestamp)
    return render_template_string(
        HTML_TEMPLATE,
        overlay_js_url=overlay_js_url,
        css_href=css_href,
        timestamp=timestamp,
    )


@app.route("/1")
def overlay_1():
    timestamp = int(time.time())
    overlay_js_url = url_for("static", filename="js/overlay.js", v=timestamp)
    css_href = url_for("static", filename="css/overlay1.css", v=timestamp)
    return render_template_string(
        HTML_TEMPLATE,
        overlay_js_url=overlay_js_url,
        css_href=css_href,
        timestamp=timestamp,
    )

@app.route("/2")
def overlay_2():
    timestamp = int(time.time())
    overlay_js_url = url_for("static", filename="js/overlay.js", v=timestamp)
    css_href = url_for("static", filename="css/overlay2.css", v=timestamp)
    return render_template_string(
        HTML_TEMPLATE,
        overlay_js_url=overlay_js_url,
        css_href=css_href,
        timestamp=timestamp,
    )

@app.route("/data.json")
def data_json():
    try:
        if os.path.exists(OVERLAY_JSON):
            with open(OVERLAY_JSON, "r", encoding="utf-8") as f:
                file_data = json.load(f)
            data = _merge_defaults(file_data)
            app.logger.info("Serving overlay data for: %s", data.get("name", "No name"))
        else:
            app.logger.warning("overlay_data.json not found at %s; serving defaults.", OVERLAY_JSON)
            data = DEFAULT_PAYLOAD
    except (IOError, OSError, json.JSONDecodeError, ValueError) as e:
        app.logger.error(f"Error reading/parsing JSON from {OVERLAY_JSON}: {e}", exc_info=True)
        data = DEFAULT_PAYLOAD

    _enrich_trade_records(data)

    # Add weak ETag to help some browser sources detect change while no-store is set
    resp = make_response(jsonify(data))
    resp.headers["ETag"] = str(int(time.time()))
    return resp

@app.route("/health")
def health():
  """Basic health check endpoint."""
  status = {
    "status": "ok",
    "overlay_json": OVERLAY_JSON,
    "exists": bool(os.path.exists(OVERLAY_JSON)),
    "mtime": os.path.getmtime(OVERLAY_JSON) if os.path.exists(OVERLAY_JSON) else None,
  }
  return jsonify(status)

@app.route("/config")
def config_ui():
    """Configuration UI."""
    CONFIG_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Matcher Configuration</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #1a1a2e;
      color: #eee;
      padding: 20px;
      margin: 0;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      background: #16213e;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    h1 {
      color: #7ec8ff;
      margin-top: 0;
    }
    .control-group {
      margin: 25px 0;
      padding: 20px;
      background: #0f1624;
      border-radius: 8px;
      border-left: 3px solid #7ec8ff;
    }
    label {
      display: block;
      font-weight: 450;
      margin-bottom: 8px;
      color: #7ec8ff;
    }
    .description {
      font-size: 13px;
      color: #999;
      margin-bottom: 10px;
    }
    input[type="range"] {
      width: 100%;
      height: 8px;
      border-radius: 5px;
      background: #0a0e1a;
      outline: none;
      -webkit-appearance: none;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #7ec8ff;
      cursor: pointer;
    }
    input[type="range"]::-moz-range-thumb {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #7ec8ff;
      cursor: pointer;
      border: none;
    }
    .value-display {
      display: inline-block;
      min-width: 60px;
      text-align: right;
      font-weight: bold;
      color: #78ffd6;
      font-size: 16px;
    }
    .button-group {
      margin-top: 30px;
      display: flex;
      gap: 15px;
    }
    button {
      padding: 12px 24px;
      border: none;
      border-radius: 6px;
      font-size: 15px;
      font-weight: 450;
      cursor: pointer;
      transition: all 0.2s;
    }
    .btn-save {
      background: #7ec8ff;
      color: #0f1624;
    }
    .btn-save:hover {
      background: #6ab8ef;
      transform: translateY(-1px);
    }
    .btn-reset {
      background: #ff6b6b;
      color: white;
    }
    .btn-reset:hover {
      background: #ee5555;
    }
    .btn-reject {
      background: #ffb347;
      color: #0f1624;
    }
    .btn-reject:hover {
      background: #f89a2c;
      transform: translateY(-1px);
    }
    .status {
      margin-top: 20px;
      padding: 12px;
      border-radius: 6px;
      display: none;
    }
    .status.success {
      background: #2d7a4f;
      color: #78ffd6;
      display: block;
    }
    .status.error {
      background: #7a2d2d;
      color: #ff8b8b;
      display: block;
    }
    .links {
      margin-top: 30px;
      padding-top: 20px;
      border-top: 1px solid #0a0e1a;
    }
    a {
      color: #7ec8ff;
      text-decoration: none;
      margin-right: 20px;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎛️ Matcher Configuration</h1>
    
    <div class="control-group">
      <label>Manual Override</label>
      <div class="description">Click to tell the matcher the current overlay is wrong. It will suppress that identity briefly and surface the next-best match.</div>
      <button class="btn-reject" onclick="rejectCurrent()">⛔ Reject Current Match</button>
      <div class="status" id="reject-status"></div>
    </div>
    
    <div class="control-group">
      <label>Face Recognition Threshold <span class="value-display" id="val-face_threshold">0.35</span></label>
      <div class="description">Minimum similarity score for face matches (0.0-1.0)</div>
      <input type="range" id="face_threshold" min="0" max="1" step="0.01" value="0.35">
    </div>
    
    <div class="control-group">
      <label>Use GPU Acceleration</label>
      <div class="description">Enable CoreML/GPU for face detection (requires restart)</div>
      <input type="checkbox" id="use_gpu" checked>
      <span class="value-display" id="val-use_gpu">On</span>
    </div>
    
    <div class="control-group">
      <label>Beam Minimum Score <span class="value-display" id="val-beam_min">0.45</span></label>
      <div class="description">Minimum combined score to display overlay (0.0-1.0)</div>
      <input type="range" id="beam_min" min="0" max="1" step="0.01" value="0.45">
    </div>
    
    <div class="control-group">
      <label>Switch Margin <span class="value-display" id="val-switch_margin">0.05</span></label>
      <div class="description">Score advantage needed to switch identities (0.0-0.3)</div>
      <input type="range" id="switch_margin" min="0" max="0.3" step="0.01" value="0.05">
    </div>
    
    <div class="control-group">
      <label>Detection Size <span class="value-display" id="val-det_size">640</span></label>
      <div class="description">Face detection resolution (480=fast, 640=balanced, 960=accurate). Requires restart.</div>
      <input type="range" id="det_size" min="320" max="960" step="160" value="640">
    </div>
    
    <div class="control-group">
      <label>Cooldown Frames <span class="value-display" id="val-cooldown_frames">30</span></label>
      <div class="description">Number of frames before allowing identity switch (5-120)</div>
      <input type="range" id="cooldown_frames" min="5" max="120" step="1" value="30">
    </div>
    
    <div class="control-group">
      <label>Face Detection Interval <span class="value-display" id="val-face_interval">2</span></label>
      <div class="description">Process face detection every N frames (1-10)</div>
      <input type="range" id="face_interval" min="1" max="10" step="1" value="2">
    </div>
    
    <div class="control-group">
      <label>Frame Processing Interval <span class="value-display" id="val-frame_interval">1</span></label>
      <div class="description">Process every N frames from camera (1-60)</div>
      <input type="range" id="frame_interval" min="1" max="60" step="1" value="1">
    </div>
    
    <div class="button-group">
      <button class="btn-save" onclick="saveConfig()">💾 Save Configuration</button>
      <button class="btn-reset" onclick="resetDefaults()">🔄 Reset to Defaults</button>
    </div>
    
    <div class="status" id="status"></div>
    
    <div class="links">
      <a href="/">← Back to Overlay</a>
      <a href="/health">Health Check</a>
      <a href="/config.json">View Config JSON</a>
    </div>
  </div>
  
  <script>
    const sliders = ['face_threshold', 'beam_min', 'switch_margin',
                     'cooldown_frames', 'face_interval', 'frame_interval', 'det_size'];
    
    // Load current config on page load
    async function loadConfig() {
      try {
        const response = await fetch('/config.json');
        const config = await response.json();

        sliders.forEach(key => {
          const slider = document.getElementById(key);
          const display = document.getElementById('val-' + key);
          if (config[key] !== undefined) {
            slider.value = config[key];
            display.textContent = config[key];
          }
        });

        // Handle GPU checkbox
        const gpuCheckbox = document.getElementById('use_gpu');
        const gpuDisplay = document.getElementById('val-use_gpu');
        if (config.use_gpu !== undefined) {
          gpuCheckbox.checked = config.use_gpu;
          gpuDisplay.textContent = config.use_gpu ? 'On' : 'Off';
        }
      } catch (e) {
        console.error('Failed to load config:', e);
      }
    }
    
    // Update value displays when sliders change
    sliders.forEach(key => {
      const slider = document.getElementById(key);
      const display = document.getElementById('val-' + key);

      slider.addEventListener('input', (e) => {
        display.textContent = e.target.value;
      });
    });

    // Handle GPU checkbox display
    document.getElementById('use_gpu').addEventListener('change', (e) => {
      document.getElementById('val-use_gpu').textContent = e.target.checked ? 'On' : 'Off';
    });
    
    async function rejectCurrent() {
      const status = document.getElementById('reject-status');
      if (status) {
        status.style.display = 'block';
        status.className = 'status';
        status.textContent = 'Submitting reject request...';
      }
      try {
        const response = await fetch('/commands/reject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
        const result = await response.json();
        if (status) {
          const command = result && result.command ? result.command : {};
          const label = command.display_name || command.bioguide || 'current match';
          if (response.ok) {
            status.className = 'status success';
            status.textContent = `✅ Reject queued for ${label}`;
          } else {
            status.className = 'status error';
            status.textContent = `❌ Reject failed: ${result.error || 'Unknown error'}`;
          }
        }
      } catch (e) {
        if (status) {
          status.className = 'status error';
          status.textContent = `❌ Network error: ${e.message}`;
        }
      } finally {
        if (status) {
          setTimeout(() => { status.style.display = 'none'; }, 5000);
        }
      }
    }
    
    async function saveConfig() {
      const config = {};
      sliders.forEach(key => {
        const value = parseFloat(document.getElementById(key).value);
        config[key] = value;
      });
      // Add GPU checkbox
      config.use_gpu = document.getElementById('use_gpu').checked;

      try {
        const response = await fetch('/config/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        const status = document.getElementById('status');
        if (response.ok) {
          status.className = 'status success';
          status.textContent = '✅ Configuration saved successfully! Changes will apply to new frames.';
        } else {
          status.className = 'status error';
          status.textContent = '❌ Failed to save: ' + (result.error || 'Unknown error');
        }
        
        setTimeout(() => {
          status.style.display = 'none';
        }, 5000);
      } catch (e) {
        const status = document.getElementById('status');
        status.className = 'status error';
        status.textContent = '❌ Network error: ' + e.message;
      }
    }
    
    function resetDefaults() {
      if (!confirm('Reset all values to defaults?')) return;
      
      const defaults = {{ defaults | tojson }};
      
      sliders.forEach(key => {
        const slider = document.getElementById(key);
        const display = document.getElementById('val-' + key);
        slider.value = defaults[key];
        display.textContent = defaults[key];
      });
    }
    
    // Load config on startup
    loadConfig();
  </script>
</body>
</html>
    """
    return render_template_string(CONFIG_HTML, defaults=SETTINGS.runtime_defaults_dict())

@app.route("/config.json")
def config_json():
    """Return current configuration as JSON."""
    return jsonify(RUNTIME_CONFIG)


def _sanitize_identity_string(s: str) -> str:
    """Remove dangerous characters from identity strings.
    
    Args:
        s: Input string to sanitize
        
    Returns:
        Sanitized string with dangerous characters removed, max 256 chars
    """
    if not isinstance(s, str):
        return ""
    # Remove null bytes, path separators, and special characters
    cleaned = re.sub(r'[\x00\\\/:*?"<>|]', '', s)
    return cleaned.strip()[:256]  # Limit length


@app.route("/commands/reject", methods=["POST"])
def reject_command():
    """Queue a reject command so the matcher skips the current identity."""
    body = request.get_json(silent=True) or {}
    ttl_frames = body.get("ttl_frames")
    ttl_seconds = body.get("ttl_seconds")
    ttl_default = 90  # ~3 seconds at 30fps

    ttl_calc = None
    if isinstance(ttl_frames, (int, float)):
        ttl_calc = int(ttl_frames)
    elif isinstance(ttl_seconds, (int, float)):
        ttl_calc = int(float(ttl_seconds) * 30.0)

    ttl_final = max(15, min(900, ttl_calc if ttl_calc is not None else ttl_default))

    # Sanitize input strings to prevent injection attacks
    target_bioguide = _sanitize_identity_string(body.get("bioguide", ""))
    target_name = _sanitize_identity_string(body.get("display_name", ""))
    target_filename = _sanitize_identity_string(body.get("filename", ""))

    if not (target_bioguide or target_name or target_filename):
        current = _current_overlay_identity()
        target_bioguide = _sanitize_identity_string(current.get("bioguide", ""))
        target_name = _sanitize_identity_string(current.get("display_name", ""))
        target_filename = _sanitize_identity_string(current.get("image", ""))
        if not (target_bioguide or target_name or target_filename):
            return jsonify({"error": "No active match to reject."}), 400

    command = {
        "id": str(uuid.uuid4()),
        "type": "reject",
        "bioguide": target_bioguide,
        "display_name": target_name,
        "filename": target_filename,
        "note": _sanitize_identity_string(body.get("note", "")),
        "ttl_frames": ttl_final,
        "ts": time.time(),
        "source_ip": request.remote_addr or "",
    }
    _append_command(command)
    app.logger.info(
        "Queued reject command %s for %s (%s)",
        command["id"],
        target_bioguide or target_name or target_filename,
        command["filename"],
    )
    return jsonify({"status": "queued", "command": command}), 200

def _validate_config_value(key: str, value: Any) -> bool:
    """Validate configuration values with type and range checking."""
    validation_rules = {
        "face_threshold": (float, 0.0, 1.0),
        "beam_min": (float, 0.0, 1.0),
        "switch_margin": (float, 0.0, 0.3),
        "cooldown_frames": (int, 5, 120),
        "face_interval": (int, 1, 30),
        "frame_interval": (int, 1, 200),
        "det_size": (int, 320, 960),
    }

    # Boolean settings
    if key == "use_gpu":
        return isinstance(value, bool)

    if key not in validation_rules:
        return False

    expected_type, min_val, max_val = validation_rules[key]

    # Type check
    if not isinstance(value, (expected_type, int if expected_type == float else float)):
        return False

    # Range check
    if not (min_val <= value <= max_val):
        return False

    return True

@app.route("/config/update", methods=["POST"])
def update_config():
    """Update configuration from POST request."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate all keys and values
        errors = []
        for key, value in data.items():
            if key not in RUNTIME_CONFIG:
                errors.append(f"Unknown configuration key: {key}")
            elif not _validate_config_value(key, value):
                errors.append(f"Invalid value for {key}: {value}")
        
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        # Update config if validation passed
        for key, value in data.items():
            RUNTIME_CONFIG[key] = value
        
        # Save to disk so changes persist
        _save_runtime_config()
        
        app.logger.info(f"Config updated: {data}")
        return jsonify({"status": "success", "config": RUNTIME_CONFIG}), 200
    
    except (ValueError, TypeError, KeyError) as e:
        app.logger.error(f"Failed to update config: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
  # Load runtime configuration at startup
  _load_runtime_config()

  app.logger.info("Overlay server running at http://localhost:%s/", PORT)
  app.logger.info("Reading overlay JSON from: %s", OVERLAY_JSON)
  app.logger.info("Main overlay: http://localhost:%s/", PORT)
  app.logger.info("Configuration: http://localhost:%s/config", PORT)
  app.run(host="0.0.0.0", port=PORT)
