"""Compatibility shim exposing Quiver helpers from 03_scripts/quiver_api.py."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _import_scripts_module():
    """Import the shared 03_scripts/quiver_api module.

    The helpers were originally implemented under 03_scripts/, but the matcher
    runs from core.  This shim ensures relative imports keep working no
    matter which entry point executes the code.
    """

    project_root = Path(__file__).resolve().parents[1]
    scripts_module_path = project_root / "03_scripts" / "quiver_api.py"
    if not scripts_module_path.is_file():  # pragma: no cover - defensive diagnostic
        raise ImportError(
            "Unable to locate 03_scripts/quiver_api.py. "
            "Run commands from the repository root?"
        )

    module_name = "scripts_quiver_api_bridge"
    if module_name in sys.modules:  # reuse if previously loaded
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, scripts_module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive diagnostic
        raise ImportError("Failed to load 03_scripts/quiver_api.py via importlib")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_MODULE = _import_scripts_module()

QuiverQuantError = _MODULE.QuiverQuantError
fetch_congress_trading = _MODULE.fetch_congress_trading
load_token = getattr(_MODULE, "load_token", None)

__all__ = ["QuiverQuantError", "fetch_congress_trading", "load_token"]
