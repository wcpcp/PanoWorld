from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is importable when running scripts.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from erp_meta.config import load_config


def load_cfg(path: str) -> Dict[str, Any]:
    return load_config(path)


def root_dir() -> Path:
    return _ROOT
