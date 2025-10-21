from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import settings

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent

for path in (ROOT, WORKSPACE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

settings.register_profile("ci", max_examples=200, deadline=None)
settings.load_profile("ci")
