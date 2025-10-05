from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import settings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

settings.register_profile("ci", max_examples=200, deadline=None)
settings.load_profile("ci")
