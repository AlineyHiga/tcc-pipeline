from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import settings, HealthCheck

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent

for path in (ROOT, WORKSPACE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Configure Hypothesis for property-based testing
settings.register_profile("default", 
    max_examples=200, 
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("default")

# Ensure reports directory exists
reports_dir = ROOT / "reports"
reports_dir.mkdir(exist_ok=True)
