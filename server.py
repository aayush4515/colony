"""
Root entrypoint for Render/Vercel: sets up paths and exposes the dashboard FastAPI app.
The app is defined in frontend/dashboard/server.py and uses backend/protein_swarm.
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "backend"))
sys.path.insert(0, str(_root / "frontend"))

from dashboard.server import app

__all__ = ["app"]
