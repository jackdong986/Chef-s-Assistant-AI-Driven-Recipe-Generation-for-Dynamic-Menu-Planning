"""Compatibility entry point for the single-model Gradio application."""

from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "gradio_app.py"), run_name="__main__")
