"""Utilities for loading and encoding images for the GLM vision API."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


_MAX_SIDE = 2048  # GLM-4V recommended maximum dimension


def load_and_encode(path: Path, max_side: int = _MAX_SIDE) -> str:
    """Load an image file, optionally downscale it, and return a base64 string."""
    with Image.open(path) as img:
        img = _maybe_resize(img.convert("RGB"), max_side)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


def encode_pil(img: Image.Image, max_side: int = _MAX_SIDE) -> str:
    """Encode a PIL Image to base64 JPEG."""
    img = _maybe_resize(img.convert("RGB"), max_side)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _maybe_resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)
