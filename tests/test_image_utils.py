"""Tests for glm_ocr.image_utils."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from glm_ocr.image_utils import _maybe_resize, encode_pil, load_and_encode


def _make_png_bytes(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMaybeResize:
    def test_no_resize_when_small(self):
        img = Image.new("RGB", (100, 100))
        result = _maybe_resize(img, max_side=200)
        assert result.size == (100, 100)

    def test_resize_when_large(self):
        img = Image.new("RGB", (4000, 2000))
        result = _maybe_resize(img, max_side=2048)
        assert max(result.size) == 2048
        # aspect ratio preserved
        assert result.size[0] == 2048
        assert result.size[1] == 1024

    def test_resize_portrait(self):
        img = Image.new("RGB", (1000, 4000))
        result = _maybe_resize(img, max_side=2000)
        assert result.size[1] == 2000
        assert result.size[0] == 500


class TestEncodePil:
    def test_returns_valid_base64(self):
        img = Image.new("RGB", (10, 10))
        b64 = encode_pil(img)
        decoded = base64.b64decode(b64)
        # JPEG magic bytes
        assert decoded[:2] == b"\xff\xd8"

    def test_large_image_downscaled(self):
        img = Image.new("RGB", (5000, 5000))
        b64 = encode_pil(img, max_side=100)
        decoded = base64.b64decode(b64)
        reopened = Image.open(io.BytesIO(decoded))
        assert max(reopened.size) <= 100


class TestLoadAndEncode:
    def test_loads_png(self, tmp_path):
        png = tmp_path / "test.png"
        png.write_bytes(_make_png_bytes(50, 50))
        b64 = load_and_encode(png)
        decoded = base64.b64decode(b64)
        assert decoded[:2] == b"\xff\xd8"  # JPEG

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_and_encode(tmp_path / "missing.png")
