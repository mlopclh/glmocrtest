"""Tests for glm_ocr.models."""

import pytest
from pathlib import Path

from glm_ocr.models import (
    BoundingBox,
    OCRRequest,
    OCRResult,
    OutputFormat,
    TextRegion,
)


class TestOCRRequest:
    def test_text_format_default(self):
        req = OCRRequest(image_url="https://example.com/img.png")
        assert req.output_format == OutputFormat.TEXT

    def test_has_image_url(self):
        req = OCRRequest(image_url="https://example.com/img.png")
        assert req.has_image()

    def test_has_image_base64(self):
        req = OCRRequest(image_base64="abc123")
        assert req.has_image()

    def test_has_image_false_when_empty(self):
        req = OCRRequest()
        assert not req.has_image()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            OCRRequest(image_path=tmp_path / "nonexistent.png")

    def test_existing_file_accepted(self, tmp_path: Path):
        img = tmp_path / "test.png"
        img.write_bytes(b"fake")
        req = OCRRequest(image_path=img)
        assert req.image_path == img


class TestOCRResult:
    def test_text_property(self):
        result = OCRResult(raw_text="Hello world", model="glm-4v")
        assert result.text == "Hello world"

    def test_regions_default_empty(self):
        result = OCRResult(raw_text="x", model="glm-4v")
        assert result.regions == []

    def test_with_regions(self):
        region = TextRegion(text="foo", confidence=0.99)
        result = OCRResult(raw_text="foo", regions=[region], model="glm-4v")
        assert len(result.regions) == 1
        assert result.regions[0].text == "foo"


class TestBoundingBox:
    def test_fields(self):
        bb = BoundingBox(x=0.1, y=0.2, width=0.3, height=0.4)
        assert bb.x == pytest.approx(0.1)
        assert bb.height == pytest.approx(0.4)
