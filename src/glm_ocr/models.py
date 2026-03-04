"""Pydantic models for GLM OCR requests and results."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


class OCRRequest(BaseModel):
    """Input specification for an OCR job."""

    image_path: Path | None = None
    image_url: str | None = None
    image_base64: str | None = None
    language_hint: str | None = Field(
        default=None,
        description="BCP-47 language code hint, e.g. 'zh', 'en', 'ja'.",
    )
    output_format: OutputFormat = OutputFormat.TEXT
    extra_instructions: str | None = Field(
        default=None,
        description="Additional freeform instructions passed to the model.",
    )

    @field_validator("image_path")
    @classmethod
    def path_must_exist(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Image file not found: {v}")
        return v

    def has_image(self) -> bool:
        return any(
            x is not None
            for x in (self.image_path, self.image_url, self.image_base64)
        )


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalised [0, 1] coordinates."""

    x: float
    y: float
    width: float
    height: float


class TextRegion(BaseModel):
    """A detected text region within the image."""

    text: str
    confidence: float | None = None
    bounding_box: BoundingBox | None = None
    language: str | None = None


class OCRResult(BaseModel):
    """Result returned by the OCR pipeline."""

    raw_text: str
    regions: list[TextRegion] = Field(default_factory=list)
    model: str
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return the full extracted text."""
        return self.raw_text
