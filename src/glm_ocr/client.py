"""High-level GLM OCR client wrapping the ZhipuAI SDK."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from glm_ocr.image_utils import load_and_encode
from glm_ocr.models import OCRRequest, OCRResult, OutputFormat, TextRegion
from glm_ocr.prompt import build_user_prompt, get_system_prompt

try:
    from zhipuai import ZhipuAI  # type: ignore[import-untyped]
except ImportError:
    ZhipuAI = None  # type: ignore[assignment,misc]

_DEFAULT_MODEL = "glm-4v-plus"


class GLMOCRClient:
    """OCR client backed by GLM-4V vision models via the ZhipuAI API.

    Parameters
    ----------
    api_key:
        ZhipuAI API key.  Falls back to the ``ZHIPUAI_API_KEY`` environment
        variable when not supplied.
    model:
        Model name to use.  Defaults to ``glm-4v-plus``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A ZhipuAI API key is required. "
                "Pass `api_key=` or set the ZHIPUAI_API_KEY environment variable."
            )

        if ZhipuAI is None:
            raise ImportError("zhipuai is required: pip install zhipuai")

        self._client = ZhipuAI(api_key=resolved_key)
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ocr(self, request: OCRRequest) -> OCRResult:
        """Run OCR on the image described by *request*."""
        if not request.has_image():
            raise ValueError("OCRRequest must specify an image source.")

        content = self._build_content(request)
        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(request)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    *content,
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        raw_text: str = response.choices[0].message.content or ""
        usage: dict[str, Any] = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        regions = self._parse_regions(raw_text, request.output_format)

        return OCRResult(
            raw_text=raw_text,
            regions=regions,
            model=self.model,
            usage=usage,
        )

    def ocr_file(self, path: Path | str, **kwargs: Any) -> OCRResult:
        """Convenience wrapper: run OCR on a local image file."""
        return self.ocr(OCRRequest(image_path=Path(path), **kwargs))

    def ocr_url(self, url: str, **kwargs: Any) -> OCRResult:
        """Convenience wrapper: run OCR on a remote image URL."""
        return self.ocr(OCRRequest(image_url=url, **kwargs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_content(self, request: OCRRequest) -> list[dict[str, Any]]:
        if request.image_base64:
            return [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{request.image_base64}"
                    },
                }
            ]
        if request.image_path:
            b64 = load_and_encode(request.image_path)
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            ]
        if request.image_url:
            return [{"type": "image_url", "image_url": {"url": request.image_url}}]
        return []

    @staticmethod
    def _parse_regions(
        raw_text: str, output_format: OutputFormat
    ) -> list[TextRegion]:
        if output_format != OutputFormat.JSON:
            return []
        try:
            data = json.loads(raw_text)
            regions_data = data.get("regions", [])
            return [TextRegion(**r) for r in regions_data]
        except (json.JSONDecodeError, TypeError, KeyError):
            return []
