"""OCR client backed by a local Ollama instance running glm-ocr."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from glm_ocr.image_utils import load_and_encode
from glm_ocr.models import OCRRequest, OCRResult, OutputFormat, TextRegion
from glm_ocr.prompt import build_user_prompt, get_system_prompt

_DEFAULT_MODEL = "glm-ocr:latest"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaGLMOCRClient:
    """OCR client that sends requests to a local Ollama instance.

    GLM-OCR is best served via Ollama's native ``/api/generate`` endpoint
    because the OpenAI-compatible chat endpoint has limited vision support for
    this model family.

    Parameters
    ----------
    model:
        Ollama model tag.  Defaults to ``glm-ocr:latest``.
    base_url:
        Base URL of the Ollama server.  Defaults to ``http://localhost:11434``.
    timeout:
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ocr(self, request: OCRRequest) -> OCRResult:
        """Run OCR on the image described by *request*."""
        if not request.has_image():
            raise ValueError("OCRRequest must specify an image source.")

        b64_image = self._resolve_image(request)
        prompt = f"{get_system_prompt()}\n\n{build_user_prompt(request)}"

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
        }

        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()

        data = response.json()
        raw_text: str = data.get("response", "")
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }

        regions = self._parse_regions(raw_text, request.output_format)

        return OCRResult(
            raw_text=raw_text,
            regions=regions,
            model=self.model,
            usage=usage,
            metadata={"backend": "ollama", "base_url": self._base_url},
        )

    def ocr_file(self, path: Path | str, **kwargs: Any) -> OCRResult:
        """Convenience wrapper: run OCR on a local image file."""
        return self.ocr(OCRRequest(image_path=Path(path), **kwargs))

    def ocr_url(self, url: str, **kwargs: Any) -> OCRResult:
        """Convenience wrapper: run OCR on a remote image URL."""
        return self.ocr(OCRRequest(image_url=url, **kwargs))

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable and the model is loaded."""
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            return self.model in models
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_image(self, request: OCRRequest) -> str:
        """Return a base64-encoded JPEG string for the image in *request*."""
        if request.image_base64:
            return request.image_base64
        if request.image_path:
            return load_and_encode(request.image_path)
        if request.image_url:
            # Fetch the remote image and encode it locally.
            resp = httpx.get(request.image_url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            import base64
            import io
            from PIL import Image
            from glm_ocr.image_utils import encode_pil
            img = Image.open(io.BytesIO(resp.content))
            return encode_pil(img)
        return ""

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
