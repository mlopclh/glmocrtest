"""Tests for glm_ocr.ollama_client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import httpx

from glm_ocr.models import OCRRequest, OCRResult, OutputFormat
from glm_ocr.ollama_client import OllamaGLMOCRClient


def _mock_generate_response(text: str, prompt_tokens: int = 10, eval_tokens: int = 5):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "response": text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
    }
    return mock_resp


class TestOllamaGLMOCRClientInit:
    def test_defaults(self):
        client = OllamaGLMOCRClient()
        assert client.model == "glm-ocr:latest"
        assert "11434" in client._base_url

    def test_custom_params(self):
        client = OllamaGLMOCRClient(model="glm-ocr:q4", base_url="http://host:1234")
        assert client.model == "glm-ocr:q4"
        assert client._base_url == "http://host:1234"

    def test_trailing_slash_stripped(self):
        client = OllamaGLMOCRClient(base_url="http://localhost:11434/")
        assert not client._base_url.endswith("/")


class TestIsAvailable:
    def test_returns_true_when_model_present(self):
        client = OllamaGLMOCRClient()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "glm-ocr:latest"}]}
        with patch("httpx.get", return_value=mock_resp):
            assert client.is_available() is True

    def test_returns_false_when_model_absent(self):
        client = OllamaGLMOCRClient()
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3:latest"}]}
        with patch("httpx.get", return_value=mock_resp):
            assert client.is_available() is False

    def test_returns_false_on_connection_error(self):
        client = OllamaGLMOCRClient()
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            assert client.is_available() is False


def _patch_resolve(b64: str = "fakeb64"):
    """Patch _resolve_image so URL-based tests don't need a real HTTP fetch."""
    return patch.object(OllamaGLMOCRClient, "_resolve_image", return_value=b64)


class TestOCR:
    def test_ocr_url(self):
        client = OllamaGLMOCRClient()
        mock_resp = _mock_generate_response("Hello OCR")
        with _patch_resolve(), patch("httpx.post", return_value=mock_resp):
            result = client.ocr(OCRRequest(image_url="https://x.com/a.png"))
        assert isinstance(result, OCRResult)
        assert result.text == "Hello OCR"
        assert result.metadata["backend"] == "ollama"

    def test_ocr_base64(self):
        client = OllamaGLMOCRClient()
        mock_resp = _mock_generate_response("b64 result")
        with patch("httpx.post", return_value=mock_resp):
            result = client.ocr(OCRRequest(image_base64="abc123"))
        assert result.text == "b64 result"

    def test_ocr_file(self, tmp_path):
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"fake")
        client = OllamaGLMOCRClient()
        mock_resp = _mock_generate_response("file result")
        with patch("httpx.post", return_value=mock_resp), \
             patch("glm_ocr.ollama_client.load_and_encode", return_value="b64"):
            result = client.ocr_file(img_path)
        assert result.text == "file result"

    def test_usage_populated(self):
        client = OllamaGLMOCRClient()
        mock_resp = _mock_generate_response("text", prompt_tokens=20, eval_tokens=8)
        with _patch_resolve(), patch("httpx.post", return_value=mock_resp):
            result = client.ocr(OCRRequest(image_url="https://x.com/a.png"))
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 28

    def test_no_image_raises(self):
        client = OllamaGLMOCRClient()
        with pytest.raises(ValueError, match="image source"):
            client.ocr(OCRRequest())

    def test_http_error_propagates(self):
        client = OllamaGLMOCRClient()
        with _patch_resolve(), patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            with pytest.raises(httpx.ConnectError):
                client.ocr(OCRRequest(image_url="https://x.com/a.png"))

    def test_json_regions_parsed(self):
        client = OllamaGLMOCRClient()
        payload = json.dumps({"regions": [{"text": "foo", "confidence": 0.9}]})
        mock_resp = _mock_generate_response(payload)
        with _patch_resolve(), patch("httpx.post", return_value=mock_resp):
            result = client.ocr(
                OCRRequest(image_url="https://x.com/a.png", output_format=OutputFormat.JSON)
            )
        assert len(result.regions) == 1
        assert result.regions[0].text == "foo"
