"""Tests for glm_ocr.client – uses a mock ZhipuAI backend."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import glm_ocr.client as _client_module
from glm_ocr.client import GLMOCRClient
from glm_ocr.models import OCRRequest, OCRResult, OutputFormat


def _make_mock_client(response_text: str, usage: dict | None = None):
    """Return a GLMOCRClient whose underlying ZhipuAI client is mocked."""
    usage_obj = None
    if usage:
        usage_obj = SimpleNamespace(**usage)

    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=response_text)
            )
        ],
        usage=usage_obj,
    )

    mock_zhipu = MagicMock()
    mock_zhipu.chat.completions.create.return_value = completion

    with patch.object(_client_module, "ZhipuAI", mock_zhipu):
        client = GLMOCRClient(api_key="test-key")

    client._client = mock_zhipu
    return client


class TestGLMOCRClientInit:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
        mock_zhipu = MagicMock()
        with patch.object(_client_module, "ZhipuAI", mock_zhipu):
            with pytest.raises(ValueError, match="API key"):
                GLMOCRClient(api_key=None)

    def test_env_var_accepted(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "env-key")
        mock_zhipu = MagicMock()
        with patch.object(_client_module, "ZhipuAI", mock_zhipu):
            client = GLMOCRClient()
        assert client.model == "glm-4v-plus"

    def test_missing_zhipuai_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ZHIPUAI_API_KEY", "key")
        with patch.object(_client_module, "ZhipuAI", None):
            with pytest.raises(ImportError, match="zhipuai"):
                GLMOCRClient()


class TestOCR:
    def test_ocr_returns_result(self):
        client = _make_mock_client("Hello, World!")
        req = OCRRequest(image_url="https://example.com/img.png")
        result = client.ocr(req)
        assert isinstance(result, OCRResult)
        assert result.text == "Hello, World!"

    def test_ocr_no_image_raises(self):
        client = _make_mock_client("")
        with pytest.raises(ValueError, match="image source"):
            client.ocr(OCRRequest())

    def test_ocr_usage_populated(self):
        client = _make_mock_client(
            "text",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        result = client.ocr(OCRRequest(image_url="https://x.com/a.png"))
        assert result.usage["total_tokens"] == 15

    def test_ocr_json_format_parses_regions(self):
        regions_json = json.dumps(
            {"regions": [{"text": "foo", "confidence": 0.95}]}
        )
        client = _make_mock_client(regions_json)
        req = OCRRequest(
            image_url="https://x.com/a.png",
            output_format=OutputFormat.JSON,
        )
        result = client.ocr(req)
        assert len(result.regions) == 1
        assert result.regions[0].text == "foo"
        assert result.regions[0].confidence == pytest.approx(0.95)

    def test_ocr_json_bad_payload_gives_empty_regions(self):
        client = _make_mock_client("not json at all")
        req = OCRRequest(
            image_url="https://x.com/a.png",
            output_format=OutputFormat.JSON,
        )
        result = client.ocr(req)
        assert result.regions == []

    def test_ocr_file_convenience(self, tmp_path):
        img_path = tmp_path / "img.png"
        img_path.write_bytes(b"fake")

        client = _make_mock_client("file text")
        with patch("glm_ocr.client.load_and_encode", return_value="b64data"):
            result = client.ocr_file(img_path)
        assert result.text == "file text"

    def test_ocr_url_convenience(self):
        client = _make_mock_client("url text")
        result = client.ocr_url("https://x.com/a.png")
        assert result.text == "url text"
