"""Tests for glm_ocr.prompt."""

from glm_ocr.models import OCRRequest, OutputFormat
from glm_ocr.prompt import build_user_prompt, get_system_prompt


class TestSystemPrompt:
    def test_non_empty(self):
        assert len(get_system_prompt()) > 10

    def test_mentions_ocr(self):
        assert "OCR" in get_system_prompt() or "text" in get_system_prompt().lower()


class TestUserPrompt:
    def test_text_format(self):
        req = OCRRequest(image_url="https://x.com/a.png", output_format=OutputFormat.TEXT)
        prompt = build_user_prompt(req)
        assert "text" in prompt.lower()

    def test_json_format(self):
        req = OCRRequest(image_url="https://x.com/a.png", output_format=OutputFormat.JSON)
        prompt = build_user_prompt(req)
        assert "json" in prompt.lower()

    def test_markdown_format(self):
        req = OCRRequest(image_url="https://x.com/a.png", output_format=OutputFormat.MARKDOWN)
        prompt = build_user_prompt(req)
        assert "markdown" in prompt.lower() or "Markdown" in prompt

    def test_language_hint_included(self):
        req = OCRRequest(image_url="https://x.com/a.png", language_hint="zh")
        prompt = build_user_prompt(req)
        assert "zh" in prompt

    def test_extra_instructions_appended(self):
        req = OCRRequest(
            image_url="https://x.com/a.png",
            extra_instructions="Ignore watermarks.",
        )
        prompt = build_user_prompt(req)
        assert "Ignore watermarks." in prompt
