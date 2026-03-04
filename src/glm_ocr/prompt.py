"""System and user prompt builders for GLM OCR tasks."""

from __future__ import annotations

from glm_ocr.models import OCRRequest, OutputFormat


_SYSTEM_PROMPT = (
    "You are an expert OCR (Optical Character Recognition) engine. "
    "When given an image, extract ALL visible text exactly as it appears, "
    "preserving the original layout, line breaks, and formatting as much as possible. "
    "Do not add commentary, explanations, or corrections unless explicitly asked."
)


def build_user_prompt(request: OCRRequest) -> str:
    parts: list[str] = []

    if request.language_hint:
        parts.append(
            f"The document is primarily in language: {request.language_hint}. "
            "Preserve that language faithfully."
        )

    match request.output_format:
        case OutputFormat.TEXT:
            parts.append(
                "Return only the extracted text, preserving newlines and indentation."
            )
        case OutputFormat.JSON:
            parts.append(
                "Return a JSON object with the key 'regions', where each element has "
                "'text' (string) and 'confidence' (0.0-1.0 float) fields. "
                "Output raw JSON only – no markdown fences."
            )
        case OutputFormat.MARKDOWN:
            parts.append(
                "Return the extracted text formatted as Markdown, using headings, "
                "lists, tables and code blocks where appropriate."
            )

    if request.extra_instructions:
        parts.append(request.extra_instructions)

    if not parts:
        parts.append("Extract all text from this image.")

    return " ".join(parts)


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT
