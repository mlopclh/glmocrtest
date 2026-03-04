"""Basic usage examples for the GLM OCR library."""

import os
from pathlib import Path

from glm_ocr import GLMOCRClient
from glm_ocr.models import OCRRequest, OutputFormat

# -----------------------------------------------------------
# 1. OCR from a local file (plain text output)
# -----------------------------------------------------------
def example_local_file():
    client = GLMOCRClient()  # reads ZHIPUAI_API_KEY from env
    result = client.ocr_file("invoice.png")
    print(result.text)


# -----------------------------------------------------------
# 2. OCR from a URL with Markdown output
# -----------------------------------------------------------
def example_url_markdown():
    client = GLMOCRClient()
    result = client.ocr_url(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
        output_format=OutputFormat.MARKDOWN,
    )
    print(result.text)


# -----------------------------------------------------------
# 3. Advanced: Chinese document with JSON regions
# -----------------------------------------------------------
def example_advanced():
    client = GLMOCRClient()
    request = OCRRequest(
        image_url="https://example.com/chinese_doc.jpg",
        language_hint="zh",
        output_format=OutputFormat.JSON,
        extra_instructions="If a table is present, extract each cell as a separate region.",
    )
    result = client.ocr(request)
    for region in result.regions:
        print(f"[{region.confidence:.0%}] {region.text}")


if __name__ == "__main__":
    example_local_file()
