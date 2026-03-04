# glm-ocr

Optical Character Recognition powered by [GLM-4V](https://open.bigmodel.cn/) vision models.

## Features

- Extract text from local images, remote URLs, or base64-encoded data
- Multiple output formats: plain text, Markdown, structured JSON regions
- Language hints for improved accuracy on non-English documents
- Simple Python API and a `glm-ocr` CLI

## Installation

```bash
pip install glm-ocr
```

Set your ZhipuAI API key:

```bash
export ZHIPUAI_API_KEY="your-key-here"
```

## Quick start

### Python API

```python
from glm_ocr import GLMOCRClient

client = GLMOCRClient()

# From a local file
result = client.ocr_file("scan.png")
print(result.text)

# From a URL
result = client.ocr_url("https://example.com/doc.jpg")
print(result.text)
```

### Advanced usage

```python
from glm_ocr import GLMOCRClient
from glm_ocr.models import OCRRequest, OutputFormat

client = GLMOCRClient()

request = OCRRequest(
    image_path="invoice.png",
    language_hint="zh",
    output_format=OutputFormat.JSON,
    extra_instructions="Extract each line item as a separate region.",
)

result = client.ocr(request)
for region in result.regions:
    print(f"[{region.confidence:.0%}] {region.text}")
```

### CLI

```bash
# OCR a local file
glm-ocr scan.png

# OCR from URL with Markdown output
glm-ocr --url https://example.com/doc.jpg --format markdown

# Save result to file
glm-ocr invoice.png --output extracted.txt
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
