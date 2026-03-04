"""Extract text from an image using GLM-OCR via the Transformers library.

Usage
-----
    # Basic text extraction
    python transformers_ocr.py image.png

    # Specific recognition mode
    python transformers_ocr.py image.png --mode formula
    python transformers_ocr.py image.png --mode table

    # Save output to a file
    python transformers_ocr.py image.png --output result.txt

    # Use a local model checkpoint
    python transformers_ocr.py image.png --model-path ./glm-ocr-local

Requirements
------------
    pip install "transformers>=4.49.0" torch Pillow
    # If transformers does not yet include GLM-OCR support on PyPI, install
    # directly from source:
    #   pip install git+https://github.com/huggingface/transformers.git
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_MODES = ("text", "formula", "table")
MODE_PROMPTS = {
    "text": "Text Recognition:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}
DEFAULT_MODEL = "zai-org/GLM-OCR"


def build_messages(image_path: Path, mode: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": MODE_PROMPTS[mode]},
            ],
        }
    ]


def run_ocr(
    image_path: Path,
    mode: str = "text",
    model_path: str = DEFAULT_MODEL,
    max_new_tokens: int = 8192,
    device_map: str = "auto",
) -> str:
    """Load GLM-OCR and extract text from *image_path*.

    Parameters
    ----------
    image_path:
        Path to the input image (JPEG, PNG, TIFF, …).
    mode:
        Recognition mode – one of ``"text"``, ``"formula"``, ``"table"``.
    model_path:
        Hugging Face model ID or path to a local checkpoint.
    max_new_tokens:
        Maximum number of tokens the model may generate.
    device_map:
        Passed verbatim to ``from_pretrained`` – ``"auto"`` selects GPU when
        available, otherwise CPU.

    Returns
    -------
    str
        The raw text produced by the model.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with:\n"
            '  pip install "transformers>=4.49.0" torch Pillow'
        ) from exc

    print(f"Loading model from: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
    )

    messages = build_messages(image_path, mode)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # token_type_ids is not used by this model family
    inputs.pop("token_type_ids", None)

    print(f"Running OCR (mode={mode}, max_new_tokens={max_new_tokens}) …", flush=True)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
    output_text: str = processor.decode(new_tokens, skip_special_tokens=False)
    return output_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from an image using GLM-OCR (Transformers)."
    )
    parser.add_argument("image", type=Path, help="Path to the input image file.")
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="text",
        help="Recognition mode (default: text).",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help=f"HF model ID or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write extracted text to this file instead of stdout.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Torch device map passed to from_pretrained (default: auto).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    text = run_ocr(
        image_path=args.image,
        mode=args.mode,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
    )

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Result written to {args.output}")
    else:
        print("\n--- Extracted text ---")
        print(text)


if __name__ == "__main__":
    main()
