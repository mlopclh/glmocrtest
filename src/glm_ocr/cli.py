"""Command-line interface for GLM OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel

from glm_ocr.client import GLMOCRClient
from glm_ocr.models import OCRRequest, OutputFormat
from glm_ocr.ollama_client import OllamaGLMOCRClient

app = typer.Typer(
    name="glm-ocr",
    help="Optical Character Recognition powered by GLM-4V vision models.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")


@app.command()
def run(
    image: Annotated[
        Optional[Path],
        typer.Argument(help="Path to a local image file."),
    ] = None,
    url: Annotated[
        Optional[str],
        typer.Option("--url", "-u", help="URL of a remote image."),
    ] = None,
    ollama: Annotated[
        bool,
        typer.Option("--ollama", help="Use local Ollama backend instead of ZhipuAI cloud."),
    ] = False,
    ollama_host: Annotated[
        str,
        typer.Option("--ollama-host", help="Ollama server base URL."),
    ] = "http://localhost:11434",
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="ZHIPUAI_API_KEY", help="ZhipuAI API key (cloud only)."),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Model name (cloud: glm-4v-plus, ollama: glm-ocr:latest)."),
    ] = None,
    language: Annotated[
        Optional[str],
        typer.Option("--language", "-l", help="Language hint (BCP-47 code)."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TEXT,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Write result to this file."),
    ] = None,
) -> None:
    """Extract text from an image using GLM vision models."""
    if image is None and url is None:
        err_console.print("Error: provide either an image path or --url.")
        raise typer.Exit(1)

    request = OCRRequest(
        image_path=image,
        image_url=url,
        language_hint=language,
        output_format=output_format,
    )

    try:
        if ollama:
            effective_model = model or "glm-ocr:latest"
            client: GLMOCRClient | OllamaGLMOCRClient = OllamaGLMOCRClient(
                model=effective_model, base_url=ollama_host
            )
        else:
            effective_model = model or "glm-4v-plus"
            client = GLMOCRClient(api_key=api_key, model=effective_model)
        result = client.ocr(request)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"OCR failed: {exc}")
        raise typer.Exit(1)

    text = result.raw_text

    if output:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]Result written to {output}[/green]")
    else:
        console.print(Panel(text, title="Extracted Text", expand=False))

    if result.usage:
        console.print(
            f"[dim]Tokens – prompt: {result.usage.get('prompt_tokens', '?')}, "
            f"completion: {result.usage.get('completion_tokens', '?')}[/dim]"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
