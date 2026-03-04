"""GLM OCR – optical character recognition powered by GLM-4V vision models."""

from glm_ocr.client import GLMOCRClient
from glm_ocr.models import OCRRequest, OCRResult
from glm_ocr.ollama_client import OllamaGLMOCRClient

__all__ = ["GLMOCRClient", "OllamaGLMOCRClient", "OCRRequest", "OCRResult"]
__version__ = "0.1.0"
