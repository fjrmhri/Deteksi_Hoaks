"""FastAPI service for IndoBERT-based hoax detection."""

from .main import app  # re-export for ``uvicorn app:app`` style launches

__all__ = ["app"]
