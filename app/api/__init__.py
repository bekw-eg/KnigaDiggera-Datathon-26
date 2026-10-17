"""API package for ForteBank Cash Discipline Inspector."""

from app.api.auth import verify_api_key
from app.api.routes import router

__all__ = ["router", "verify_api_key"]
