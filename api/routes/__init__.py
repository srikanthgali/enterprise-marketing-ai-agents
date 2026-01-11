"""
API routes module.

Exports all API routers for inclusion in the main FastAPI app.
"""

from .prompts import router as prompts_router

__all__ = ["prompts_router"]
