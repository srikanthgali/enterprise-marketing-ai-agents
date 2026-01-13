"""
Custom middleware for the FastAPI application.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time
from typing import Callable
import logging

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a unique request ID to each request.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    In production, use Redis-based rate limiting.
    """

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}  # {ip: [(timestamp, count)]}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        from fastapi.responses import JSONResponse
        import time

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old entries
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                (ts, count)
                for ts, count in self.request_counts[client_ip]
                if current_time - ts < 60
            ]

        # Count requests in last minute
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        request_count = sum(count for _, count in self.request_counts[client_ip])

        if request_count >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "RateLimitExceeded",
                    "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                },
            )

        # Add current request
        self.request_counts[client_ip].append((current_time, 1))

        response = await call_next(request)
        response.headers["X-Rate-Limit"] = str(self.requests_per_minute)
        response.headers["X-Rate-Limit-Remaining"] = str(
            self.requests_per_minute - request_count - 1
        )

        return response
