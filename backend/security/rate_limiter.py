"""
Rate limiting middleware and sliding window bucket for Argus API endpoints.
Protects against brute-force, credential stuffing, and DoS attacks.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    Thread-safe in-memory sliding window rate limiter.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # map key -> list of float timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def is_allowed(
        self, key: str, max_requests: int, window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check if a request under `key` is allowed.
        Returns:
            (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds

        with self._lock:
            # Periodic cleanup of expired entries (every 60s)
            if now - self._last_cleanup > 60:
                self._cleanup(window_start)
                self._last_cleanup = now

            timestamps = self._requests[key]
            # Prune timestamps older than window
            self._requests[key] = [t for t in timestamps if t > window_start]
            active_count = len(self._requests[key])

            if active_count >= max_requests:
                # Find earliest timestamp in window to compute retry-after
                earliest = self._requests[key][0]
                retry_after = max(1, int(window_seconds - (now - earliest)))
                return False, retry_after

            self._requests[key].append(now)
            return True, 0

    def _cleanup(self, cutoff: float) -> None:
        """Remove keys that have no timestamps newer than cutoff."""
        empty_keys = []
        for key, timestamps in self._requests.items():
            valid = [t for t in timestamps if t > cutoff]
            if not valid:
                empty_keys.append(key)
            else:
                self._requests[key] = valid
        for k in empty_keys:
            del self._requests[k]


_LIMITER = SlidingWindowRateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI / Starlette middleware applying tiered sliding-window rate limits.
    """

    def __init__(self, app):
        super().__init__(app)
        self.enabled = os.environ.get("ARGUS_DISABLE_RATE_LIMIT", "0").strip() != "1"
        self.auth_limit = int(os.environ.get("ARGUS_RATE_LIMIT_AUTH", "30"))  # per min
        self.api_limit = int(os.environ.get("ARGUS_RATE_LIMIT_API", "300"))   # per min

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path
        # Static files and docs are exempt from rate limiting
        if path in ("/", "/docs", "/openapi.json", "/redoc") or path.startswith("/static") or path.startswith("/assets"):
            return await call_next(request)

        # Extract client IP safely (respecting reverse-proxy headers if configured)
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # Tier 1: Auth endpoints (stricter limit)
        if path.startswith("/api/v1/auth/"):
            key = f"auth:{client_ip}"
            allowed, retry_after = _LIMITER.is_allowed(key, max_requests=self.auth_limit, window_seconds=60)
            if not allowed:
                logger.warning(f"Rate limit exceeded on auth endpoint by {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded on authentication endpoints. Retry after {retry_after}s."
                    },
                    headers={"Retry-After": str(retry_after)},
                )

        # Tier 2: General API endpoints
        elif path.startswith("/api/"):
            key = f"api:{client_ip}"
            allowed, retry_after = _LIMITER.is_allowed(key, max_requests=self.api_limit, window_seconds=60)
            if not allowed:
                logger.warning(f"API rate limit exceeded by {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"API rate limit exceeded. Retry after {retry_after}s."
                    },
                    headers={"Retry-After": str(retry_after)},
                )

        return await call_next(request)
