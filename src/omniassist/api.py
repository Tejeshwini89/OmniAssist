from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import Lock
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.omniassist.generator import generate_answer


logger = logging.getLogger("omniassist")

app = FastAPI(
    title="OmniAssist",
    description="Enterprise GenAI Assistant using RAG",
    version="1.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask OmniAssist",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class RateLimiter:
    """Small process-local limiter for accidental or abusive request bursts."""

    def __init__(self, limit: int = 30, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = monotonic()
        with self._lock:
            timestamps = self._requests[key]
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True


rate_limiter = RateLimiter()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or uuid4().hex


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = _request_id(request)
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "omniassist_request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred."},
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(elapsed_ms)
    logger.info(
        "omniassist_request request_id=%s method=%s path=%s status=%s latency_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "OmniAssist",
    }


@app.get("/metrics")
def metrics():
    """Expose non-sensitive process-local operational counters."""
    tracked_requests = sum(len(values) for values in rate_limiter._requests.values())
    return {
        "service": "OmniAssist",
        "rate_limit_window_seconds": rate_limiter.window_seconds,
        "active_rate_limit_entries": tracked_requests,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request):
    client_key = http_request.client.host if http_request.client else "unknown"
    if not rate_limiter.allow(client_key):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
        )

    try:
        return generate_answer(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "answer_generation_failed request_id=%s",
            getattr(http_request.state, "request_id", "unknown"),
        )
        raise HTTPException(
            status_code=502,
            detail="The answer generation service is temporarily unavailable.",
        ) from exc
