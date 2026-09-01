from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Iterator


logger = logging.getLogger("omniassist")


@contextmanager
def request_context() -> Iterator[dict]:
    """Create lightweight request metadata without logging user content."""
    context = {"request_id": uuid.uuid4().hex, "started_at": time.perf_counter()}
    try:
        yield context
    finally:
        context["latency_ms"] = round(
            (time.perf_counter() - context["started_at"]) * 1000, 2
        )


def log_request(context: dict, *, status: str, retrieved: int | None = None) -> None:
    """Emit structured-ish operational information while avoiding prompt logging."""
    fields = {
        "request_id": context["request_id"],
        "latency_ms": context.get("latency_ms"),
        "status": status,
    }
    if retrieved is not None:
        fields["retrieved_documents"] = retrieved
    logger.info("omniassist_request %s", fields)
