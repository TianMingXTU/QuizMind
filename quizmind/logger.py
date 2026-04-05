from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import perf_counter
from typing import Iterator


LOG_DIR = Path(".quizmind_runtime")
LOG_FILE = LOG_DIR / "quizmind.log"


def get_logger(name: str = "quizmind") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(event: str, **fields: object) -> None:
    logger = get_logger()
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False))


@contextmanager
def timed_event(event: str, **fields: object) -> Iterator[None]:
    start = perf_counter()
    log_event(f"{event}.start", **fields)
    try:
        yield
    except Exception as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        log_event(f"{event}.error", duration_ms=duration_ms, error=str(exc), **fields)
        raise
    else:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        log_event(f"{event}.success", duration_ms=duration_ms, **fields)


def read_recent_logs(limit: int = 30) -> list[str]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]
