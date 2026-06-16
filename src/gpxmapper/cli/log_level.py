"""Root logging level for the Typer CLI."""

from __future__ import annotations

import logging

_VALID = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_NOISY_HTTP_LOGGERS = ("httpx", "httpcore", "urllib3")


def apply_cli_log_level(log_level: str) -> None:
    """Set root and handler levels; keep third-party HTTP libraries quiet unless DEBUG.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Raises:
        ValueError: If ``log_level`` is not a valid level name.
    """
    name = log_level.strip().upper()
    if name not in _VALID:
        raise ValueError(
            f"Invalid --log-level {log_level!r}; use one of: {', '.join(sorted(_VALID))}."
        )
    level = getattr(logging, name)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setLevel(level)

    third_party_level = logging.DEBUG if level == logging.DEBUG else logging.WARNING
    for logger_name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(logger_name).setLevel(third_party_level)
