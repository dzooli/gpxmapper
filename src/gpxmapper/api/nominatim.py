"""Nominatim service status check API."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from ..nominatim_config import get_nominatim_base_url, probe_nominatim_status_sync

logger = logging.getLogger(__name__)


def check_nominatim_status() -> Tuple[bool, str, Optional[str]]:
    """Check whether the configured Nominatim server is reachable.

    Returns:
        Tuple of (is_available, base_url, error_message).
    """
    base_url = get_nominatim_base_url()
    ok, err = probe_nominatim_status_sync()
    if not ok:
        logger.warning("Nominatim server at %s is not reachable: %s", base_url, err)
    else:
        logger.info("Nominatim server at %s is reachable", base_url)
    return ok, base_url, err
