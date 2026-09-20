"""CLI: verify Nominatim ``/status`` (same probe as ``generate --geolocate``)."""

from __future__ import annotations

import typer

from ..api import check_nominatim_status
from . import app


@app.command("check-nominatim")
def check_nominatim() -> None:
    """Check that the configured Nominatim server responds to GET /status."""
    ok, base, err = check_nominatim_status()
    if not ok:
        typer.secho(f"Nominatim not reachable at {base!r}: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Nominatim OK at {base}", fg=typer.colors.GREEN)
