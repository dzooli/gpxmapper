"""CLI: verify Nominatim ``/status`` (same probe as ``generate --geolocate``)."""

import typer

from ..nominatim_config import get_nominatim_base_url, probe_nominatim_status_sync
from . import app


@app.command("check-nominatim")
def check_nominatim() -> None:
    """Check that the configured Nominatim server responds to GET /status."""
    ok, err = probe_nominatim_status_sync()
    base = get_nominatim_base_url()
    if not ok:
        typer.secho(f"Nominatim not reachable at {base!r}: {err}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho(f"Nominatim OK at {base}", fg=typer.colors.GREEN)
