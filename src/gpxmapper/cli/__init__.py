"""Command-line interface for GPX to video mapper."""

import logging
import sys

import typer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# Create typer app before importing command modules (they register on ``app``).
app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=True,
    help="GPX to video mapper - creates videos from GPX tracks",
)


@app.callback()
def _cli_root(
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="GPXMAPPER_LOG_LEVEL",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). "
        "Per-request HTTP and tile-cache detail uses DEBUG.",
        show_default=True,
    ),
) -> None:
    """Shared CLI options (run before any subcommand)."""
    from .log_level import apply_cli_log_level

    try:
        apply_cli_log_level(log_level)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


# Import commands (side-effect: register subcommands on ``app``)
from . import check_nominatim  # noqa: E402, F401
from . import clear_cache  # noqa: E402, F401
from . import generate  # noqa: E402, F401
from . import info  # noqa: E402, F401

if __name__ == "__main__":
    app()
