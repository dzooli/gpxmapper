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

# Import commands (side-effect: register subcommands on ``app``)
from . import check_nominatim  # noqa: E402, F401
from . import clear_cache  # noqa: E402, F401
from . import generate  # noqa: E402, F401
from . import info  # noqa: E402, F401

if __name__ == "__main__":
    app()
