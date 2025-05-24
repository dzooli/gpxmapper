"""Command-line interface for GPX to video mapper."""

import logging
import sys
import typer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Create typer app
app = typer.Typer(invoke_without_command=True, no_args_is_help=True,
                  help="GPX to video mapper - creates videos from GPX tracks")

# Import commands
from .generate import generate
from .info import info
from .clear_cache import clear_cache

if __name__ == "__main__":
    app()
