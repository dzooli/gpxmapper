#!/usr/bin/env bash
# Build MkDocs site without Material / pymdownx MkDocs 2.0 console notices.
# Usage: ./scripts/build-docs.sh build | serve | ...
set -euo pipefail
export NO_MKDOCS_2_WARNING=1
export DISABLE_MKDOCS_2_WARNING=true
exec uv run mkdocs "$@"
