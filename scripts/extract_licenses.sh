#!/usr/bin/env bash
# Dump pip license metadata for production dependencies only (excludes dev group).
# Output: scripts/licenses/licenses.txt (gitignored — regenerate as needed).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/scripts/licenses"
mkdir -p "$OUT"

mapfile -t pkgs < <(
  uv export --frozen --no-dev --format requirements-txt --no-annotate --no-header --no-hashes \
    | awk '!/^-e[[:space:]]/ && NF { sub(/=.*/, ""); print }'
)
if ((${#pkgs[@]} == 0)); then
  echo "extract_licenses: no packages from uv export" >&2
  exit 1
fi

uv run pip-licenses --packages "${pkgs[@]}" --output-file "$OUT/licenses.txt"
