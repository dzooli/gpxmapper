#!/usr/bin/env bash
# Quick health check for a local Nominatim instance (e.g. Docker on port 8080).
# Default extract in start_server.sh is hungary-latest.osm.pbf — these probes
# expect Hungarian results and country_code "hu".
# Repository: ./nominatim/verify_local_hungary.sh from repo root.
# Windows release ZIP: same file is copied next to gpxmapper.exe — run:  bash verify_local_hungary.sh
#
# Usage:
#   ./nominatim/verify_local_hungary.sh
#   NOMINATIM_URL=http://host.docker.internal:8080 ./nominatim/verify_local_hungary.sh
set -euo pipefail

BASE="${NOMINATIM_URL:-http://127.0.0.1:8080}"
BASE="${BASE%/}"
UA="${NOMINATIM_USER_AGENT:-GPXMapper-NominatimVerify/1.0 (local)}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

echo "=== Nominatim Hungary checks against ${BASE} ==="

code="$(curl -sS -o /tmp/nominatim_status.txt -w '%{http_code}' -A "$UA" "${BASE}/status" || true)"
[[ "$code" == "200" ]] || die "/status HTTP ${code} (expected 200)"
grep -q 'OK' /tmp/nominatim_status.txt || die "/status body missing OK (got: $(cat /tmp/nominatim_status.txt))"

echo "OK  /status -> 200 $(tr -d '\n' </tmp/nominatim_status.txt)"

curl -sS -f -A "$UA" "${BASE}/reverse?lat=47.5070&lon=19.0456&format=json" | python -c "
import json, sys
d = json.load(sys.stdin)
addr = d.get('address') or {}
cc = (addr.get('country_code') or '').lower()
country = (addr.get('country') or '').lower()
if cc != 'hu' and 'magyarország' not in country:
    sys.exit('expected Hungary (country_code hu or Magyarország)')
if 'budapest' not in (d.get('display_name') or '').lower():
    sys.exit('expected Budapest in display_name')
print('OK  reverse Budapest ->', (d.get('display_name') or '')[:120], '...')
" || die "reverse Budapest validation failed"

curl -sS -f -A "$UA" "${BASE}/reverse?lat=47.5316&lon=21.6273&format=json" | python -c "
import json, sys
d = json.load(sys.stdin)
addr = d.get('address') or {}
cc = (addr.get('country_code') or '').lower()
if cc != 'hu':
    sys.exit(f'expected country_code hu, got {cc!r}')
if 'debrecen' not in (addr.get('city') or '').lower() and 'debrecen' not in (d.get('display_name') or '').lower():
    sys.exit('expected Debrecen in result')
print('OK  reverse Debrecen ->', (d.get('display_name') or '')[:120], '...')
" || die "reverse Debrecen validation failed"

curl -sS -f -A "$UA" --get "${BASE}/search" \
  --data-urlencode "q=Pécs" \
  --data-urlencode "format=json" \
  --data-urlencode "limit=3" | python -c "
import json, sys
a = json.load(sys.stdin)
if not a:
    sys.exit('empty search results')
if not any('pécs' in (x.get('display_name') or '').lower() for x in a):
    sys.exit('expected Pécs in at least one result')
print('OK  search Pécs ->', len(a), 'result(s), first:', (a[0].get('display_name') or '')[:100], '...')
" || die "search Pécs validation failed"

echo "=== All Hungary checks passed ==="
