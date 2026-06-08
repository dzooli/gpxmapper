#!/usr/bin/env python3
"""Write install/doc/THIRD_PARTY_NOTICES.md from prod-only deps + static license appendices.

Run from repo root:  python scripts/generate_third_party_notices.py
Requires: uv (calls `uv export` and `uv run pip-licenses`).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "install" / "doc" / "THIRD_PARTY_NOTICES.md"
APPENDIX_DIR = REPO_ROOT / "install" / "doc" / "license-appendix"


def prod_package_names() -> list[str]:
    cmd = [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--format",
        "requirements-txt",
        "--no-annotate",
        "--no-header",
        "--no-hashes",
    ]
    raw = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
    names: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("-e"):
            continue
        name = line.split("=", 1)[0].strip()
        if name:
            names.append(name)
    return names


def pip_licenses_markdown(packages: list[str]) -> str:
    cmd = [
        "uv",
        "run",
        "pip-licenses",
        "--format",
        "markdown",
        "--with-authors",
        "--with-urls",
        "--packages",
        *packages,
    ]
    return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)


def main() -> int:
    packages = prod_package_names()
    if not packages:
        print("No packages from uv export", file=sys.stderr)
        return 1

    table = pip_licenses_markdown(packages)

    apache = (APPENDIX_DIR / "APACHE-2.0.txt").read_text(encoding="utf-8")
    mpl = (APPENDIX_DIR / "MPL-2.0.txt").read_text(encoding="utf-8")

    body = f"""# Third-party software notices (GPX Mapper)

This document lists **Python distribution packages** bundled into the standalone
**`gpxmapper.exe`** build (production dependencies from **`pyproject.toml`** / **`uv.lock`**,
excluding the dev dependency group). It is **not legal advice**.

- **Native libraries** shipped inside wheels (for example codecs inside **OpenCV**) are **not**
  fully enumerated here; operators may need separate notices for those binaries.
- **OpenH264** and similar components are **out of scope** for this file unless you add them
  explicitly after your own verification.

The table below is generated with **`pip-licenses`**. For **BSD-3-Clause**, **MIT**, **ISC**,
and similar permissive licenses, the **copyright and disclaimer** text for each package is
typically found in that package’s **PyPI sdist** or repository (see **URL** column). Packages
under **Apache-2.0** or **MPL-2.0** are also covered by the **full license texts** in the
appendices at the end of this file.

Regenerate after dependency changes:

```bash
python scripts/generate_third_party_notices.py
```

---

{table}

---

## Appendix A — Apache License, Version 2.0

Applies to components in the table above whose license is **Apache License 2.0** / **Apache Software License**
(for example **gpxpy**, **opencv-python**, **requests**).

```
{apache.strip()}
```

---

## Appendix B — Mozilla Public License 2.0

Applies to components in the table above whose license is **MPL 2.0** (for example **certifi**).

```
{mpl.strip()}
```
"""

    OUT.write_text(body, encoding="utf-8", newline="\n")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
