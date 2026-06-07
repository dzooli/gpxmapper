# Windows install bundle (source files)

These files are **not** part of the Python package; they are staged for the **Windows ZIP** produced in CI (`.github/workflows/build.yml`, `release.yml`):

| Source | In the ZIP |
|--------|------------|
| `nominatim/start_server.bat` | `start_server.bat` (next to `gpxmapper.exe`) |
| `nominatim/start_server.sh` | `start_server.sh` |
| `nominatim/verify_local_hungary.sh` | `verify_local_hungary.sh` |
| `install/doc/USER_GUIDE.md` | `doc/USER_GUIDE.md` |

Edit **`install/doc/USER_GUIDE.md`** for end-user wording aimed at the unzip layout; **`docs/userdocs/index.md`** stays the MkDocs “home” and documents **both** clone paths and ZIP paths. The same **`USER_GUIDE.md`** is embedded on the static site via **`docs/userdocs/windows-bundle.md`** (pymdownx snippets). Build the site with **`scripts/build-docs.ps1`** / **`scripts/build-docs.sh`** (see README **Documentation (MkDocs)**).

**Local ZIP (same as CI):** from the repository root, with **`dist\gpxmapper.exe`** already built:

```powershell
pwsh -File .\scripts\package-windows-release.ps1
```

→ **`gpxmapper-release.zip`** and a **`release\`** staging directory (see script parameters).
