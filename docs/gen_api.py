from __future__ import annotations

from pathlib import Path
import mkdocs_gen_files as gen_files


PACKAGE_ROOT = Path("src")
PACKAGE_IMPORT = "gpxmapper"

nav = gen_files.Nav()

modules = []
for path in sorted((PACKAGE_ROOT / PACKAGE_IMPORT).rglob("*.py")):
    # Compute the module path relative to src and strip suffix
    module_path = path.relative_to(PACKAGE_ROOT).with_suffix("")
    parts = list(module_path.parts)

    # Skip private or cache dirs
    if any(p.startswith("_") or p == "__pycache__" for p in parts):
        continue

    # Drop __init__ from the end
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        continue

    mod_name = ".".join(parts)
    modules.append(mod_name)

    doc_path = Path("reference", *parts).with_suffix(".md")
    nav_path = parts

    nav[nav_path] = doc_path.as_posix()

    with gen_files.open(doc_path, "w") as fd:
        # Root heading for the module
        fd.write(f"# {mod_name}\n\n")
        # mkdocstrings directive
        fd.write(f"::: {mod_name}\n")

# Write the summary for literate-nav
with gen_files.open(Path("reference", "SUMMARY.md"), "w") as fd:
    fd.writelines(nav.build_literate_nav())
