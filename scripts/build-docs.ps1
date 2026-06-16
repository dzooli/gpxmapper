# Build MkDocs without Material / pymdownx MkDocs 2.0 console notices.
# Usage: pwsh -File .\scripts\build-docs.ps1 build
#        pwsh -File .\scripts\build-docs.ps1 serve
$ErrorActionPreference = 'Stop'
$env:NO_MKDOCS_2_WARNING = '1'
$env:DISABLE_MKDOCS_2_WARNING = 'true'
if ($args.Count -eq 0) {
    & uv run mkdocs build
}
else {
    & uv run mkdocs @args
}
