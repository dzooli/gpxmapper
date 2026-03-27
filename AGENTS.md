# Agent instructions (canonical: Cursor rules)

**Cursor AI** uses **`.cursor/rules/*.mdc`** as the versioned source of truth for this repo.

| File                                                                             | When it applies                                          |
|----------------------------------------------------------------------------------|----------------------------------------------------------|
| [`.cursor/rules/gpxmapper-core.mdc`](.cursor/rules/gpxmapper-core.mdc)           | Every session (project, tooling, architecture, workflow) |
| [`.cursor/rules/python-standards.mdc`](.cursor/rules/python-standards.mdc)       | When editing Python (`**/*.py`)                          |
| [`.cursor/rules/git-github-workflow.mdc`](.cursor/rules/git-github-workflow.mdc) | Every session (commits, branches, PRs)                   |

This file remains a **short pointer** for Copilot and other tools that expect `AGENTS.md` at the root. **Do not** grow
large duplicated policy here — update the `.mdc` rules instead.

Human-facing detail: **README.md**, **docs/**.
