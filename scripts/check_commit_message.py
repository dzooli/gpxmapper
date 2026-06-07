"""commit-msg hook: conventional subject + Jira issue block in commit body.

See .cursor/rules/git-github-workflow.mdc. Invoked by pre-commit (commit-msg stage).

Subject line must match Commitizen / changelog parsers (type(scope): summary).
Body must start after a blank line with ``[PROJ-123]: Issue title`` then details.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# First line only — Commitizen conventional (no ticket prefix on subject).
SUBJECT = re.compile(
    r"^(build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([^)]*\))?"
    r"(!)?"
    r": .+"
)

# First non-empty line of the body (after blank line following subject).
ISSUE_TITLE_LINE = re.compile(r"^\[[A-Z]{2,}-\d+\]:\s+.+$")

ALLOW_PREFIX = (
    "Merge ",
    "Revert ",
    "Pull request",
    "fixup!",
    "squash!",
    "amend!",
)


def _strip_commit_comments(text: str) -> str:
    """Drop git commit --verbose trailer and # comment lines."""
    lines: list[str] = []
    for line in text.splitlines():
        if "# ------------------------ >8 ------------------------" in line:
            break
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_commit_message.py <commit-msg-file>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    raw = _strip_commit_comments(path.read_text(encoding="utf-8")).lstrip("\ufeff")
    lines = raw.splitlines()
    first = lines[0].strip() if lines else ""
    if not first:
        print("empty commit message", file=sys.stderr)
        return 1
    if any(first.startswith(p) for p in ALLOW_PREFIX):
        return 0

    if not SUBJECT.match(first):
        print(
            "Commit subject (first line) must be conventional (Commitizen / changelog):\n"
            "  <type>(<scope>): <short summary>\n"
            "Examples: feat(cli): add flag  |  fix: handle edge case\n"
            "Exempt: Merge / Revert / Pull request / fixup! / squash! / amend!\n"
            f"Got: {first!r}",
            file=sys.stderr,
        )
        return 1

    # Require a blank line after the subject, then body with Jira-style issue line.
    if "\n\n" not in raw:
        print(
            "After the subject line, add a blank line, then the issue block and details.\n"
            "Example:\n"
            "  feat(cli): add geolocate flag\n"
            "\n"
            "  [GPXM-1]: Short issue title\n"
            "  - bullet change\n",
            file=sys.stderr,
        )
        return 1

    _, body = raw.split("\n\n", 1)
    body_lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not body_lines:
        print(
            "Commit body is empty. After the blank line, add:\n"
            "  [<KEY>-<number>]: <issue title>\n"
            "  - change bullets…",
            file=sys.stderr,
        )
        return 1

    if not ISSUE_TITLE_LINE.match(body_lines[0]):
        print(
            "First line of the commit body (after the blank line) must be:\n"
            "  [<KEY>-<number>]: <issue title>\n"
            "Use the Jira work item key in brackets, then colon and space, then title.\n"
            f"Got: {body_lines[0]!r}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
