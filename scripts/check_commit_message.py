"""commit-msg hook: require issue key on the first line of the commit message.

See .cursor/rules/git-github-workflow.mdc. Invoked by pre-commit (commit-msg stage).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# [<KEY>-<digits>] <type>(<scope>): <description> — scope and ! optional
SUBJECT = re.compile(
    r"^(\[[A-Z]+-\d+\]) "
    r"(build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)"
    r"(\([^)]*\))?"
    r"(!)?"
    r": .+"
)

ALLOW_PREFIX = (
    "Merge ",
    "Revert ",
    "Pull request",
    "fixup!",
    "squash!",
    "amend!",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_commit_message.py <commit-msg-file>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    raw = path.read_text(encoding="utf-8")
    lines = raw.lstrip("\ufeff").splitlines()
    first = lines[0].strip() if lines else ""
    if not first:
        print("empty commit message", file=sys.stderr)
        return 1
    if any(first.startswith(p) for p in ALLOW_PREFIX):
        return 0
    if SUBJECT.match(first):
        return 0
    print(
        "Commit subject (first line) must be:\n"
        "  [<KEY>-<number>] <type>(<scope>): <description>\n"
        "Derive the key from the branch when possible (e.g. [GPXM-1] feat(cli): add flag).\n"
        "Exempt: Merge / Revert / Pull request / fixup! / squash! / amend!\n"
        f"Got: {first!r}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
