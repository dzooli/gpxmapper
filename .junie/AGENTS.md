# AI Agent instructions

## For Junie

- When commit messages are checked with (for example) `commitizen`, use conventional commit messages. For example, use `feat: add new feature` for a new feature, `fix: fix a bug` for a bug fix, and `docs: update documentation` for documentation changes.
- You need to include the Jira ticket number in the commit message. For example, if the Jira ticket number is `PROJ-123`, the commit message should be:
  ```
  feat(PROJ-123): add new feature

  [PROJ-123]: short title

  - change 1
  - change 2
  ```
  without the codefence markers.
- Do not start the message with codeblocks or any other formatting. The commit message should be a plain text message that describes the changes made in the commit.