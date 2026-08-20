"""Command safety and read-only whitelist policies for tool execution."""

from __future__ import annotations

SAFE_PREFIXES: tuple[str, ...] = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "git tag",
    "git rev-parse",
    "git remote",
    "git describe",
    "ls",
    "dir",
    "pwd",
    "echo",
    "cat",
    "head",
    "tail",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "ag",
    "find",
    "which",
    "where",
    "type",
    "wc",
    "diff",
    "stat",
    "file",
    "tree",
    "env",
    "printenv",
    "pytest",
    "python -m unittest",
    "python3 -m unittest",
    ".venv/bin/python -m unittest",
    "npm test",
    "yarn test",
    "pnpm test",
    "cargo test",
    "go test",
    "ruff check",
    "mypy",
    "flake8",
    "black --check",
    "pylint",
    "eslint",
)

DANGEROUS_SUBSTRINGS: tuple[str, ...] = (
    ">",
    ">>",
    "rm ",
    "rmdir",
    "mv ",
    "chmod ",
    "chown ",
    "kill ",
    "pkill ",
    "sudo ",
    "dd ",
    "mkfs",
    "git push",
    "git reset",
    "git clean",
    "git checkout",
    "git commit",
    "git merge",
    "git rebase",
    "git stash pop",
    "git stash drop",
    "pip install",
    "pip uninstall",
    "npm install",
    "npm uninstall",
    "yarn add",
    "yarn remove",
    "cargo install",
)


def is_safe_command(command: str) -> bool:
    """Return True if command is a recognized safe read-only inspection command."""
    cmd = command.strip()
    if not cmd:
        return False
    if any(dang in cmd for dang in DANGEROUS_SUBSTRINGS):
        return False
    return any(cmd == prefix or cmd.startswith(f"{prefix} ") for prefix in SAFE_PREFIXES)
