"""Check that tracked repository paths follow the research-artifact policy."""
from __future__ import annotations

import re
import subprocess


FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
VERSIONED_STEM = re.compile(r"(?:^|[_-])(?:v\d+|reprocheck)(?:[_-]|\.|$)", re.IGNORECASE)
PROVIDER_NAME = re.compile(r"gemini", re.IGNORECASE)
PROVIDER_ALLOWED_PREFIX = "docs/notebooks/providers/gemini/"


def tracked_paths() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def findings(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for path in paths:
        parts = set(path.split("/"))
        if parts & FORBIDDEN_PARTS:
            problems.append(f"tracked cache path: {path}")
        if VERSIONED_STEM.search(path):
            problems.append(f"unstable version suffix: {path}")
        if PROVIDER_NAME.search(path) and not path.startswith(PROVIDER_ALLOWED_PREFIX):
            problems.append(f"provider name in canonical path: {path}")
    return problems


def main() -> None:
    problems = findings(tracked_paths())
    if problems:
        raise SystemExit("Repository hygiene audit failed:\n- " + "\n- ".join(problems))
    print("Repository hygiene audit passed")


if __name__ == "__main__":
    main()
