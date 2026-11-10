#!/usr/bin/env python3
"""Fail if a committed notebook's source differs from its markdown pair.

The .md files in soln/ and examples/ are canonical; each .ipynb is generated
from its .md by scripts/book_pipeline.py. The HTML book is built from the
.ipynb, so an edit to a .md that is never rebuilt does not reach readers.
That happened to three published examples from August to November 2026
(PROJECT_BOARD Task 3).

Each notebook is rendered to markdown with jupytext and compared with its .md,
ignoring the YAML header (it records the jupytext version). Only git-tracked
pairs are checked, so local drafts don't trip it.

Usage:
    python scripts/check_sync.py              # tracked soln/ and examples/ pairs
    python scripts/check_sync.py path.ipynb ...
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

import jupytext

REPO_ROOT = Path(__file__).resolve().parents[1]
DIRS = ["soln", "examples"]


def strip_header(text: str) -> list[str]:
    """Drop a leading YAML front-matter block and return the remaining lines."""
    lines = text.splitlines()
    if lines and lines[0] == "---":
        try:
            end = lines.index("---", 1)
        except ValueError:
            return lines
        lines = lines[end + 1 :]
    return lines


def tracked_notebooks() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", *[f"{d}/*.ipynb" for d in DIRS]],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return [REPO_ROOT / p for p in out.split()]
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for d in DIRS for p in (REPO_ROOT / d).glob("*.ipynb"))


def drift(ipynb: Path, md: Path) -> list[str]:
    """Return a unified diff (notebook vs markdown), empty if in sync."""
    rendered = jupytext.writes(jupytext.read(ipynb), fmt="md")
    source = md.read_text(encoding="utf-8")
    return list(
        difflib.unified_diff(
            strip_header(rendered),
            strip_header(source),
            fromfile=str(ipynb.relative_to(REPO_ROOT)),
            tofile=str(md.relative_to(REPO_ROOT)),
            lineterm="",
            n=1,
        )
    )


def main(argv: list[str]) -> int:
    paths = [Path(p).resolve() for p in argv] if argv else tracked_notebooks()
    checked = failed = 0
    for ipynb in paths:
        md = ipynb.with_suffix(".md")
        if not md.exists():
            continue
        checked += 1
        diff = drift(ipynb, md)
        if diff:
            failed += 1
            changed = sum(1 for ln in diff if ln[:1] in "+-" and ln[:3] not in ("+++", "---"))
            print(f"{ipynb.relative_to(REPO_ROOT)}: {changed} line(s) differ from {md.name}")
            for line in diff[2:14]:
                print("    " + line[:100])

    if failed:
        print(
            f"\n{failed} notebook(s) out of sync with their .md. The .md is canonical;\n"
            "rebuild with  python scripts/book_pipeline.py <chapNN | example:name>"
        )
        return 1
    print(f"Sync check: {checked} notebook/markdown pairs OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
