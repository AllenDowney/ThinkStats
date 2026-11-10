#!/usr/bin/env python3
"""Fail if a committed notebook has code cells that were never executed.

`pytest --nbmake` executes notebooks before checking them, so it passes even
when the committed .ipynb has no stored outputs. That is how chapters 8, 9 and
13 were published without figures for seven months (PROJECT_BOARD Task 4):
`jupytext --to ipynb` rewrote them from markdown and the execute step never ran.

A code cell counts as unexecuted when its execution_count is null. Empty cells
and cells that contain only comments are ignored.

Usage:
    python scripts/check_outputs.py              # soln/chap*.ipynb, examples/*.ipynb
    python scripts/check_outputs.py path.ipynb ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBS = ["soln/chap*.ipynb", "examples/*.ipynb"]


def is_trivial(source: str) -> bool:
    """True if the cell has no code: empty, or only comments and blank lines."""
    lines = [line.strip() for line in source.splitlines()]
    return all(not line or line.startswith("#") for line in lines)


def unexecuted_cells(path: Path) -> list[tuple[int, str]]:
    """Return (cell index, first line) for each unexecuted code cell."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        if is_trivial(source):
            continue
        if cell.get("execution_count") is None:
            first = next(
                (ln for ln in source.splitlines() if ln.strip() and not ln.lstrip().startswith("#")),
                "",
            )
            problems.append((i, first.strip()[:60]))
    return problems


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(p) for p in argv]
    else:
        paths = sorted(p for g in DEFAULT_GLOBS for p in REPO_ROOT.glob(g))

    failed = 0
    for path in paths:
        problems = unexecuted_cells(path)
        if problems:
            failed += 1
            print(f"{path}: {len(problems)} unexecuted code cell(s)")
            for i, first in problems[:5]:
                print(f"    cell {i}: {first}")
            if len(problems) > 5:
                print(f"    ... and {len(problems) - 5} more")

    if failed:
        print(
            f"\n{failed} notebook(s) have unexecuted cells. Re-run them, e.g.\n"
            "    jupyter nbconvert --to notebook --execute --inplace <notebook>"
        )
        return 1
    print(f"Outputs check: {len(paths)} notebooks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
