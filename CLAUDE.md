# ThinkStats — working notes for Claude

Companion repo for *Think Stats*, 3rd edition. The working branch is **`v3`**
(not `main`); CI, the Colab links and the published book all point at `v3`.
The book is Jupyter notebooks, published as a Jupyter Book at
<https://allendowney.github.io/ThinkStats/>.

## Environment

```bash
make create_environment_dev   # conda env "ThinkStats", Python 3.12, via mamba
conda activate ThinkStats
```

The dependencies are listed in **four places**: `requirements.txt`,
`requirements-dev.txt`, `environment.yml` + `environment-dev.yml`, and
`pyproject.toml` (for uv, added by PR #23). Change one, change all of them.
CI installs from `requirements-dev.txt`, unpinned, so it picks up new library
releases before the local env does (that is how pandas 3 broke chapter 12
in CI but not locally; PROJECT_BOARD Task 1).

## Which file is the source?

Since 2026-08-27 the **`.md` files in `soln/` and `examples/` are canonical**.
The `.ipynb` beside each one is generated from it, then executed:

```bash
make update-chapter CHAPTER=chap07           # or:
python scripts/book_pipeline.py chap07 example:binom_skeet
```

`book_pipeline.py` runs `jupytext --to ipynb` (which **discards every stored
output**), then `nbconvert --execute`, then writes the solution-free copy in
`nb/`. If the execute step fails or is skipped (`--no-execute`), the notebook
is left with no outputs and the published page loses its figures. That
happened to chapters 8, 9 and 13 and went unnoticed for seven months
(Task 4). Always check that the execute step finished.

The pairs are **not** linked with jupytext pairing metadata, so
`jupytext --sync` does not apply. Edit the `.md`, then rebuild.

## Where copies go

```
soln/<name>.md ──jupytext──> soln/<name>.ipynb ──nbconvert --execute──> outputs
                                    │
                                    ├──> nb/   student copies, solutions and outputs
                                    │          stripped by nb/prep_notebooks.py
                                    └──> jb/   copied by jb/build.sh for the HTML book;
                                               jb/*.ipynb and jb/*.md are gitignored
```

Do not edit notebooks in `nb/` or `jb/`; the next build overwrites them.
Data files are canonical in `data/`; the copies under `soln/` are gitignored.

Adding a chapter or example to the book means touching `jb/_toc.yml`, the
`cp` list in `jb/build.sh` (examples are listed one by one), a page
description in `jb/descriptions.yml`, and the Colab links in `jb/index.md`.

## Tests

```bash
make tests
```

This runs `scripts/check_outputs.py` (fails if any committed notebook in
`soln/` or `examples/` has a code cell that was never executed), then
`pytest --nbmake` on `soln/chap*.ipynb`. nbmake re-executes the notebooks, so
on its own it cannot see missing outputs. That is why the outputs check exists.

The `examples/` notebooks are checked for outputs but **not executed** by the
tests. `examples/temperature` calls the Our World in Data API at run time and
breaks when OWID changes its schema.

`make lint` and `make format` pass `--config pyproject.toml`. Before PR #23
there was no `pyproject.toml`, so they never actually worked; now `make format`
**will reformat every `.py` file** in the repo with black's defaults. Run it
only on purpose.

## Scripts that commit, push or publish

- `build.sh` (top level) commits and pushes `soln/`, then runs both scripts below.
- `nb/build.sh` commits the student notebooks and `ThinkStats.zip`, then
  **pushes**.
- `jb/build.sh` ends in `ghp-import -n -p -f _build/html`, which **publishes
  the live site** with no confirmation.

Don't run them wholesale. Run the steps you need individually and leave
committing, pushing and publishing to Allen. Before a book build,
`rm -rf jb/_build`: Jupyter Book does not remove pages that left the table of
contents, and `ghp-import -f` would publish the stale ones (Task 4).
`jb/_config.yml`'s `extra_footer` holds the page-view counter (Task 6); keep it.

## Data

`data/marriage_nsfg_*.csv.gz` is respondent-level NSFG data, which the NSFG
data use agreement may not allow us to redistribute, and it has a known
age-at-marriage error in cycles 10–12. See PROJECT_BOARD Task 5 before
touching chapter 13's data.

## Task tracking

`PROJECT_BOARD.md` holds numbered tasks. Unlike in ThinkBayes2, it is
**tracked in git** here, so board updates need a commit. Add new tasks at the
end and keep existing numbers stable.
