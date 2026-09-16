# Book update workflow

Canonical source for chapter and example content is **Markdown** (`.md`), not Jupyter notebooks. Edit the `.md` files in `soln/` and `examples/`, then run the build pipeline to regenerate executed notebooks and student copies.

## Prerequisites

```bash
conda activate ThinkStats
# Dev tools (jupytext, nbmake, …) from environment-dev.yml
make create_environment_dev   # first time only
```

## When you fix a typo or revise a chapter

Example: typo in Chapter 7, section 7.5 Rank Correlation.

### 1. Edit the markdown source

```bash
$EDITOR soln/chap07.md
```

Commit the `.md` change. Do **not** hand-edit `soln/chap07.ipynb` or `nb/chap07.ipynb`.

### 2. Run the pipeline for that chapter

```bash
make update-chapter CHAPTER=chap07
```

This runs `scripts/book_pipeline.py`, which:

1. **jupytext** — `soln/chap07.md` → `soln/chap07.ipynb`
2. **nbconvert** — executes the notebook in place (refreshes outputs)
3. **strip solutions** — copies to `nb/chap07.ipynb` and removes solution cells

Equivalent direct invocation:

```bash
python scripts/book_pipeline.py chap07
```

### 3. Test

```bash
cd soln && pytest --nbmake chap07.ipynb
```

Or test all chapters:

```bash
make tests
```

### 4. Publish HTML (required for reader-visible changes)

Online readers see [allendowney.github.io/ThinkStats](https://allendowney.github.io/ThinkStats/), built from `jb/` and pushed to the **`gh-pages`** branch. After updating a chapter or a published example, rebuild and push:

```bash
make publish-html
```

This runs `jb/build.sh`, which:

1. Copies executed notebooks from `soln/` and `examples/` into `jb/`
2. Strips solution cells (`jb/prep_notebooks.py`)
3. Builds HTML with Jupyter Book (`jb build`)
4. Pushes `_build/html` to **`gh-pages`** via `ghp-import`

Requires dev tools from `environment-dev.yml` / `requirements-dev.txt` (`jupyter-book<2`, `ghp-import`). The `jb/` config (`_config.yml`, `_toc.yml` with `format: jb-book`) targets **Jupyter Book 1.x**, not 2.x. Install with `make create_environment_dev`.

For a chapter-only edit, the usual sequence is:

```bash
make update-chapter CHAPTER=chap07
make publish-html
git push origin v3          # source changes on v3
git push origin gh-pages    # if ghp-import did not push automatically
```

(`ghp-import -p` pushes `gh-pages` by default; verify with `git log origin/gh-pages -1`.)

### 5. Full release (optional)

For a full book release (all chapters, student zip, etc.):

```bash
./build.sh          # soln → nb zip → Jupyter Book → gh-pages
```

Future work ([Task 2](../PROJECT_BOARD.md)): wire `build.sh` to the markdown-first pipeline and add PDF/GTP release tooling.

## What we publish today

| Format | Published from this repo? | Where readers get it |
|--------|---------------------------|----------------------|
| **HTML** | Yes | [allendowney.github.io/ThinkStats](https://allendowney.github.io/ThinkStats/) (`gh-pages`, built from `jb/`) |
| **Notebook zips** | Yes | `ThinkStats.zip`, `ThinkStatsSolutions.zip` on the `v3` branch ([README](../README.md)) |
| **PDF** | **No** | Print/ebook via [O'Reilly / retailers](https://greenteapress.com/wp/think-stats-3e/); not built or hosted from this repo |

There is no `make pdf` or committed `.pdf` in ThinkStats v3. The `think-stats-3e/` tree (gitignored here) is a separate production project for the print layout. [Task 2](../PROJECT_BOARD.md) may add GitHub/GTP PDF mirroring later, following the ThinkJava2/ThinkDSP pattern.

## Examples

Examples in `examples/` follow the same pattern:

```bash
python scripts/book_pipeline.py example:binom_skeet
```

## Generating markdown from existing notebooks (one-time migration)

If a chapter has `.ipynb` but no `.md` yet:

```bash
cd soln
jupytext --to md chap07.ipynb
```

After that, treat the `.md` file as canonical.

## File roles

| Path | Role |
|------|------|
| `soln/*.md` | **Canonical** chapter source (edit here) |
| `soln/*.ipynb` | Generated + executed solution notebooks |
| `nb/*.ipynb` | Generated student notebooks (solutions stripped) |
| `examples/*.md` | **Canonical** example source |
| `examples/*.ipynb` | Generated + executed example notebooks |

## Makefile targets

| Target | Purpose |
|--------|---------|
| `make update-chapter CHAPTER=chap07` | Rebuild one chapter from markdown |
| `make publish-html` | Build Jupyter Book and push to `gh-pages` |
| `make md-from-ipynb` | One-time jupytext export for chapters missing `.md` |
| `make tests` | Run nbmake on all solution notebooks |
