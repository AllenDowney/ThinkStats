# ThinkStats Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

### Current focus (2026-09-16)

- **Task 1:** Chapter 12 intermittent missing values in Exercise 12.1 — **resolved** (pandas 3 `stack()`).
- **Task 2:** Redesign release tooling (GitHub canonical + SSH rsync to GTP) — not started.
- **Task 3:** Markdown-first authoring workflow (jupytext + documented build) — largely done; needs a status review.
- **Task 4:** Missing figures in published HTML — **resolved**; follow-ups open.
- **Task 5:** Chapter 13 uses a stale NSFG extract that carries a known error — not started.

---

## Task 1: Chapter 12 intermittent missing values error in Exercise 12.1

**Status:** Resolved 2026-09-16 (commit `e5df20d`)

**Resolution:** Not intermittent, and not an environment quirk — a pandas 3 behavior change. The 2024 column of the temperature data has four trailing NaNs. Under pandas 2, `DataFrame.stack()` dropped them silently; pandas 3 keeps them, so `seasonal_decompose` raised. CI installs unpinned from `requirements-dev.txt` and had picked up pandas 3; the local conda env was still on pandas 2.3.3, which is exactly why it reproduced in CI but not by hand.

Fixed by adding the `.dropna()` proposed under "Next steps" below. Both pandas versions now yield the same 296 values, so no published results change. CI had been failing on this every run since at least March 2026 (seven consecutive scheduled builds) with nobody watching; it is green on all six matrix legs as of 2026-09-16.

**Original context:** [GitHub issues](https://github.com/AllenDowney/ThinkStats/issues) report an intermittent failure when running tests with `pytest --nbmake` on the Chapter 12 solution notebook. The notebook runs successfully when executed manually.

**Location:** `soln/chap12.ipynb`, Exercise 12.1, cell with `seasonal_decompose(temp_series, ...)`

**Error:** `ValueError: This function does not handle missing values`

**Observations:**

- Error occurs when running tests with `pytest --nbmake`, but notebook runs successfully when executed manually
- The error happens in the solution cell that calls `seasonal_decompose(temp_series, model="additive", period=12)`
- `temp_series` is created from temperature data by transposing and stacking: `temp_us.loc[:, columns].transpose().stack()`
- When checking locally, both `data/` and `soln/` versions of the CSV file have no missing values
- Fresh download from GitHub also shows no missing values
- Current test run passes without error

**Possible causes:**

- Different pandas versions or parsing behavior between environments

**Next steps:**

- Monitor for recurrence, consider adding `.dropna()` as defensive programming if error persists

---

## Task 2: Redesign release tooling (GitHub canonical + SSH to GTP)

**Status:** Not started

**Context:** ThinkStats v3 releases today are driven by `build.sh`: commit/push solution notebooks, run `nb/build.sh` (student notebooks + `ThinkStats.zip`), then `jb/build.sh` (Jupyter Book HTML → GitHub Pages). That already makes **GitHub** the canonical host for HTML and the zip download. Older GTP-hosted books (including `thinkstats/` and `thinkstats2/` on [greenteapress.com](https://greenteapress.com)) still used bob-local staging paths like `/home/downey/public_html/greenteapress/...` + `sh back`, which cannot run from other machines. ThinkJava2 and ThinkDSP moved to a repeatable pattern: build → commit artifacts to GitHub → optional `rsync` to GTP via `Host gtp` ([ThinkJava2 Task 6](https://github.com/ChrisMayfield/ThinkJava2/blob/master/PROJECT_BOARD.md), [ThinkDSP Task 10–11](https://github.com/AllenDowney/ThinkDSP/blob/master/PROJECT_BOARD.md)).

**ThinkJava2 / ThinkDSP pattern:**

| Layer | Role |
|-------|------|
| **GitHub** | Canonical release artifacts (PDF/EPUB/zips; committed after build) |
| **GitHub Pages** | Canonical HTML for ThinkStats v3 (`allendowney.github.io/ThinkStats/`) |
| **GTP** `greenteapress.com/...` | Stable old URLs; `rsync` via `Host gtp` (`make publish-gtp`) |

**Goal:** Replace any bob-only release assumptions with: build → commit release artifacts to GitHub → optional `rsync` to GTP from any machine with SSH access.

### Scope

- [ ] Add Makefile targets: `distrib` (stage/check artifacts for GitHub commit), `publish-gtp-dry`, `publish-gtp` (rsync to GTP via `Host gtp`; no `--delete`)
- [ ] Decide which artifacts belong in the GitHub release ritual (`ThinkStats.zip`, `ThinkStatsSolutions.zip`, and any future PDF)
- [ ] Refactor or document `build.sh` / `nb/build.sh` / `jb/build.sh` so the publish steps are explicit and repeatable off bob
- [ ] README: GitHub download links as canonical; document optional GTP mirror and `~/.ssh/config` `Host gtp`
- [ ] Inventory live GTP paths for Think Stats (1e/2e static trees vs [WP landing for 3e](https://greenteapress.com/wp/think-stats-3e/))

### Deliverables

1. New distrib docs + Makefile targets (ThinkJava2-style)
2. Release artifacts committed on GitHub; optional GTP mirror documented
3. Board note on canonical vs mirrored paths

### Out of scope (for first pass)

- Rebuilding legacy Hevea HTML on GTP
- GitHub Actions deploy (direct rsync from a trusted laptop is enough for now)

---

## Task 3: Markdown-first authoring workflow

**Status:** Not started

**Context:** Today the canonical source for book content is the `.ipynb` files in `soln/` (and, for supplementary material, `.ipynb` in `examples/`). Edits happen in notebooks; `nb/build.sh` copies from `soln/` and strips solutions into `nb/`; `jb/build.sh` copies from `soln/` (and `examples/`) for the Jupyter Book HTML build. Some `.md` files already exist (partial jupytext exports in `soln/` and `examples/`), and `jupytext` is in `environment-dev.yml`, but there is no single documented pipeline and markdown is not yet authoritative.

**Goal:** Make `.md` the canonical source in `soln/` and `examples/`. Add a workflow file that records how to update the book after a fix. Generated `.ipynb` files become build artifacts, not the place to edit.

### Target pipeline

```text
Edit soln/*.md or examples/*.md
    ↓
jupytext → soln/*.ipynb (and examples/*.ipynb)
    ↓
nbconvert --execute (run notebooks; refresh outputs)
    ↓
prep / strip solutions → nb/*.ipynb (student notebooks, no solutions)
    ↓
(existing downstream: jb/build.sh → GitHub Pages, zip, tests)
```

**Authoring rule:** commit changes to `.md`; treat paired `.ipynb` as generated unless we adopt explicit jupytext pairing metadata.

### Scope

- [ ] Workflow file (e.g. `book-update.md`) documenting the edit → build → test → publish steps
- [ ] One-time or scripted jupytext conversion: ensure every chapter in `soln/` and every example in `examples/` has a canonical `.md` (jupytext front matter, consistent kernelspec)
- [ ] Build script(s) or Makefile targets: `md → ipynb` (jupytext), `execute` (nbconvert), `nb/` generation (adapt `prep_notebooks.py` / replace ad-hoc copy in `nb/build.sh`)
- [ ] jupytext pairing strategy : `.md` canonical, `.ipynb` generated and committed for convenience)
- [ ] Update `build.sh`, `Makefile`, and README so the documented workflow matches what we run
- [ ] Wire `pytest --nbmake` / CI to run against executed notebooks derived from markdown

### Deliverables

1. Workflow file a contributor can follow after fixing a chapter or example
2. Repeatable build commands (`make notebooks` or similar) implementing the pipeline above
3. `soln/` and `examples/` edited as markdown going forward; `nb/` remains solution-free output

### Out of scope (for first pass)

- Migrating `jb/` or `quarto/` to read `.md` directly (they can keep copying/converting from `soln/` for now)
- Retrofitting every historical notebook output into markdown (execute fresh on first migration)
- Replacing Jupyter Book with another HTML builder

---

## Task 4: Missing figures in published HTML

**Status:** Resolved 2026-09-16 (commits `242dff2`..`f561120`); follow-ups below still open

**Context:** A reader reported that the HTML built from `jb/` was missing figures. Chapters 8, 9, and 13 had been committed straight from jupytext with every cell unexecuted — `execution_count: null`, zero stored outputs — so their published pages contained no figures at all.

**Root cause:** Commit `4b32fa8` ("Updating notebooks", 2026-02-13) stripped 1190, 908, and 1288 lines from those three notebooks. That was the session that fixed issues #13, #15, and #16 — which are chapters 8, 9, and 13 exactly. The markdown edits landed; `jupytext --to ipynb` discarded the outputs; the `nbconvert --execute` step never completed. The figures stayed missing for seven months.

**Why nothing caught it:** `pytest --nbmake` *executes* notebooks before checking them, so stored outputs are irrelevant to the test suite. CI was green on all three chapters the entire time. Nothing in the repo verifies that a committed notebook contains outputs.

### What was fixed

| Item | Before | After |
|------|--------|-------|
| `soln/chap08` | 0 figures | 6 |
| `soln/chap09` | 0 figures | 8 |
| `soln/chap13` | 0 figures | 18 |
| `examples/variability` | 1 of 5 figures; 20 of 40 cells unexecuted | 5 |
| `examples/temperature` | broken — `KeyError: 'Day'` | 2 |
| `soln/chap12` | failing CI since ~March 2026 | passing (Task 1) |

Also in this pass:

- **OWID schema change:** `examples/temperature` reads the Our World in Data API at build time. OWID renamed `Day` → `month`, lowercased `Entity`/`Code`, and dropped the annual-average column. Same monthly data, so no results changed.
- **CI matrix:** `actions/checkout` and `setup-python` v2 → v7 (v2 ran on Node 20, already deprecated and being forced onto Node 24); added Python 3.13; re-enabled macOS (the "cannot pip install pytables" comment was obsolete — `tables` 3.11.1 ships `cp311-abi3` macOS wheels); `fail-fast: false`; pip caching.
- **Issue #17:** closed 2026-02-13 with "I'll reword", but the rewording was never applied. Chapter 1 conflated stratified sampling with oversampling. Dropped "stratified" from the book, fixed the `oversampled` and `cross-sectional study` glossary entries, and updated six uses in chapter 13. [Commented on the issue](https://github.com/AllenDowney/ThinkStats/issues/17#issuecomment-5702912223).
- **Stale published pages:** `jb/_build/` retained `chap15`–`chap19` HTML from an earlier edition, and `ghp-import -f` was pushing them live on every publish. Fixed by wiping `_build/` before building.

### Follow-ups (open)

- [ ] **Add an outputs check to `make tests`.** A notebook committed with `execution_count: null` in every cell should fail the build. This is the single check that would have caught the original bug the same day.
- [ ] **Test `examples/` too.** `make tests` only runs `soln/chap*.ipynb`, which is why the OWID break went unnoticed. `temperature` needs network access, so it may need its own marker.
- [ ] **Write a repo `CLAUDE.md`.** ThinkStats has none; [ThinkBayes2's](https://github.com/AllenDowney/ThinkBayes2/blob/master/CLAUDE.md) covers three hazards that bit us here: `jupytext --to ipynb` destroys outputs before `nbconvert` can fail (use `--update` for notebooks that cannot be re-executed), `jb/build.sh` ends in `ghp-import -p` and publishes without confirmation, and adding a chapter means touching `_toc.yml`, the `cp` list in `build.sh`, and the Colab links together.
- [ ] **Decide on `examples/temperature`'s live API dependency.** It will break again on the next OWID schema change. Chapter 12 avoids this by reading a committed CSV snapshot, but the example is explicitly *about* calling the API.
- [ ] **Three cosmetic `execution_count: null` cells remain** in `soln/chap03`, `soln/chap12`, and `examples/marathon_splits` — all assignment- or comment-only, no outputs affected.


---

## Task 5: Chapter 13 uses a stale NSFG extract that carries a known error

**Status:** Not started, 2026-09-20. The data fix is straightforward; the analysis question is not.

**Context:** [MarriageNSFG](https://github.com/AllenDowney/MarriageNSFG) — the repository that harmonizes NSFG across survey cycles — was reorganized and re-estimated this week. Chapter 13 reads its output:

```python
download("https://github.com/AllenDowney/ThinkStats/raw/v3/data/" + filename)
...
read_csv("marriage_nsfg_female.csv.gz")
```

`data/marriage_nsfg_female.csv.gz` in this repository is dated **2024-09-29**. Two things are wrong with it.

**It is missing a survey cycle.** It covers cycles 3–11, so it stops at 2017–2019. The 2022–2023 cycle has since been released and harmonized.

**It carries an error that reached published figures.** From cycle 10 the NSFG stopped publishing century-month dates of birth, and MarriageNSFG reconstructed them with a sign error that made everyone half a year younger than the reported age allows. The effect is a full year of bias in age at first marriage, in cycles 10–12 only. It shows up as an artificial dip at the cycle 9/10 boundary:

| mean `agemarry`, ever-married women | cycle 9 | cycle 10 | cycle 11 | cycle 12 |
|---|---|---|---|---|
| the extract this repo serves (2024-09) | 24.271 | **23.549** | 24.100 | — |
| corrected, with 2022–2023 added | 24.271 | 24.549 | 25.100 | 26.205 |

The dip is entirely an artifact. Corrected, the series rises monotonically across all ten cycles. Any figure in Chapter 13 that shows recent birth cohorts is affected.

**The good news.** Chapter 13 is already on `empiricaldist` (`Surv`, `Hazard`) and `lifelines`, so none of the library retirement that happened in MarriageNSFG applies here. This is a data swap, not a rewrite.

### The harder question

MarriageNSFG now treats age at first marriage as *interval-censored* for cycles 10–12, because both endpoints are published as years rather than months. That has two consequences Chapter 13 may want to reflect:

- Kaplan-Meier tails degenerate when the risk set empties, and the coarse dates make it worse by piling every censored respondent onto a dozen integer ages. The youngest cohort's curve can jump to 100% on the strength of one respondent.
- The fix — imputing a time within each respondent's interval, repeatedly, and reporting where the draws agree — changes recent-cohort estimates by several points.

Whether a textbook chapter should carry that is a judgment call. The plain Kaplan-Meier is the better teaching tool; the interval-censored version is the better estimate. It may be an exercise rather than a section.

### Also worth checking

This repository serves `marriage_nsfg_female.csv.gz` publicly from GitHub. That file is respondent-level NSFG data. MarriageNSFG stopped distributing exactly this file for that reason, and purged it from its history, on the grounds that NSFG's data user agreement does not permit redistribution. The same reasoning presumably applies here.

### Scope

- [ ] Regenerate the extract from MarriageNSFG and replace `data/marriage_nsfg_*.csv.gz`
- [ ] Re-run Chapter 13 and see which figures move
- [ ] Decide whether to extend the chapter to the 2022–2023 cycle or hold at 2017–2019
- [ ] Decide whether interval censoring belongs in the chapter, an exercise, or nowhere
- [ ] Settle whether this repository should be serving NSFG microdata at all
