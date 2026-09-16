PROJECT_NAME = ThinkStats
PYTHON_VERSION = 3.12

.PHONY: help create_environment create_environment_dev delete_environment \
	update_environment update_environment_dev clean lint format tests \
	update-chapter md-from-ipynb publish-html

CHAPTER ?= chap07

help:
	@echo "Available targets:"
	@echo "  create_environment    - Create conda environment from environment.yml"
	@echo "  create_environment_dev - Create dev environment (base + dev packages)"
	@echo "  delete_environment    - Remove conda environment"
	@echo "  update_environment    - Update environment from environment.yml (with --prune)"
	@echo "  update_environment_dev - Update dev environment"
	@echo "  clean                 - Remove temporary files and caches"
	@echo "  lint                  - Run linters (flake8, black --check)"
	@echo "  format                - Format code with black"
	@echo "  tests                 - Run tests with pytest"
	@echo "  update-chapter        - Rebuild one chapter from markdown (CHAPTER=chap07)"
	@echo "  md-from-ipynb         - Export missing soln/*.md from existing notebooks"
	@echo "  publish-html          - Build Jupyter Book and push to gh-pages"

create_environment:
	mamba env create -f environment.yml
	@echo ""
	@echo ">>> Environment created successfully!"
	@echo ">>> Activate with: conda activate $(PROJECT_NAME)"

create_environment_dev: create_environment
	mamba env update -f environment-dev.yml --name $(PROJECT_NAME)
	@echo ""
	@echo ">>> Dev environment created successfully!"
	@echo ">>> Activate with: conda activate $(PROJECT_NAME)"

delete_environment:
	mamba env remove --name $(PROJECT_NAME)
	@echo ">>> Environment $(PROJECT_NAME) removed"

update_environment:
	mamba env update -f environment.yml --prune
	@echo ">>> Environment updated successfully!"

update_environment_dev: update_environment
	mamba env update -f environment-dev.yml --name $(PROJECT_NAME) --prune
	@echo ">>> Dev environment updated successfully!"

clean:
	@echo "Cleaning temporary files and caches..."
	-find . -type f -name "*.py[co]" -delete
	-find . -type d -name "__pycache__" -delete
	-find . -type d -name ".pytest_cache" -delete
	-find . -type d -name ".ipynb_checkpoints" -delete
	-rm -rf .ruff_cache/
	-rm -rf .mypy_cache/
	-rm -rf build/
	-rm -rf dist/
	-rm -rf *.egg-info
	@echo ">>> Cleanup complete!"

lint:
	@echo "Running linters..."
	flake8 . --config pyproject.toml || true
	black --check --config pyproject.toml . || true
	@echo ">>> Linting complete!"

format:
	@echo "Formatting code..."
	black --config pyproject.toml .
	@echo ">>> Formatting complete!"

tests:
	@echo "Running tests..."
	cd soln; pytest --nbmake chap*.ipynb
	@echo ">>> Tests complete!"

update-chapter:
	@echo "Updating $(CHAPTER) from markdown..."
	python scripts/book_pipeline.py $(CHAPTER)
	@echo ">>> $(CHAPTER) updated (soln + nb)"

md-from-ipynb:
	@for f in soln/chap*.ipynb; do \
		md="$${f%.ipynb}.md"; \
		if [ ! -f "$$md" ]; then \
			echo "Exporting $$md"; \
			jupytext --to md "$$f"; \
		fi; \
	done
	@echo ">>> md-from-ipynb complete"

publish-html:
	@echo "Building Jupyter Book and pushing to gh-pages..."
	cd jb && bash build.sh
	@echo ">>> HTML published (see origin/gh-pages)"
