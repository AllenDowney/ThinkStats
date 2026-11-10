# ThinkStats

This repository contains notebooks, data, and supplementary material for the third edition of *Think Stats*.

[Click here to read the book online and run the notebooks](https://allendowney.github.io/ThinkStats/).

## Downloadable Files

- **[ThinkStats.zip](https://github.com/AllenDowney/ThinkStats/raw/refs/heads/v3/ThinkStats.zip)**: Zip file of notebooks without solutions (download this if you just want to run the notebooks without cloning the repository)  
- **[ThinkStatsSolutions.zip](https://github.com/AllenDowney/ThinkStats/raw/refs/heads/v3/ThinkStatsSolutions.zip)**: Zip file of notebooks with solutions  


## Directories

- **data**: Data files used in the book  
- **examples**: Notebooks with additional examples  
- **figs**: Figures other than the ones generated in the notebooks  
- **jb**: Directory where I build the HTML version of the book with Jupyter Book  
- **nb**: Notebooks without solutions  
- **soln**: Notebooks with solutions (these are the source files)  
- **tutorial**: Notebooks for a tutorial on time series analysis  


### Note for users of earlier editions

Older versions of this repository used Git LFS to store data files. 
As of June 2025, all data files are included directly in Git to simplify access.


## Setup

- **Makefile**: Instructions for creating a Conda environment for the book  
- **requirements.txt**: Required packages to run the notebooks  
- **requirements-dev.txt**: Additional packages used for developing the notebooks
- **pyproject.toml**: The same packages for [uv](https://docs.astral.sh/uv/): run `uv sync`, then `uv run jupyter lab` (`uv sync --no-dev` skips the development tools)
