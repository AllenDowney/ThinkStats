# Requires dev environment: make create_environment_dev
# (jupyter-book<2 and ghp-import — jb/_config.yml and _toc.yml target JB 1.x)

# copy the notebooks
cp ../soln/chap[01][0-9]*.ipynb .
cp ../examples/binom_skeet.ipynb .
cp ../examples/ripoff_etf.ipynb .
cp ../examples/fourier.ipynb .
cp ../examples/temperature.ipynb .
cp ../examples/variability.ipynb .
cp ../examples/moneyline.ipynb .

# add tags to hide the solutions
python prep_notebooks.py

# build the HTML version
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
