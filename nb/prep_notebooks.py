import nbformat as nbf
from glob import glob

# Collect a list of all notebooks in the content folder
filenames = glob("chap*.ipynb")

text = '# Solution'
replacement = '# Solution goes here'

# Search through each notebook
for filename in sorted(filenames):
    print('Removing solutions from', filename)
    ntbk = nbf.read(filename, nbf.NO_CONVERT)

    # if the second element of ntbk.cells loads nb_black, remove it
    second = ntbk.cells[1]
    if second.source.startswith(r'%load_ext nb_black'):
        ntbk.cells.pop(1)

    for cell in ntbk.cells:
        # remove tags
        if 'tags' in cell['metadata']:
            tags = cell['metadata']['tags']
            cell['metadata']['tags'] = []
        else:
            tags = []

        # remove output
        if 'outputs' in cell:
            cell['outputs'] = []

        # remove solutions
        if cell['source'].startswith(text):
            cell['source'] = replacement

        # remove solutions
        if 'solution' in tags:
            cell['source'] = replacement

        # add reference label
        for tag in tags:
            if tag.startswith('chapter') or tag.startswith('section'):
                print(tag)
                label = f'({tag})=\n'
                cell['source'] = label + cell['source']

    nbf.write(ntbk, filename)
