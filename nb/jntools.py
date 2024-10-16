import ast
import astor
from pathlib import Path
from typing import List
import nbformat as nbf
import typer
import shelve
app = typer.Typer()



@app.command()
def replace_pattern(filenames: List[Path]):
    """Replace patterns in the given files

    Args:
        filenames (List[Path]): List of files to process
    """
    for filename in filenames:
        if filename.suffix == '.ipynb':
            replace_pattern_in_notebook(filename)
        elif filename.suffix == '.py':
            replace_pattern_in_module(filename)

def replace_pattern_in_notebook(filename):
    """Replace patterns in a notebook
    
    Args:
        filename (Path): The notebook to process
    """
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell['cell_type'] == 'code':
            source = cell['source']
            cell.source = replace_pattern_in_source(source)
        elif cell['cell_type'] == 'markdown':
            source = cell['source']
            cell.source = replace_pattern_in_markdown(source)

    nbf.write(notebook, filename)

def replace_pattern_in_module(filename):
    """Replace patterns in a module
    
    Args:
        filename (Path): The module to process
    """
    source = filename.read_text()
    source = replace_pattern_in_source(source)
    filename.write_text(source)

def replace_pattern_in_source(source):
    """Replace patterns in a source string

    Args:
        source (str): The source to process

    Returns:
        str: The modified source
    """
    # Don't replace patterns in Python code
    return source

def replace_pattern_in_markdown(source):
    """Replace patterns in a markdown string
    
    Args:
        source (str): The markdown string to process

    Returns:
        str: The modified markdown string
    """
    pattern = r'(?<!`)(Series|DataFrame|Index)(?!`)'
    replacement = r'`\1`'
    source = re.sub(pattern, replacement, source)
    
    pattern = r'(?<!`)(Hist|Pmf|Cdf|Surv|Hazard)(?!`)'
    replacement = r'`\1`'
    source = re.sub(pattern, replacement, source)
    
    pattern = r'(?<!`)(Normal|NormalPdf|EstimatedPdf|HypothesisTest)(?!`)'
    replacement = r'`\1`'
    source = re.sub(pattern, replacement, source)
    
    pattern = r'pandas'
    replacement = r'Pandas'
    source = re.sub(pattern, replacement, source)

    return source

def put_each_sentence_on_a_new_line(source):
    """Put each sentence on a new line
    
    Args:
        source (str): The source to process

    Returns:
        str: The modified source
    """
    # replace a single newline with a space    
    pattern = r'(?<!\n)\n(?!\n)'
    replacement = r' '
    source = re.sub(pattern, replacement, source)
    
    # replace four spaces with one
    pattern = r'     '
    replacement = r' '
    source = re.sub(pattern, replacement, source)
    
    # split between sentences
    pattern = r'(?<=[^\s.]{3})\.[ ]+'
    replacement = r'.\n'
    source = re.sub(pattern, replacement, source)
    
    return source


@app.command()
def replace_functions(filenames: List[Path]):
    with shelve.open('function_names') as db:
        function_names = db['function_names']

    for filename in filenames:
        if filename.suffix == '.ipynb':
            replace_functions_in_notebook(filename, function_names)
        elif filename.suffix == '.py':
            replace_functions_in_module(filename, function_names)

def replace_functions_in_notebook(filename, function_names):
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell['cell_type'] == 'code':
            source = cell['source']
            modified_source = replace_functions_in_source(source, function_names)
            cell.source = modified_source.rstrip()
    nbf.write(notebook, filename)

def replace_functions_in_module(filename, function_names):
    source = filename.read_text()
    source = replace_functions_in_source(source, function_names)
    filename.write_text(source)

def replace_functions_in_source(source, function_names):
    class_names = ['Cdf', 'Pmf', 'Hist', 'Pdf', 
                   'HazardFunction', 'SurvivalFunction', 'HypothesisTest']

    class FunctionNameReplacer(ast.NodeTransformer):

        def visit_FunctionDef(self, node):
            if node.name in function_names:
                node.name = camel_to_snake(node.name)
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                function_name = node.func.attr
                if module_name == 'thinkstats2' and function_name in class_names:
                    return node
                if function_name in function_names:
                    node.func.attr = camel_to_snake(function_name)
            elif isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    node.func.id = camel_to_snake(node.func.id)
            self.generic_visit(node)
            return node
    try:
        tree = ast.parse(source)
        replacer = FunctionNameReplacer()
        tree = replacer.visit(tree)
        return astor.to_source(tree)
    except SyntaxError as e:
        print(f'Syntax error in code:\n{source}\nError: {e}')
        return source

@app.command()
def get_functions(filenames: List[Path]):
    function_names = set()
    for filename in filenames:
        if filename.suffix == '.ipynb':
            print(f'Getting functions from {filename}')
            notebook = nbf.read(filename, nbf.NO_CONVERT)
            for cell in notebook.cells:
                if cell['cell_type'] == 'code':
                    source = ''.join(cell['source'])
                    extract_function_names(source, function_names)
        elif filename.suffix == '.py':
            print(f'Getting functions from {filename}')
            source = filename.read_text()
            extract_function_names(source, function_names)
    with shelve.open('function_names') as db:
        db['function_names'] = function_names
    return function_names

def to_camelcase(s):
    parts = s.split('_')
    return parts[0] + ''.join((word.capitalize() for word in parts[1:]))

def is_camelcase(s):
    return s[0].isupper() and s == to_camelcase(s)
import re

def camel_to_snake(s):
    return re.sub('(?<!^)(?=[A-Z])', '_', s).lower()

def extract_function_names(source, function_names):

    class FunctionNameExtractor(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            if is_camelcase(node.name):
                function_names.add(node.name)
            self.generic_visit(node)
    try:
        tree = ast.parse(source)
        extractor = FunctionNameExtractor()
        extractor.visit(tree)
    except SyntaxError as e:
        print(f'Syntax error in code:\n{source}\nError: {e}')
    for name in list(function_names):
        if is_camelcase(name):
            print(name, camel_to_snake(name))

@app.command()
def add_header(header_filename: Path, filenames: List[Path]):
    """
    Add a header to the beginning of every notebook
    """
    typer.echo(f'Header file: {header_filename}')
    typer.echo(f'Target files: {filenames}')
    header = nbf.read(header_filename, nbf.NO_CONVERT)
    for filename in filenames:
        typer.echo(f'Adding {header_filename} to {filename}')
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = header.cells + notebook.cells
        nbf.write(notebook, filename)

@app.command()
def add_footer(footer_filename: Path, filenames: List[Path]):
    """
    Add a footer to the end of every notebook
    """
    typer.echo(f'Footer file: {footer_filename}')
    typer.echo(f'Target files: {filenames}')
    footer = nbf.read(footer_filename, nbf.NO_CONVERT)
    for filename in filenames:
        typer.echo(f'Adding {footer_filename} to {filename}')
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells + footer.cells
        nbf.write(notebook, filename)

@app.command()
def remove_header(n: int, filenames: List[Path]):
    """Remove n cells from the beginning of every notebook
    """
    if n == 0:
        print('Removing 0 cells. Nothing to do.')
        return
    for filename in filenames:
        print('Removing first', n, 'cells from', filename)
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells[n:]
        nbf.write(notebook, filename)

@app.command()
def remove_footer(n: int, filenames: List[Path]):
    """Remove n cells from the end of every notebook
    """
    if n == 0:
        print('Removing 0 cells. Nothing to do.')
        return
    for filename in filenames:
        print('Removing last', n, 'cells from', filename)
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells[:-n]
        nbf.write(notebook, filename)

@app.command()
def prepare_latex(filenames: List[Path]):
    for filename in filenames:
        print(f'Preparing {filename} for LaTeX')
        process_notebook(filename, process_cell_latex)

def process_notebook(filename, cell_func):
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        cell_func(cell)
    nbf.write(notebook, filename)

def process_cell_latex(cell):
    tags = cell['metadata'].get('tags', [])
    if cell['cell_type'] == 'code':
        source = cell['source']
        if source.startswith('# Solution'):
            tag = 'hide-cell'
            if tag not in tags:
                tags.append(tag)
        if source.startswith('%%expect'):
            t = source.split('\n')[1:]
            cell['source'] = '\n'.join(t)
    for tag in tags:
        if tag.startswith('chapter') or tag.startswith('section'):
            print(tag)
            label = f'({tag})=\n'
            cell['source'] = label + cell['source']
    if len(tags) > 0:
        cell['metadata']['tags'] = tags

@app.command()
def add_tags(filenames: List[Path]):
    for filename in filenames:
        print(f'Adding tags to {filename}')
        add_tags_to_notebook(filename)

def add_tags_to_notebook(filename):
    tags = ['remove-print', 'hide-cell']

    notebook = nbf.read(filename, nbf.NO_CONVERT)

    cell = notebook.cells[1]
    if cell['source'].startswith('[Click'):
        add_tags_to_cell(cell, ['remove-print'])

    cell = notebook.cells[2]
    if cell['source'].startswith('%load'):
        add_tags_to_cell(cell, tags)

    cell = notebook.cells[3]
    if cell['source'].startswith('from'):
        add_tags_to_cell(cell, tags)

    cell = notebook.cells[4]
    if cell['source'].startswith('try'):
        add_tags_to_cell(cell, tags)

    cell = notebook.cells[5]
    if cell['source'].startswith('import'):
        add_tags_to_cell(cell, tags)
     
    nbf.write(notebook, filename)

def add_tags_to_cell(cell, tags):
    """Add tags to a cell
    
    Args:
        cell (dict): A Jupyter cell
        tags (list): A list of tags to add
    """
    cell_tags = cell['metadata'].get('tags', [])

    #if hide is in the list of tags, remove it
    if 'hide' in cell_tags:
        cell_tags.remove('hide')

    # Add tags only if they are not already present
    for tag in tags:
        if tag not in cell_tags:
            cell_tags.append(tag)
    cell['metadata']['tags'] = cell_tags




if __name__ == '__main__':
    app()

