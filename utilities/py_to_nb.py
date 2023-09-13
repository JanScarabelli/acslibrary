import json

from argparse import ArgumentParser
from pathlib import Path


def python_to_notebook(input_file: Path, output_file: Path) -> None:
    """Convert a python script generated from a notebook using VSCode back to notebook.

    Using the fact that VSCode adds `# %%` at the beginning of each cell when it converts it to
    script, we can use some JSON templates of the different cell types and put the notebook back
    together. As long as the script has these `# %%`, this function will be able to convert it.
    It is suggested to use the scripts as snapshots of notebooks and then convert them back using
    this utility.

    Parameters
    ----------
    input_file : Path
        Path to the file to be converted
    output_file : Path
        Path in which to save the converted notebook

    Returns
    -------
    None

    Notes
    -----
    Suggested alias:
        ```,py2nb () {
            current_path=`pwd`

            if [ $# -lt 2 ]
            then
                file=$current_path/$1
                input_file=$file.py
                output_file=$file.ipynb
            else
                input_file=$current_path/$1.py
                output_file=$current_path/$2.ipynb
            fi

            python <path_to_script> -if $input_file -of $output_file
        }```

    If using this alias, you can pass it just the name of the script (without extension) or you can
    pass it both the input and output names (both without extension). You can also add subfolders if
    it's required, but you can't put a folder in another branch of the folder structure, example:

    `,py2nb scripts/script1 notebooks/notebook1  # This is correct`

    `,py2nb ../scripts/script1 notebooks/notebook1  # This will fail`
    """
    # Check if input file exists and read
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            data = [line.rstrip("\n") for line in infile]
    except FileNotFoundError:
        print("Input file not found. Specify a valid input file.")
        quit()

    templates_path = Path(__file__).resolve().parent / 'notebook_templates'
    # Read JSON files for .ipynb template
    with open(templates_path / 'cell_code.json', encoding="utf-8") as file:
        CODE_TEMPLATE = json.load(file)
    with open(templates_path / 'cell_markdown.json', encoding="utf-8") as file:
        MARKDOWN_TEMPLATE = json.load(file)
    with open(templates_path / 'metadata.json', encoding="utf-8") as file:
        MISC_TEMPLATE = json.load(file)

    # Initialise variables
    final_json = {}
    cells = []
    cell_start_number = []

    # Read source code line by line
    for i, line in enumerate(data):
        if line.startswith('# %%'):
            cell_start_number.append(i)

    for i, start_line in enumerate(cell_start_number):
        if i >= len(cell_start_number) - 1:
            cell_source = data[start_line + 1:]
        else:
            cell_source = data[start_line + 1:cell_start_number[i+1] - 1]

        # Add '\n' too all the lines except the last one
        cell_source = [
            line + '\n' if i < len(cell_source) - 1 else line for i, line in enumerate(cell_source)
        ]

        if 'markdown' in data[start_line]:
            # Remove the '# ' from the beginning of markdown cell's lines
            cell_source = [line[2:] for line in cell_source]

            MARKDOWN_TEMPLATE['source'] = cell_source
            cell_json = dict(MARKDOWN_TEMPLATE)

        else:
            CODE_TEMPLATE['source'] = cell_source
            cell_json = dict(CODE_TEMPLATE)

        cells.append(cell_json)

    # Finalise the contents of notebook
    final_json["cells"] = cells
    final_json.update(MISC_TEMPLATE)

    # Write JSON to output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(final_json, outfile, indent=1, ensure_ascii=False)
        print(f'Notebook {output_file} written.')


if __name__ == '__main__':
    arg_parser = ArgumentParser()

    arg_parser.add_argument('-if', '--input-file', help='script to convert')
    arg_parser.add_argument('-of', '--output-file', help='output notebook')

    args = arg_parser.parse_args()

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)

    python_to_notebook(input_file, output_file)

