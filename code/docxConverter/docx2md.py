import pypandoc
from pathlib import Path

def docx_to_md(input_docx, output_md):
    pypandoc.convert_file(
        input_docx,
        'md',
        outputfile=output_md
    )

if __name__ == "__main__":
    input_file = "input.docx"
    output_file = "output.md"

    if not Path(input_file).exists():
        raise FileNotFoundError(f"{input_file} not found")

    docx_to_md(input_file, output_file)
    print(f"Converted {input_file} → {output_file}")
