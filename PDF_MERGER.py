import sys
import os
from pypdf import PdfMerger


def normalize_path(path):
    """
    Ensure the file has a .pdf extension.
    """
    if not path.lower().endswith(".pdf"):
        path += ".pdf"
    return path


def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_pdfs.py <file1> <file2>")
        sys.exit(1)

    file1 = normalize_path(sys.argv[1])
    file2 = normalize_path(sys.argv[2])

    if not os.path.exists(file1):
        print(f"Error: File not found -> {file1}")
        sys.exit(1)

    if not os.path.exists(file2):
        print(f"Error: File not found -> {file2}")
        sys.exit(1)

    output_file = "merged_output.pdf"

    merger = PdfMerger()
    merger.append(file1)
    merger.append(file2)

    with open(output_file, "wb") as f:
        merger.write(f)

    merger.close()

    print(f"Successfully merged into {output_file}")


if __name__ == "__main__":
    main()