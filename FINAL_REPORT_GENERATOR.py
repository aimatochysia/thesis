from pypdf import PdfReader, PdfWriter
import os

# ===== CONFIG =====
input_pdf_path = "./docs/Final-Skripsi-revisi-1.pdf"   # change to your file
output_folder = "FINAL_REPORT"

sections = [
    ("cover", 1, 2),
    ("bab1", 11, 15),
    ("bab2", 17, 31),
    ("bab3", 32, 45),
    ("bab4", 46, 60),
    ("bab5", 61, 62),
    ("referensi", 63, 66),
    ("lampiran_surat_survey", 67, 75),
    ("abstrak", 4, 4),
    ("daftar_isi", 6, 10),
]

# ==================

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

reader = PdfReader(input_pdf_path)

for title, start_page, end_page in sections:
    writer = PdfWriter()

    # PDF pages are 0-indexed internally
    for page_num in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_num])

    output_path = os.path.join(output_folder, f"{title}.pdf")

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"Created: {output_path}")

print("Done splitting PDF.")
