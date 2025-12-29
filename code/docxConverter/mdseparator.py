import re
from pathlib import Path

def split_md_sections(md_text):
    """
    Splits markdown text into sections starting with **BAB x
    where x is a number (1, 2, 3, ...)
    """
    # Match lines starting with **BAB <number>
    pattern = re.compile(
        r"^\*\*BAB\s+(\d+)(.*?)$",
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(pattern.finditer(md_text))
    sections = []

    for i, match in enumerate(matches):
        bab_number = match.group(1)
        bab_title_extra = match.group(2).strip()
        full_title = f"BAB {bab_number}{bab_title_extra}"

        start = match.end()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        content = md_text[start:end].strip()

        sections.append({
            "index": int(bab_number),
            "title": full_title,
            "content": content
        })

    return sections


def normalize_title(title):
    title = title.upper().strip()
    title = re.sub(r"\s+", "_", title)
    title = re.sub(r"[^A-Z0-9_]", "", title)
    return title


def save_sections(sections, output_dir="sections"):
    Path(output_dir).mkdir(exist_ok=True)

    for sec in sections:
        safe_title = normalize_title(sec["title"])
        filename = f"{sec['index']}_{safe_title}.md"
        path = Path(output_dir) / filename

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"**{sec['title']}**\n\n")
            f.write(sec["content"])

        print(f"Saved: {path}")


# -------- USAGE --------
with open("thesis_base.md", "r", encoding="utf-8") as f:
    md_text = f.read()

sections = split_md_sections(md_text)
save_sections(sections)
