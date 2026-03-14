import re
from typing import List, Dict


def chunk_markdown(markdown_text: str) -> List[Dict]:
    lines = markdown_text.splitlines()
    chunks = []

    current_section_title = "Untitled"
    current_content = []
    chunk_index = 0

    for line in lines:
        stripped_line = line.strip()

        if re.match(r"^##\s+", stripped_line) or re.match(r"^###\s+", stripped_line):
            if current_content:
                chunks.append(
                    {
                        "section_title": current_section_title,
                        "content": "\n".join(current_content).strip(),
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            current_section_title = stripped_line.lstrip("#").strip()
            current_content = [stripped_line]
        else:
            if stripped_line:
                current_content.append(line)

    if current_content:
        chunks.append(
            {
                "section_title": current_section_title,
                "content": "\n".join(current_content).strip(),
                "chunk_index": chunk_index,
            }
        )

    return chunks