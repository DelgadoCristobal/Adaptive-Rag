import datetime
from pathlib import Path
import yaml


class DOCXParser:
    """Parser for DOCX documents using mammoth for semantic markdown conversion."""

    def parse(self, file_path: Path) -> str:
        """Parse DOCX file into Obsidian-compatible Markdown with YAML frontmatter."""
        file_path = Path(file_path)

        try:
            import mammoth
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_markdown(docx_file)
                markdown_content = result.value
        except Exception as e:
            markdown_content = f"# {file_path.stem}\n\n*Error parsing DOCX: {e}*"

        metadata = {
            "title": file_path.stem,
            "source": file_path.name,
            "file_type": "docx",
            "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tags": ["document", "docx"]
        }

        frontmatter = f"---\n{yaml.dump(metadata, sort_keys=False)}---\n\n"
        return frontmatter + markdown_content
