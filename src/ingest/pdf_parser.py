import datetime
from pathlib import Path
import yaml


class PDFParser:
    """Parser for PDF documents using pymupdf4llm with layout awareness."""

    def parse(self, file_path: Path) -> str:
        """Parse PDF file into Obsidian-compatible Markdown with YAML frontmatter."""
        file_path = Path(file_path)
        
        try:
            import pymupdf4llm
            markdown_content = pymupdf4llm.to_markdown(str(file_path))
        except Exception as e:
            # Fallback using fitz directly if pymupdf4llm fails or is missing
            try:
                import fitz
                doc = fitz.open(file_path)
                pages = []
                for page in doc:
                    pages.append(page.get_text())
                markdown_content = "\n\n---\n\n".join(pages)
            except Exception as inner_e:
                markdown_content = f"# {file_path.stem}\n\n*Error extracting text: {e} | {inner_e}*"

        metadata = {
            "title": file_path.stem,
            "source": file_path.name,
            "file_type": "pdf",
            "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "tags": ["document", "pdf"]
        }

        frontmatter = f"---\n{yaml.dump(metadata, sort_keys=False)}---\n\n"
        return frontmatter + markdown_content
