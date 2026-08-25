from pathlib import Path
from typing import List, Dict
import logging
from src.config import settings
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .pptx_parser import PPTXParser

logger = logging.getLogger(__name__)


class DocumentIngestPipeline:
    """Orchestrator for ingesting raw multi-format files into Obsidian vault markdown."""

    def __init__(self, raw_dir: Path = None, vault_dir: Path = None):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.data_raw_dir
        self.vault_dir = Path(vault_dir) if vault_dir else settings.data_vault_dir
        
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.pptx_parser = PPTXParser()

    def process_file(self, file_path: Path) -> Path:
        """Parse a single file and write its markdown output to vault."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            content = self.pdf_parser.parse(file_path)
        elif ext in [".docx", ".doc"]:
            content = self.docx_parser.parse(file_path)
        elif ext in [".pptx", ".ppt"]:
            content = self.pptx_parser.parse(file_path)
        elif ext == ".md":
            # Already markdown, just copy/pass-through
            content = file_path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        output_filename = f"{file_path.stem}.md"
        output_path = self.vault_dir / output_filename
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"Ingested {file_path.name} -> {output_path}")
        return output_path

    def run_all(self) -> List[Path]:
        """Process all supported raw files in data/raw/ directory."""
        output_files = []
        if not self.raw_dir.exists():
            logger.warning(f"Raw directory does not exist: {self.raw_dir}")
            return output_files

        supported_extensions = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".md"}
        for path in self.raw_dir.iterdir():
            if path.is_file() and path.suffix.lower() in supported_extensions:
                try:
                    out_path = self.process_file(path)
                    output_files.append(out_path)
                except Exception as e:
                    logger.error(f"Failed to ingest {path.name}: {e}")
        
        return output_files
