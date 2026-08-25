from pathlib import Path
import pytest
from src.ingest.pdf_parser import PDFParser
from src.ingest.docx_parser import DOCXParser
from src.ingest.pptx_parser import PPTXParser
from src.ingest.pipeline import DocumentIngestPipeline


def test_parsers_fallback_with_dummy_files(tmp_path):
    # Test markdown input pipeline processing
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Test Title\nThis is test content.", encoding="utf-8")

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    pipeline = DocumentIngestPipeline(raw_dir=tmp_path, vault_dir=vault_dir)
    output_path = pipeline.process_file(md_file)

    assert output_path.exists()
    assert output_path.name == "test_doc.md"
    assert "This is test content" in output_path.read_text(encoding="utf-8")


def test_pptx_parser(tmp_path):
    parser = PPTXParser()
    dummy_file = tmp_path / "sample.pptx"
    dummy_file.write_bytes(b"dummy pptx content")
    result = parser.parse(dummy_file)
    assert "title: sample" in result
    assert "file_type: pptx" in result
