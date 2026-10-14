from pathlib import Path
from src.ingestion.universal_parser import UniversalDocumentParser, SourceType


def test_universal_parser_markdown(tmp_path):
    md_file = tmp_path / "test_doc.md"
    md_file.write_text("# Test Title\nThis is test content.", encoding="utf-8")

    md_dir = tmp_path / "markdown"
    vault_dir = tmp_path / "vault"
    parser = UniversalDocumentParser(markdown_dir=md_dir, vault_dir=vault_dir)
    
    proc_doc = parser.parse_and_export(md_file)
    assert proc_doc.doc_id == "test_doc"
    assert proc_doc.source_type == SourceType.MARKDOWN
    assert len(proc_doc.chunks) >= 1
    assert "This is test content." in proc_doc.markdown_content
    assert (md_dir / "test_doc.md").exists()
    assert (vault_dir / "test_doc.md").exists()


def test_universal_parser_text(tmp_path):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("Simple plain text notes for indexing.", encoding="utf-8")

    parser = UniversalDocumentParser(markdown_dir=tmp_path / "md", vault_dir=tmp_path / "vault")
    proc_doc = parser.parse_file(txt_file)
    assert proc_doc.doc_id == "notes"
    assert "Simple plain text notes" in proc_doc.markdown_content
