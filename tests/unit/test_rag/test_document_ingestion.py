"""Tests for DocumentIngestionPipeline."""

import asyncio
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.marketing_agents.rag.document_ingestion import DocumentIngestionPipeline


@pytest.fixture
def temp_knowledge_base(tmp_path):
    """Create a temporary knowledge base directory with test files."""
    # Create directory structure
    api_dir = tmp_path / "api"
    api_dir.mkdir()

    tutorials_dir = tmp_path / "tutorials"
    tutorials_dir.mkdir()

    # Create test files
    # Markdown file with title
    md_content = """# Stripe API Overview

This is a comprehensive guide to the Stripe API.

## Getting Started

The Stripe API allows you to process payments programmatically.
"""
    (api_dir / "overview.md").write_text(md_content)

    # HTML file
    html_content = """<html>
<head><title>Payment Processing</title></head>
<body>
<h1>Payment Processing Guide</h1>
<p>Learn how to process payments with Stripe.</p>
<script>console.log('test');</script>
</body>
</html>"""
    (tutorials_dir / "payment.html").write_text(html_content)

    # Plain text file
    txt_content = """Frequently Asked Questions

Q: How do I get started?
A: First, create an account.

Q: What payment methods are supported?
A: Credit cards, debit cards, and bank transfers.
"""
    (tmp_path / "faq.txt").write_text(txt_content)

    # Unsupported file (should be skipped)
    (tmp_path / "image.png").write_bytes(b"fake image data")

    return tmp_path


class TestDocumentIngestionPipeline:
    """Tests for DocumentIngestionPipeline class."""

    def test_initialization(self, temp_knowledge_base):
        """Test pipeline initialization."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        assert pipeline.source_dir == temp_knowledge_base
        assert pipeline.source_type == "stripe_docs"
        assert pipeline.encoding == "utf-8"

    def test_initialization_invalid_directory(self):
        """Test initialization with invalid directory."""
        with pytest.raises(ValueError, match="does not exist"):
            DocumentIngestionPipeline("/nonexistent/directory")

    def test_discover_files(self, temp_knowledge_base):
        """Test file discovery."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        files = pipeline._discover_files()

        # Should find 3 supported files (.md, .html, .txt)
        assert len(files) == 3

        # Check that file paths are absolute
        for file_path in files:
            assert Path(file_path).is_absolute()

        # Check that unsupported files are skipped
        assert pipeline.stats["skipped"] == 1

    def test_load_file_markdown(self, temp_knowledge_base):
        """Test loading markdown file."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        file_path = str(temp_knowledge_base / "api" / "overview.md")

        content = pipeline._load_file(file_path)

        assert "Stripe API Overview" in content
        assert "Getting Started" in content
        assert len(content) > 0

    def test_load_file_html(self, temp_knowledge_base):
        """Test loading and cleaning HTML file."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        file_path = str(temp_knowledge_base / "tutorials" / "payment.html")

        content = pipeline._load_file(file_path)

        # HTML tags should be removed
        assert "<html>" not in content
        assert "<body>" not in content

        # Content should be preserved
        assert "Payment Processing Guide" in content
        assert "Learn how to process payments" in content

        # Script content should be removed
        assert "console.log" not in content

    def test_extract_metadata(self, temp_knowledge_base):
        """Test metadata extraction."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        file_path = str(temp_knowledge_base / "api" / "overview.md")
        content = pipeline._load_file(file_path)

        metadata = pipeline._extract_metadata(file_path, content)

        # Check required metadata fields
        assert metadata["source"] == "api/overview.md"
        assert metadata["source_type"] == "stripe_docs"
        assert metadata["file_name"] == "overview.md"
        assert metadata["file_extension"] == ".md"
        assert metadata["title"] == "Stripe API Overview"
        assert metadata["category"] == "API Reference"
        assert metadata["word_count"] > 0
        assert metadata["char_count"] > 0
        assert metadata["reading_time_minutes"] >= 1
        assert metadata["language"] == "en"
        assert "ingested_at" in metadata

    def test_extract_title(self, temp_knowledge_base):
        """Test title extraction."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))

        # Markdown H1
        content = "# My Title\n\nContent here"
        title = pipeline._extract_title(content)
        assert title == "My Title"

        # No clear title
        content = "Just some text without title"
        title = pipeline._extract_title(content)
        assert title == "Just some text without title"

    def test_categorize_document(self, temp_knowledge_base):
        """Test document categorization."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))

        # API reference
        api_content = "API endpoint: GET /v1/customers\nParameters: limit, offset"
        category = pipeline._categorize_document(api_content, "/api/reference.md")
        assert category == "API Reference"

        # Tutorial
        tutorial_content = "Step 1: Install\nStep 2: Configure\nStep 3: Deploy"
        category = pipeline._categorize_document(
            tutorial_content, "/tutorials/guide.md"
        )
        assert category == "Tutorial"

    @pytest.mark.asyncio
    async def test_ingest_documents(self, temp_knowledge_base):
        """Test full document ingestion."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))

        documents = await pipeline.ingest_documents()

        # Should ingest 3 documents
        assert len(documents) == 3

        # All should be LangChain Document objects
        for doc in documents:
            assert isinstance(doc, Document)
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            assert len(doc.page_content) > 0
            assert "source" in doc.metadata
            assert "title" in doc.metadata
            assert "category" in doc.metadata

        # Check statistics
        stats = pipeline.get_statistics()
        assert stats["total_files"] == 3
        assert stats["successful"] == 3
        assert stats["failed"] == 0
        assert stats["skipped"] == 1  # The .png file
        assert stats["total_words"] > 0
        assert stats["total_chars"] > 0

    @pytest.mark.asyncio
    async def test_process_file(self, temp_knowledge_base):
        """Test processing a single file."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))
        file_path = str(temp_knowledge_base / "api" / "overview.md")

        document = await pipeline._process_file(file_path)

        assert isinstance(document, Document)
        assert "Stripe API Overview" in document.page_content
        assert document.metadata["title"] == "Stripe API Overview"
        assert document.metadata["category"] == "API Reference"

    def test_get_statistics(self, temp_knowledge_base):
        """Test statistics retrieval."""
        pipeline = DocumentIngestionPipeline(str(temp_knowledge_base))

        stats = pipeline.get_statistics()

        # Should return a copy of stats
        assert isinstance(stats, dict)
        assert "total_files" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "skipped" in stats
