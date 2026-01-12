"""Unit tests for ChunkingStrategy."""

import pytest
from langchain_core.documents import Document

from src.marketing_agents.rag.chunking import ChunkingStrategy


class TestChunkingStrategy:
    """Test suite for ChunkingStrategy class."""

    def test_initialization_default(self):
        """Test ChunkingStrategy initialization with default parameters."""
        chunker = ChunkingStrategy()

        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.separators == ["\n\n", "\n", ". ", " ", ""]

    def test_initialization_custom(self):
        """Test ChunkingStrategy initialization with custom parameters."""
        custom_separators = ["\n\n", "###"]
        chunker = ChunkingStrategy(
            chunk_size=500,
            chunk_overlap=100,
            separators=custom_separators,
        )

        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.separators == custom_separators

    def test_from_document_type_technical_docs(self):
        """Test creating strategy for technical documentation."""
        chunker = ChunkingStrategy.from_document_type("technical_docs")

        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
        assert chunker.separators == ["\n\n", "\n", ". ", " "]

    def test_from_document_type_api_reference(self):
        """Test creating strategy for API reference."""
        chunker = ChunkingStrategy.from_document_type("api_reference")

        assert chunker.chunk_size == 800
        assert chunker.chunk_overlap == 150
        assert chunker.separators == ["\n\n", "###", "##"]

    def test_from_document_type_tutorials(self):
        """Test creating strategy for tutorials."""
        chunker = ChunkingStrategy.from_document_type("tutorials")

        assert chunker.chunk_size == 1200
        assert chunker.chunk_overlap == 250
        assert chunker.separators == ["\n\n", "\n"]

    def test_from_document_type_faq(self):
        """Test creating strategy for FAQ."""
        chunker = ChunkingStrategy.from_document_type("faq")

        assert chunker.chunk_size == 600
        assert chunker.chunk_overlap == 100
        assert chunker.separators == ["\n\n", "?", ". "]

    def test_from_document_type_unknown(self):
        """Test creating strategy with unknown document type (should use defaults)."""
        chunker = ChunkingStrategy.from_document_type("unknown_type")

        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_chunk_single_document(self):
        """Test chunking a single document."""
        chunker = ChunkingStrategy(chunk_size=100, chunk_overlap=20)

        doc = Document(
            page_content="This is a test document. " * 20,  # Long enough to chunk
            metadata={"source": "test.md", "title": "Test Doc"},
        )

        chunks = chunker._chunk_single_document(doc)

        assert len(chunks) > 1  # Should create multiple chunks
        assert all(isinstance(chunk, Document) for chunk in chunks)

        # Check metadata inheritance
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata["source"] == "test.md"
            assert chunk.metadata["title"] == "Test Doc"
            assert chunk.metadata["chunk_index"] == idx
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_size"] == len(chunk.page_content)

    def test_chunk_documents_multiple(self):
        """Test chunking multiple documents."""
        chunker = ChunkingStrategy(chunk_size=100, chunk_overlap=20)

        docs = [
            Document(
                page_content="First document content. " * 15,
                metadata={"source": "doc1.md"},
            ),
            Document(
                page_content="Second document content. " * 15,
                metadata={"source": "doc2.md"},
            ),
        ]

        all_chunks = chunker.chunk_documents(docs)

        # Should have chunks from both documents
        assert len(all_chunks) > 2
        assert all(isinstance(chunk, Document) for chunk in all_chunks)

        # Check that we have chunks from both sources
        sources = {chunk.metadata["source"] for chunk in all_chunks}
        assert "doc1.md" in sources
        assert "doc2.md" in sources

    def test_preserve_code_blocks(self):
        """Test code block preservation."""
        chunker = ChunkingStrategy()

        text = """
Some text before code.

```python
def hello():
    print("Hello, world!")
```

Some text after code.
"""

        modified_text, code_blocks = chunker._preserve_code_blocks(text)

        # Should have one code block marker
        assert len(code_blocks) == 1
        assert "<<<CODE_BLOCK_0>>>" in modified_text
        assert "```python" not in modified_text  # Code should be replaced

        # Code block should be preserved
        marker = list(code_blocks.keys())[0]
        assert "def hello():" in code_blocks[marker]

    def test_restore_code_blocks(self):
        """Test code block restoration."""
        chunker = ChunkingStrategy()

        code_blocks = {
            "<<<CODE_BLOCK_0>>>": "```python\nprint('test')\n```",
            "<<<CODE_BLOCK_1>>>": "```javascript\nconsole.log('test');\n```",
        }

        chunks = [
            "Text before <<<CODE_BLOCK_0>>> text after",
            "Another chunk with <<<CODE_BLOCK_1>>> here",
        ]

        restored = chunker._restore_code_blocks(chunks, code_blocks)

        assert len(restored) == 2
        assert "```python" in restored[0]
        assert "print('test')" in restored[0]
        assert "```javascript" in restored[1]
        assert "console.log('test');" in restored[1]

    def test_split_by_sections(self):
        """Test splitting by markdown headers."""
        chunker = ChunkingStrategy()

        text = """# Main Header
Some content under main header.

## Subsection 1
Content for subsection 1.

## Subsection 2
Content for subsection 2.

### Nested Header
Nested content.
"""

        sections = chunker._split_by_sections(text)

        # Should split into sections based on headers
        assert len(sections) > 1
        assert any("# Main Header" in section for section in sections)
        assert any("## Subsection 1" in section for section in sections)

    def test_split_by_sections_no_headers(self):
        """Test splitting text without headers."""
        chunker = ChunkingStrategy()

        text = "This is plain text without any headers. Just paragraphs."

        sections = chunker._split_by_sections(text)

        # Should return single section
        assert len(sections) == 1
        assert sections[0] == text

    def test_optimize_chunk_boundaries(self):
        """Test chunk boundary optimization."""
        chunker = ChunkingStrategy()

        chunks = [
            "This is a complete sentence. This is another one.",
            "   Leading whitespace should be removed.   ",
            "incomplete sentence at end",
            "",  # Empty chunk should be removed
            "- List item 1\n- List item 2\n- List item 3",
        ]

        optimized = chunker._optimize_chunk_boundaries(chunks)

        # Empty chunks should be removed
        assert len(optimized) == 4

        # Whitespace should be stripped
        assert not optimized[1].startswith(" ")
        assert not optimized[1].endswith(" ")

        # List should be preserved
        assert "- List item 1" in optimized[3]

    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        chunker = ChunkingStrategy()

        chunk_id_1 = chunker._generate_chunk_id("docs/test.md", 0)
        chunk_id_2 = chunker._generate_chunk_id("docs/test.md", 1)
        chunk_id_3 = chunker._generate_chunk_id("docs/test.md", 0)

        # Should contain filename and index
        assert "test" in chunk_id_1
        assert "_0_" in chunk_id_1
        assert "_1_" in chunk_id_2

        # Should be unique (due to UUID)
        assert chunk_id_1 != chunk_id_3

    def test_validate_chunks_empty(self):
        """Test validation with empty chunk list."""
        chunker = ChunkingStrategy()

        validation = chunker.validate_chunks([])

        assert validation["total_chunks"] == 0
        assert validation["avg_chunk_size"] == 0
        assert validation["oversized_chunks"] == []

    def test_validate_chunks_normal(self):
        """Test validation with normal chunks."""
        chunker = ChunkingStrategy(chunk_size=100, chunk_overlap=20)

        chunks = [
            Document(
                page_content="a" * 90,  # Within target
                metadata={"chunk_id": "chunk1", "source": "test.md"},
            ),
            Document(
                page_content="b" * 110,  # Within target
                metadata={"chunk_id": "chunk2", "source": "test.md"},
            ),
            Document(
                page_content="c" * 200,  # Oversized (>150)
                metadata={"chunk_id": "chunk3", "source": "test.md"},
            ),
        ]

        validation = chunker.validate_chunks(chunks)

        assert validation["total_chunks"] == 3
        assert validation["avg_chunk_size"] == (90 + 110 + 200) / 3
        assert validation["min_chunk_size"] == 90
        assert validation["max_chunk_size"] == 200
        assert validation["within_target"] == 2  # First two chunks
        assert len(validation["oversized_chunks"]) == 1  # Third chunk
        assert validation["oversized_chunks"][0]["chunk_id"] == "chunk3"

    def test_chunk_with_code_blocks(self):
        """Test end-to-end chunking with code blocks."""
        chunker = ChunkingStrategy(chunk_size=200, chunk_overlap=50)

        doc = Document(
            page_content="""
# API Reference

This is documentation text before the code example.

```python
def example_function():
    print("This code should stay together")
    return True
```

This is text after the code block. It continues with more documentation.
""",
            metadata={"source": "api_ref.md", "doc_type": "api"},
        )

        chunks = chunker._chunk_single_document(doc)

        # Code block should be preserved in at least one chunk
        has_complete_code = any(
            "def example_function():" in chunk.page_content
            and 'print("This code should stay together")' in chunk.page_content
            for chunk in chunks
        )

        assert has_complete_code, "Code block should be kept together"

    def test_chunk_with_lists(self):
        """Test chunking with list items."""
        chunker = ChunkingStrategy(chunk_size=150, chunk_overlap=30)

        doc = Document(
            page_content="""
# Features

Our product includes:

- Feature one with description
- Feature two with description
- Feature three with description
- Feature four with description
- Feature five with description

These features work together seamlessly.
""",
            metadata={"source": "features.md"},
        )

        chunks = chunker._chunk_single_document(doc)

        # Lists should be handled properly
        assert len(chunks) >= 1

        # At least one chunk should contain list items
        has_list_items = any("-" in chunk.page_content for chunk in chunks)

        assert has_list_items
