"""Unit tests for ContextAugmenter."""

import pytest
from langchain_core.documents import Document

from src.marketing_agents.rag.context_augmenter import ContextAugmenter


class TestContextAugmenter:
    """Test suite for ContextAugmenter class."""

    def test_initialization(self):
        """Test ContextAugmenter initialization with default and custom max_context_length."""
        # Default initialization
        augmenter = ContextAugmenter()
        assert augmenter.max_context_length == 4000

        # Custom max_context_length
        augmenter = ContextAugmenter(max_context_length=2000)
        assert augmenter.max_context_length == 2000

    def test_augment_prompt_basic(self):
        """Test basic prompt augmentation with citations."""
        augmenter = ContextAugmenter()

        docs = [
            Document(
                page_content="This is test content.",
                metadata={"title": "Test Doc", "source": "https://example.com"},
            )
        ]

        result = augmenter.augment_prompt(
            query="Test query?", retrieved_docs=docs, include_citations=True
        )

        # Check result structure
        assert "augmented_prompt" in result
        assert "context" in result
        assert "sources" in result
        assert "context_length" in result

        # Check content
        assert "CONTEXT:" in result["augmented_prompt"]
        assert "QUESTION: Test query?" in result["augmented_prompt"]
        assert "ANSWER:" in result["augmented_prompt"]
        assert "[Source 1]" in result["context"]
        assert "Test Doc" in result["context"]

        # Check sources metadata
        assert len(result["sources"]) == 1
        assert result["sources"][0]["index"] == 1
        assert result["sources"][0]["title"] == "Test Doc"

    def test_augment_prompt_without_citations(self):
        """Test prompt augmentation without citation metadata."""
        augmenter = ContextAugmenter()

        docs = [
            Document(
                page_content="Content without metadata.",
                metadata={"title": "Test", "source": "https://example.com"},
            )
        ]

        result = augmenter.augment_prompt(
            query="Query?", retrieved_docs=docs, include_citations=False
        )

        # Should have source marker but not title/URL
        assert "[Source 1]" in result["context"]
        assert "Title:" not in result["context"]
        assert "URL:" not in result["context"]

    def test_augment_prompt_empty_docs(self):
        """Test augmentation with no documents."""
        augmenter = ContextAugmenter()

        result = augmenter.augment_prompt(
            query="Test query?", retrieved_docs=[], include_citations=True
        )

        assert result["context"] == ""
        assert result["sources"] == []
        assert result["context_length"] == 0
        assert "QUESTION: Test query?" in result["augmented_prompt"]

    def test_augment_prompt_multiple_docs(self):
        """Test augmentation with multiple documents."""
        augmenter = ContextAugmenter()

        docs = [
            Document(
                page_content="First document.",
                metadata={"title": "Doc 1", "source": "https://example.com/1"},
            ),
            Document(
                page_content="Second document.",
                metadata={"title": "Doc 2", "source": "https://example.com/2"},
            ),
            Document(
                page_content="Third document.",
                metadata={"title": "Doc 3", "source": "https://example.com/3"},
            ),
        ]

        result = augmenter.augment_prompt(
            query="Multi-doc query?", retrieved_docs=docs, include_citations=True
        )

        # Check all sources are present
        assert len(result["sources"]) == 3
        assert "[Source 1]" in result["context"]
        assert "[Source 2]" in result["context"]
        assert "[Source 3]" in result["context"]

        # Check indices are correct
        indices = [s["index"] for s in result["sources"]]
        assert indices == [1, 2, 3]

    def test_truncate_context_no_truncation_needed(self):
        """Test truncation when context is within limits."""
        augmenter = ContextAugmenter(max_context_length=1000)

        short_context = "This is a short context that doesn't need truncation."
        result = augmenter._truncate_context(short_context, max_length=1000)

        assert result == short_context

    def test_truncate_context_with_sources(self):
        """Test truncation preserves source structure."""
        augmenter = ContextAugmenter(max_context_length=500)

        long_context = (
            "[Source 1] First document content " + "x" * 200 + "\n\n"
            "[Source 2] Second document content " + "y" * 200 + "\n\n"
            "[Source 3] Third document content " + "z" * 200
        )

        result = augmenter._truncate_context(long_context, max_length=500)

        assert len(result) <= 500
        assert "[truncated]" in result or "Earlier sources truncated" in result

    def test_extract_citations_basic(self):
        """Test basic citation extraction."""
        augmenter = ContextAugmenter()

        sources = [
            {"index": 1, "title": "Source 1", "url": "https://example.com/1"},
            {"index": 2, "title": "Source 2", "url": "https://example.com/2"},
        ]

        response = "This uses source [1] and also [2]."
        citations = augmenter.extract_citations(response, sources)

        assert citations["total_citations"] == 2
        assert len(citations["cited_sources"]) == 2
        assert citations["citation_indices"] == [1, 2]
        assert len(citations["uncited_sources"]) == 0

    def test_extract_citations_with_source_prefix(self):
        """Test extraction with [Source N] format."""
        augmenter = ContextAugmenter()

        sources = [
            {"index": 1, "title": "Source 1", "url": "https://example.com/1"},
            {"index": 2, "title": "Source 2", "url": "https://example.com/2"},
        ]

        response = "Using [Source 1] and [Source 2] here."
        citations = augmenter.extract_citations(response, sources)

        assert len(citations["cited_sources"]) == 2
        assert citations["citation_indices"] == [1, 2]

    def test_extract_citations_duplicate_citations(self):
        """Test that duplicate citations are counted but deduplicated."""
        augmenter = ContextAugmenter()

        sources = [{"index": 1, "title": "Source 1", "url": "https://example.com/1"}]

        response = "Using [1] multiple times [1] and again [1]."
        citations = augmenter.extract_citations(response, sources)

        assert citations["total_citations"] == 3  # Total mentions
        assert len(citations["cited_sources"]) == 1  # Unique sources
        assert citations["citation_indices"] == [1]

    def test_extract_citations_partial_usage(self):
        """Test when only some sources are cited."""
        augmenter = ContextAugmenter()

        sources = [
            {"index": 1, "title": "Source 1", "url": "https://example.com/1"},
            {"index": 2, "title": "Source 2", "url": "https://example.com/2"},
            {"index": 3, "title": "Source 3", "url": "https://example.com/3"},
        ]

        response = "Only using [1] and [3]."
        citations = augmenter.extract_citations(response, sources)

        assert len(citations["cited_sources"]) == 2
        assert citations["citation_indices"] == [1, 3]
        assert len(citations["uncited_sources"]) == 1
        assert citations["uncited_sources"][0]["index"] == 2

    def test_extract_citations_no_citations(self):
        """Test extraction when response has no citations."""
        augmenter = ContextAugmenter()

        sources = [{"index": 1, "title": "Source 1", "url": "https://example.com/1"}]

        response = "This response has no citations."
        citations = augmenter.extract_citations(response, sources)

        assert citations["total_citations"] == 0
        assert len(citations["cited_sources"]) == 0
        assert len(citations["uncited_sources"]) == 1

    def test_validate_citations_all_valid(self):
        """Test validation with all valid citations."""
        augmenter = ContextAugmenter()

        sources = [{"index": 1, "title": "Source 1"}, {"index": 2, "title": "Source 2"}]

        cited_indices = [1, 2]
        validated = augmenter._validate_citations(cited_indices, sources)

        assert validated == [1, 2]

    def test_validate_citations_invalid_indices(self):
        """Test validation with invalid citation indices."""
        augmenter = ContextAugmenter()

        sources = [{"index": 1, "title": "Source 1"}, {"index": 2, "title": "Source 2"}]

        cited_indices = [1, 5, 10]  # 5 and 10 don't exist
        validated = augmenter._validate_citations(cited_indices, sources)

        assert validated == [1]

    def test_format_document_with_metadata(self):
        """Test document formatting with full metadata."""
        augmenter = ContextAugmenter()

        doc = Document(
            page_content="Test content here.",
            metadata={"title": "My Title", "source": "https://example.com"},
        )

        formatted = augmenter._format_document(doc, 1, include_citations=True)

        assert "[Source 1]" in formatted
        assert "Title: My Title" in formatted
        assert "URL: https://example.com" in formatted
        assert "Test content here." in formatted

    def test_format_document_without_metadata(self):
        """Test document formatting with minimal metadata."""
        augmenter = ContextAugmenter()

        doc = Document(page_content="Content only.", metadata={})

        formatted = augmenter._format_document(doc, 1, include_citations=False)

        assert "[Source 1]" in formatted
        assert "Title:" not in formatted
        assert "URL:" not in formatted
        assert "Content only." in formatted

    def test_integration_full_workflow(self):
        """Test full workflow: augment -> generate -> extract citations."""
        augmenter = ContextAugmenter(max_context_length=4000)

        # Step 1: Augment prompt
        docs = [
            Document(
                page_content="Stripe uses webhooks for events.",
                metadata={"title": "Webhooks", "source": "https://stripe.com/docs"},
            ),
            Document(
                page_content="Events are sent via POST.",
                metadata={
                    "title": "Event Delivery",
                    "source": "https://stripe.com/docs/events",
                },
            ),
        ]

        result = augmenter.augment_prompt(
            query="How do webhooks work?", retrieved_docs=docs, include_citations=True
        )

        # Step 2: Simulate LLM response
        llm_response = "Stripe uses webhooks [1] and sends them via POST [2]."

        # Step 3: Extract citations
        citations = augmenter.extract_citations(llm_response, result["sources"])

        # Verify complete workflow
        assert len(result["sources"]) == 2
        assert len(citations["cited_sources"]) == 2
        assert citations["total_citations"] == 2
        assert len(citations["uncited_sources"]) == 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
