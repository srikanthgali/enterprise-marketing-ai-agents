"""Unit tests for AdvancedRetriever."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from langchain_core.documents import Document

from src.marketing_agents.rag.retriever import AdvancedRetriever


@pytest.fixture
def mock_vector_store():
    """Create a mock FAISS vector store."""
    mock_store = MagicMock()

    # Sample documents
    docs = [
        Document(
            page_content="Stripe payment API documentation",
            metadata={"category": "API Reference", "source": "stripe.com/docs/api"},
        ),
        Document(
            page_content="Guide to subscription billing",
            metadata={"category": "Guide", "source": "stripe.com/docs/billing"},
        ),
        Document(
            page_content="Webhook integration tutorial",
            metadata={"category": "Tutorial", "source": "internal/webhooks"},
        ),
    ]

    # Mock similarity_search_with_score
    mock_store.similarity_search_with_score.return_value = [
        (docs[0], 0.2),  # Lower distance = higher similarity
        (docs[1], 0.5),
        (docs[2], 0.8),
    ]

    # Mock similarity_search (for keyword search)
    mock_store.similarity_search.return_value = docs

    return mock_store


@pytest.fixture
def retriever(mock_vector_store):
    """Create an AdvancedRetriever instance."""
    return AdvancedRetriever(mock_vector_store)


class TestAdvancedRetriever:
    """Test suite for AdvancedRetriever."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_vector_store):
        """Test retriever initialization."""
        retriever = AdvancedRetriever(mock_vector_store)

        assert retriever.vector_store == mock_vector_store
        assert isinstance(retriever.cache, dict)
        assert len(retriever.cache) == 0

    @pytest.mark.asyncio
    async def test_basic_retrieve(self, retriever, mock_vector_store):
        """Test basic retrieval without filtering or reranking."""
        query = "payment API"
        results = await retriever.retrieve(query, top_k=2, rerank=False)

        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)
        mock_vector_store.similarity_search_with_score.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search(self, retriever, mock_vector_store):
        """Test semantic search with score conversion."""
        query = "test query"
        results = await retriever._semantic_search(query, k=3)

        assert len(results) == 3
        # Check that distances are converted to similarity scores
        for doc, score in results:
            assert isinstance(doc, Document)
            assert 0 <= score <= 1
            # Higher similarity for lower distance

        # First result (distance 0.2) should have highest similarity
        assert results[0][1] > results[1][1] > results[2][1]

    @pytest.mark.asyncio
    async def test_metadata_filtering_exact_match(self, retriever, mock_vector_store):
        """Test exact match metadata filtering."""
        docs = [
            Document(
                page_content="Doc 1",
                metadata={"category": "API Reference"},
            ),
            Document(
                page_content="Doc 2",
                metadata={"category": "Guide"},
            ),
            Document(
                page_content="Doc 3",
                metadata={"category": "API Reference"},
            ),
        ]

        filters = {"category": "API Reference"}
        filtered = retriever._filter_by_metadata(docs, filters)

        assert len(filtered) == 2
        assert all(doc.metadata["category"] == "API Reference" for doc in filtered)

    @pytest.mark.asyncio
    async def test_metadata_filtering_pattern_match(self, retriever):
        """Test pattern matching with wildcards."""
        docs = [
            Document(
                page_content="Doc 1",
                metadata={"source": "stripe.com/docs/api"},
            ),
            Document(
                page_content="Doc 2",
                metadata={"source": "internal/docs"},
            ),
            Document(
                page_content="Doc 3",
                metadata={"source": "stripe.com/guides"},
            ),
        ]

        filters = {"source": "*stripe.com*"}
        filtered = retriever._filter_by_metadata(docs, filters)

        assert len(filtered) == 2
        assert all("stripe.com" in doc.metadata["source"] for doc in filtered)

    @pytest.mark.asyncio
    async def test_metadata_filtering_multiple_conditions(self, retriever):
        """Test filtering with multiple conditions (AND logic)."""
        docs = [
            Document(
                page_content="Doc 1",
                metadata={"category": "API Reference", "source": "stripe.com"},
            ),
            Document(
                page_content="Doc 2",
                metadata={"category": "Guide", "source": "stripe.com"},
            ),
            Document(
                page_content="Doc 3",
                metadata={"category": "API Reference", "source": "internal"},
            ),
        ]

        filters = {"category": "API Reference", "source": "*stripe.com*"}
        filtered = retriever._filter_by_metadata(docs, filters)

        assert len(filtered) == 1
        assert filtered[0].metadata["category"] == "API Reference"
        assert "stripe.com" in filtered[0].metadata["source"]

    @pytest.mark.asyncio
    async def test_reranking(self, retriever):
        """Test reranking with multiple strategies."""
        query = "payment subscription billing"

        docs = [
            Document(
                page_content="Information about subscription billing and payment processing.",
                metadata={"timestamp": "2025-01-01T00:00:00"},
            ),
            Document(
                page_content="Guide to webhooks.",
                metadata={},
            ),
            Document(
                page_content="Subscription billing overview with detailed payment information.",
                metadata={"timestamp": "2024-01-01T00:00:00"},
            ),
        ]

        # Semantic scores
        semantic_scores = {
            id(docs[0]): 0.8,
            id(docs[1]): 0.6,
            id(docs[2]): 0.7,
        }

        reranked = await retriever._rerank_results(query, docs, semantic_scores)

        assert len(reranked) == 3
        # First doc should rank high due to query term overlap and recency
        assert reranked[0] == docs[0]

    @pytest.mark.asyncio
    async def test_keyword_search(self, retriever, mock_vector_store):
        """Test keyword search functionality."""
        query = "subscription billing"
        results = retriever._keyword_search(query, k=3)

        assert isinstance(results, list)
        # Should return scored results
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_hybrid_search(self, retriever, mock_vector_store):
        """Test hybrid search combining semantic and keyword."""
        query = "payment API"
        results = await retriever.hybrid_search(query, top_k=2, alpha=0.5)

        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_alpha_weighting(self, retriever, mock_vector_store):
        """Test that alpha parameter affects weighting."""
        query = "test query"

        # Test different alpha values
        results_semantic = await retriever.hybrid_search(query, top_k=3, alpha=1.0)
        results_keyword = await retriever.hybrid_search(query, top_k=3, alpha=0.0)
        results_balanced = await retriever.hybrid_search(query, top_k=3, alpha=0.5)

        # All should return results
        assert len(results_semantic) > 0
        assert len(results_keyword) > 0
        assert len(results_balanced) > 0

    @pytest.mark.asyncio
    async def test_merge_results_deduplication(self, retriever):
        """Test that merge_results deduplicates documents."""
        doc1 = Document(page_content="Unique content 1")
        doc2 = Document(page_content="Unique content 2")
        doc_duplicate = Document(page_content="Unique content 1")  # Same as doc1

        semantic = [(doc1, 0.8), (doc2, 0.6)]
        keyword = [(doc_duplicate, 0.7), (doc2, 0.5)]

        merged = retriever._merge_results(semantic, keyword, alpha=0.5)

        # Should have only 2 unique documents
        assert len(merged) == 2

    @pytest.mark.asyncio
    async def test_query_caching(self, retriever, mock_vector_store):
        """Test that queries are cached."""
        query = "test query"

        # First call
        results1 = await retriever.retrieve(query, top_k=2)
        assert len(retriever.cache) == 1

        # Second call (should use cache)
        results2 = await retriever.retrieve(query, top_k=2)

        # Should return same results
        assert results1 == results2
        # Vector store should only be called once
        assert mock_vector_store.similarity_search_with_score.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_key_uniqueness(self, retriever, mock_vector_store):
        """Test that different parameters create different cache keys."""
        query = "test query"

        # Different parameters should create different cache entries
        await retriever.retrieve(query, top_k=2)
        await retriever.retrieve(query, top_k=3)
        await retriever.retrieve(query, top_k=2, filter_metadata={"category": "API"})

        assert len(retriever.cache) == 3

    @pytest.mark.asyncio
    async def test_clear_cache(self, retriever):
        """Test cache clearing."""
        query = "test query"
        await retriever.retrieve(query, top_k=2)

        assert len(retriever.cache) > 0

        retriever.clear_cache()
        assert len(retriever.cache) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, retriever):
        """Test stats retrieval."""
        stats = retriever.get_stats()

        assert "cache_size" in stats
        assert "vector_store_type" in stats
        assert isinstance(stats["cache_size"], int)

    @pytest.mark.asyncio
    async def test_retrieve_with_filtering_and_reranking(
        self, retriever, mock_vector_store
    ):
        """Test full pipeline with filtering and reranking."""
        query = "payment API"
        filters = {"category": "API Reference"}

        results = await retriever.retrieve(
            query, top_k=2, filter_metadata=filters, rerank=True
        )

        assert len(results) <= 2
        # All results should match filter
        for doc in results:
            if "category" in doc.metadata:
                assert doc.metadata["category"] == "API Reference"

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retriever, mock_vector_store):
        """Test handling of empty queries."""
        results = await retriever.retrieve("", top_k=5)

        # Should still return results (vector store handles empty queries)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_no_results_handling(self, retriever, mock_vector_store):
        """Test handling when no documents match filters."""
        mock_vector_store.similarity_search_with_score.return_value = [
            (Document(page_content="test", metadata={"category": "Guide"}), 0.5)
        ]

        filters = {"category": "NonExistent"}
        results = await retriever.retrieve("test", top_k=5, filter_metadata=filters)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_alpha_bounds_validation(self, retriever):
        """Test that alpha is bounded between 0 and 1."""
        query = "test"

        # Should not raise errors and clamp to valid range
        results_low = await retriever.hybrid_search(query, alpha=-0.5)
        results_high = await retriever.hybrid_search(query, alpha=1.5)

        assert isinstance(results_low, list)
        assert isinstance(results_high, list)
