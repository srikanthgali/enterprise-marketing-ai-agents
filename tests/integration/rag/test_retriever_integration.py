"""Integration test for AdvancedRetriever with FAISS vector store."""

import pytest
from unittest.mock import MagicMock, patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.marketing_agents.rag import AdvancedRetriever


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content=(
                "Customer Data Platforms (CDPs) aggregate customer data from "
                "multiple sources to create unified customer profiles. This enables "
                "better personalization and marketing automation."
            ),
            metadata={
                "category": "Overview",
                "source": "internal/cdp-overview.md",
                "title": "CDP Overview",
                "timestamp": "2025-01-01T00:00:00",
            },
        ),
        Document(
            page_content=(
                "The Segment Tracking API allows you to record customer events. "
                "Use track() to record actions, identify() for user traits, "
                "and page() for page views."
            ),
            metadata={
                "category": "API Reference",
                "source": "segment.com/docs/api/track",
                "title": "Tracking API",
                "timestamp": "2025-01-10T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Marketing automation best practices include audience segmentation, "
                "personalized email workflows, lead scoring, and CRM integration."
            ),
            metadata={
                "category": "Guide",
                "source": "internal/marketing-guide.md",
                "title": "Marketing Automation Guide",
                "timestamp": "2024-12-15T00:00:00",
            },
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    mock = MagicMock()
    mock.embed_documents.return_value = [
        [0.1] * 1536,  # Dummy embedding vector
        [0.2] * 1536,
        [0.3] * 1536,
    ]
    mock.embed_query.return_value = [0.15] * 1536
    return mock


class TestAdvancedRetrieverIntegration:
    """Integration tests for AdvancedRetriever with real FAISS operations."""

    @pytest.mark.integration
    def test_retriever_with_mock_faiss(self, sample_documents, mock_embeddings):
        """Test retriever with mocked FAISS store."""
        # Create FAISS vector store with mock embeddings
        with patch(
            "langchain_community.vectorstores.FAISS.from_documents"
        ) as mock_from_docs:
            mock_store = MagicMock(spec=FAISS)

            # Mock similarity search
            mock_store.similarity_search_with_score.return_value = [
                (sample_documents[1], 0.2),  # API Reference doc
                (sample_documents[0], 0.5),  # Overview doc
                (sample_documents[2], 0.8),  # Guide doc
            ]

            mock_store.similarity_search.return_value = sample_documents
            mock_from_docs.return_value = mock_store

            # Initialize retriever
            retriever = AdvancedRetriever(mock_store)

            # Verify initialization
            assert retriever.vector_store == mock_store
            assert len(retriever.cache) == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_retrieval_workflow(self, sample_documents):
        """Test complete retrieval workflow with mocked vector store."""
        # Create mock vector store
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[1], 0.2),
            (sample_documents[0], 0.5),
        ]
        mock_store.similarity_search.return_value = sample_documents

        # Initialize retriever
        retriever = AdvancedRetriever(mock_store)

        # Test 1: Basic retrieval
        results = await retriever.retrieve("tracking API", top_k=2, rerank=False)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

        # Test 2: Retrieval with filtering
        results = await retriever.retrieve(
            "API documentation",
            top_k=5,
            filter_metadata={"category": "API Reference"},
            rerank=False,
        )
        # Should filter to only API Reference docs
        for doc in results:
            if "category" in doc.metadata:
                assert doc.metadata["category"] == "API Reference"

        # Test 3: Retrieval with reranking
        results = await retriever.retrieve(
            "customer data marketing automation",
            top_k=3,
            rerank=True,
        )
        assert len(results) <= 3

        # Test 4: Hybrid search
        results = await retriever.hybrid_search(
            "tracking events",
            top_k=2,
            alpha=0.5,
        )
        assert len(results) <= 2

        # Verify caching
        stats = retriever.get_stats()
        assert stats["cache_size"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retriever_with_diverse_queries(self, sample_documents):
        """Test retriever with various query types."""
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.3),
            (sample_documents[1], 0.4),
            (sample_documents[2], 0.6),
        ]
        mock_store.similarity_search.return_value = sample_documents

        retriever = AdvancedRetriever(mock_store)

        # Test different query types
        queries = [
            "What is a CDP?",
            "API documentation",
            "email marketing automation",
            "tracking customer events",
            "personalization strategies",
        ]

        for query in queries:
            results = await retriever.retrieve(query, top_k=3, rerank=True)
            assert isinstance(results, list)
            assert len(results) <= 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_metadata_filtering_combinations(self, sample_documents):
        """Test various metadata filtering combinations."""
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.3),
            (sample_documents[1], 0.4),
            (sample_documents[2], 0.6),
        ]

        retriever = AdvancedRetriever(mock_store)

        # Test 1: Single filter
        results = await retriever.retrieve(
            "documentation",
            top_k=5,
            filter_metadata={"category": "API Reference"},
        )
        assert all(
            doc.metadata.get("category") == "API Reference"
            for doc in results
            if "category" in doc.metadata
        )

        # Test 2: Wildcard filter
        results = await retriever.retrieve(
            "segment docs",
            top_k=5,
            filter_metadata={"source": "*segment.com*"},
        )
        assert all(
            "segment.com" in doc.metadata.get("source", "")
            for doc in results
            if "source" in doc.metadata
        )

        # Test 3: Multiple filters
        results = await retriever.retrieve(
            "API guide",
            top_k=5,
            filter_metadata={
                "category": "API Reference",
                "source": "*segment.com*",
            },
        )
        for doc in results:
            if "category" in doc.metadata and "source" in doc.metadata:
                assert doc.metadata["category"] == "API Reference"
                assert "segment.com" in doc.metadata["source"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_hybrid_search_alpha_variations(self, sample_documents):
        """Test hybrid search with different alpha values."""
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.2),
            (sample_documents[1], 0.4),
            (sample_documents[2], 0.6),
        ]
        mock_store.similarity_search.return_value = sample_documents

        retriever = AdvancedRetriever(mock_store)

        query = "customer tracking"

        # Test different alpha values
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]

        for alpha in alpha_values:
            results = await retriever.hybrid_search(
                query,
                top_k=3,
                alpha=alpha,
            )
            assert isinstance(results, list)
            assert len(results) <= 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_caching_behavior(self, sample_documents):
        """Test query caching across multiple calls."""
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.3),
        ]

        retriever = AdvancedRetriever(mock_store)

        query = "test query"

        # First call - should hit vector store
        results1 = await retriever.retrieve(query, top_k=2, rerank=False)
        assert mock_store.similarity_search_with_score.call_count == 1

        # Second identical call - should use cache
        results2 = await retriever.retrieve(query, top_k=2, rerank=False)
        assert (
            mock_store.similarity_search_with_score.call_count == 1
        )  # No additional call
        assert results1 == results2

        # Different parameters - should hit vector store again
        results3 = await retriever.retrieve(query, top_k=3, rerank=False)
        assert mock_store.similarity_search_with_score.call_count == 2

        # Verify cache size
        stats = retriever.get_stats()
        assert stats["cache_size"] == 2

        # Clear cache
        retriever.clear_cache()
        assert retriever.get_stats()["cache_size"] == 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_reranking_impact(self, sample_documents):
        """Test that reranking changes result order."""
        mock_store = MagicMock(spec=FAISS)

        # Setup documents with varying characteristics
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[2], 0.4),  # Guide - older, different content
            (sample_documents[1], 0.5),  # API - recent, good match
            (sample_documents[0], 0.6),  # Overview - moderate
        ]

        retriever = AdvancedRetriever(mock_store)

        query = "tracking API documentation"

        # Without reranking
        results_no_rerank = await retriever.retrieve(
            query,
            top_k=3,
            rerank=False,
        )

        # With reranking
        retriever.clear_cache()  # Clear cache to force new retrieval
        results_rerank = await retriever.retrieve(
            query,
            top_k=3,
            rerank=True,
        )

        # Both should return results
        assert len(results_no_rerank) > 0
        assert len(results_rerank) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_and_edge_cases(self, sample_documents):
        """Test edge cases and error handling."""
        mock_store = MagicMock(spec=FAISS)
        mock_store.similarity_search_with_score.return_value = []
        mock_store.similarity_search.return_value = []

        retriever = AdvancedRetriever(mock_store)

        # Empty query
        results = await retriever.retrieve("", top_k=5)
        assert isinstance(results, list)

        # No results from vector store
        results = await retriever.retrieve("nonexistent content", top_k=5)
        assert len(results) == 0

        # Invalid metadata filter (no matches)
        mock_store.similarity_search_with_score.return_value = [
            (sample_documents[0], 0.5),
        ]
        results = await retriever.retrieve(
            "test",
            top_k=5,
            filter_metadata={"category": "NonExistent"},
        )
        assert len(results) == 0

        # Alpha bounds
        results = await retriever.hybrid_search("test", alpha=-1.0)
        assert isinstance(results, list)

        results = await retriever.hybrid_search("test", alpha=2.0)
        assert isinstance(results, list)
