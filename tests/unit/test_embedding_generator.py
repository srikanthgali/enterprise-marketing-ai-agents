"""
Unit tests for EmbeddingGenerator.

Tests cover:
- Basic embedding generation
- Batch processing
- Caching functionality
- Error handling and retries
- Validation checks
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from langchain_core.documents import Document

from src.marketing_agents.rag.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="This is a test document about AI.",
                metadata={"source": "test1.txt", "doc_id": 1},
            ),
            Document(
                page_content="Machine learning and deep learning are subfields of AI.",
                metadata={"source": "test2.txt", "doc_id": 2},
            ),
            Document(
                page_content="Natural language processing enables computers to understand text.",
                metadata={"source": "test3.txt", "doc_id": 3},
            ),
        ]

    @pytest.fixture
    def mock_embeddings(self):
        """Create mock embeddings with correct dimension."""
        return [[0.1] * 1536 for _ in range(3)]

    @pytest.fixture
    def embedder(self, tmp_path):
        """Create EmbeddingGenerator instance with temp cache."""
        return EmbeddingGenerator(
            model_name="text-embedding-3-small",
            batch_size=2,
            enable_cache=True,
            cache_dir=str(tmp_path / "cache"),
        )

    @pytest.mark.asyncio
    async def test_basic_embedding_generation(
        self, embedder, sample_documents, mock_embeddings
    ):
        """Test basic embedding generation."""
        with patch.object(
            embedder, "_batch_embed", new_callable=AsyncMock
        ) as mock_embed:
            # Configure mock to return embeddings
            mock_embed.side_effect = [mock_embeddings[:2], mock_embeddings[2:]]

            # Generate embeddings
            results = await embedder.generate_embeddings(sample_documents)

            # Assertions
            assert len(results) == 3
            assert all(isinstance(doc, Document) for doc, _ in results)
            assert all(isinstance(emb, list) for _, emb in results)
            assert all(len(emb) == 1536 for _, emb in results)

            # Verify batching (3 docs with batch_size=2 = 2 batches)
            assert mock_embed.call_count == 2

    @pytest.mark.asyncio
    async def test_caching_functionality(
        self, embedder, sample_documents, mock_embeddings
    ):
        """Test that caching works correctly."""
        with patch.object(
            embedder, "_batch_embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = [mock_embeddings[:2], mock_embeddings[2:]]

            # First call - should hit API
            results1 = await embedder.generate_embeddings(sample_documents)
            first_call_count = mock_embed.call_count

            # Reset mock for second call
            mock_embed.reset_mock()

            # Second call - should use cache
            results2 = await embedder.generate_embeddings(sample_documents)

            # Assertions
            assert len(results1) == len(results2)
            assert mock_embed.call_count == 0  # No API calls due to cache
            assert embedder.stats["cache_hits"] == 3

    @pytest.mark.asyncio
    async def test_batch_processing(self, tmp_path):
        """Test batch processing with different batch sizes."""
        documents = [
            Document(page_content=f"Document {i}", metadata={"source": f"doc{i}.txt"})
            for i in range(10)
        ]

        embedder = EmbeddingGenerator(
            model_name="text-embedding-3-small",
            batch_size=3,
            enable_cache=False,
            cache_dir=str(tmp_path / "cache"),
        )

        mock_embedding = [0.1] * 1536

        with patch.object(
            embedder, "_batch_embed", new_callable=AsyncMock
        ) as mock_embed:
            # Return appropriate number of embeddings per batch
            mock_embed.side_effect = [
                [mock_embedding] * 3,  # Batch 1: 3 docs
                [mock_embedding] * 3,  # Batch 2: 3 docs
                [mock_embedding] * 3,  # Batch 3: 3 docs
                [mock_embedding] * 1,  # Batch 4: 1 doc
            ]

            results = await embedder.generate_embeddings(documents)

            # Assertions
            assert len(results) == 10
            # 10 docs with batch_size=3 = 4 batches
            assert mock_embed.call_count == 4

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self, embedder):
        """Test exponential backoff retry logic."""
        # Mock function that fails twice then succeeds
        mock_func = AsyncMock(
            side_effect=[Exception("Error 1"), Exception("Error 2"), "success"]
        )

        result = await embedder._retry_with_backoff(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3
        assert embedder.stats["retries"] == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion(self, embedder):
        """Test that retries eventually fail."""
        # Mock function that always fails
        mock_func = AsyncMock(side_effect=Exception("Permanent error"))

        with pytest.raises(Exception, match="Permanent error"):
            await embedder._retry_with_backoff(mock_func, max_retries=2)

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_validate_embeddings_success(self, embedder, sample_documents):
        """Test validation with correct embeddings."""
        embeddings = [(doc, [0.1] * 1536) for doc in sample_documents]

        validation = embedder.validate_embeddings(embeddings)

        assert validation["validation_passed"] is True
        assert validation["total_embeddings"] == 3
        assert validation["expected_dimension"] == 1536
        assert validation["dimension_valid"] is True
        assert validation["contains_nan"] is False
        assert validation["contains_inf"] is False

    def test_validate_embeddings_dimension_mismatch(self, embedder, sample_documents):
        """Test validation with wrong dimension."""
        embeddings = [
            (sample_documents[0], [0.1] * 1536),
            (sample_documents[1], [0.1] * 512),  # Wrong dimension
            (sample_documents[2], [0.1] * 1536),
        ]

        validation = embedder.validate_embeddings(embeddings)

        assert validation["validation_passed"] is False
        assert validation["dimension_valid"] is False
        assert len(validation["dimension_mismatches"]) == 1
        assert validation["dimension_mismatches"][0]["actual"] == 512

    def test_validate_embeddings_nan_values(self, embedder, sample_documents):
        """Test validation with NaN values."""
        embeddings = [
            (sample_documents[0], [0.1] * 1536),
            (sample_documents[1], [float("nan")] * 1536),  # Contains NaN
            (sample_documents[2], [0.1] * 1536),
        ]

        validation = embedder.validate_embeddings(embeddings)

        assert validation["validation_passed"] is False
        assert validation["contains_nan"] is True
        assert len(validation["invalid_embeddings"]) == 1

    def test_validate_embeddings_inf_values(self, embedder, sample_documents):
        """Test validation with infinite values."""
        embeddings = [
            (sample_documents[0], [0.1] * 1536),
            (sample_documents[1], [float("inf")] * 1536),  # Contains inf
            (sample_documents[2], [0.1] * 1536),
        ]

        validation = embedder.validate_embeddings(embeddings)

        assert validation["validation_passed"] is False
        assert validation["contains_inf"] is True
        assert len(validation["invalid_embeddings"]) == 1

    def test_content_hashing(self, embedder):
        """Test content hash generation."""
        text1 = "This is a test"
        text2 = "This is a test"
        text3 = "This is different"

        hash1 = embedder._get_content_hash(text1)
        hash2 = embedder._get_content_hash(text2)
        hash3 = embedder._get_content_hash(text3)

        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be SHA256 (64 hex chars)
        assert len(hash1) == 64

    def test_cache_check_and_update(self, embedder, sample_documents):
        """Test cache checking and updating."""
        embedding = [0.1] * 1536
        results = [(sample_documents[0], embedding)]

        # Update cache
        embedder._update_cache(results)

        # Check cache
        cached, uncached = embedder._check_cache(sample_documents[:1])

        assert len(cached) == 1
        assert len(uncached) == 0
        assert cached[0][1] == embedding

    def test_clear_cache(self, embedder, sample_documents):
        """Test cache clearing."""
        # Add some cached data
        embedding = [0.1] * 1536
        results = [(doc, embedding) for doc in sample_documents]
        embedder._update_cache(results)

        # Verify cache exists
        assert len(list(embedder.cache_dir.glob("*.json"))) == 3

        # Clear cache
        deleted = embedder.clear_cache()

        assert deleted == 3
        assert len(list(embedder.cache_dir.glob("*.json"))) == 0

    def test_get_stats(self, embedder):
        """Test statistics retrieval."""
        embedder.stats["cache_hits"] = 10
        embedder.stats["cache_misses"] = 5
        embedder.stats["total_processed"] = 5

        stats = embedder.get_stats()

        assert stats["cache_hits"] == 10
        assert stats["cache_misses"] == 5
        assert stats["total_processed"] == 5
        assert "cache_hit_rate" in stats
        assert stats["cache_hit_rate"] == "66.67%"  # 10/15

    @pytest.mark.asyncio
    async def test_empty_documents_list(self, embedder):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot generate embeddings"):
            await embedder.generate_embeddings([])

    @pytest.mark.asyncio
    async def test_failed_batch_handling(
        self, embedder, sample_documents, mock_embeddings
    ):
        """Test that failed batches don't crash entire pipeline."""
        with patch.object(
            embedder, "_retry_with_backoff", new_callable=AsyncMock
        ) as mock_retry:
            # First batch succeeds, second fails
            mock_retry.side_effect = [
                mock_embeddings[:2],
                Exception("Batch failed"),
            ]

            # Should not raise exception
            results = await embedder.generate_embeddings(sample_documents)

            # Should have results for all documents (with placeholders for failed)
            assert len(results) == 3

            # Failed embeddings should be tracked
            assert embedder.stats["failed_embeddings"] == 1

            # Last embedding should be placeholder (all zeros)
            assert results[2][1] == [0.0] * 1536

    @pytest.mark.asyncio
    async def test_rate_limiting_delay(self, embedder, sample_documents):
        """Test that rate limiting delays are applied."""
        with patch.object(
            embedder, "_batch_embed", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 1536] * 2

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await embedder.generate_embeddings(sample_documents)

                # With batch_size=2 and 3 docs, we have 2 batches
                # Sleep should be called once (between batches)
                assert mock_sleep.call_count == 1
                mock_sleep.assert_called_with(0.1)


class TestModelDimensions:
    """Test model dimension mapping."""

    def test_known_models(self):
        """Test dimension for known models."""
        embedder_small = EmbeddingGenerator(model_name="text-embedding-3-small")
        assert embedder_small.expected_dimension == 1536

        embedder_large = EmbeddingGenerator(model_name="text-embedding-3-large")
        assert embedder_large.expected_dimension == 3072

        embedder_ada = EmbeddingGenerator(model_name="text-embedding-ada-002")
        assert embedder_ada.expected_dimension == 1536

    def test_unknown_model(self):
        """Test default dimension for unknown models."""
        embedder = EmbeddingGenerator(model_name="unknown-model")
        assert embedder.expected_dimension == 1536  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
