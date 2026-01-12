"""
Production-grade embedding generation with batching, error handling, and caching.

This module provides robust embedding generation for RAG pipelines with:
- Batch processing for efficiency
- Exponential backoff retry logic
- Rate limiting and progress tracking
- Quality validation
- Optional caching
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from langchain_core.documents import Document
from openai import AsyncOpenAI, RateLimitError, APIError

from config.settings import Settings

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Production-grade embedding generator with batching, retries, and validation.

    Features:
    - Batch processing with configurable batch size
    - Rate limiting with delays between batches
    - Exponential backoff for failed requests
    - Quality validation (dimension, NaN/inf checks)
    - Optional content-based caching
    - Progress logging for long operations

    Example:
        embedder = EmbeddingGenerator(model_name="text-embedding-3-small")
        embedded_chunks = await embedder.generate_embeddings(chunks)
        validation = embedder.validate_embeddings(embedded_chunks)
    """

    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        enable_cache: Optional[bool] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: OpenAI embedding model name (defaults to config setting)
            batch_size: Number of documents to process per batch (defaults to config setting)
            enable_cache: Whether to cache embeddings by content hash (defaults to config setting)
            cache_dir: Directory for embedding cache (default: data/embeddings/cache/)
        """
        # Initialize settings first to get defaults from config
        self.settings = Settings()

        # Use config values as defaults if not explicitly provided
        self.model_name = model_name or self.settings.vector_store.embedding_model
        self.batch_size = batch_size or self.settings.vector_store.embedding_batch_size
        self.enable_cache = (
            enable_cache
            if enable_cache is not None
            else self.settings.vector_store.embedding_cache_enabled
        )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.settings.api.openai_api_key.get_secret_value(),
            organization=self.settings.api.openai_org_id,
            timeout=self.settings.api.openai_timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Expected embedding dimension
        self.expected_dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)

        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.settings.data_dir / "embeddings" / "cache"

        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at: {self.cache_dir}")

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "failed_embeddings": 0,
            "retries": 0,
        }

    async def generate_embeddings(
        self, chunks: List[Document]
    ) -> List[Tuple[Document, List[float]]]:
        """
        Generate embeddings for a list of document chunks.

        Args:
            chunks: List of Document objects to embed

        Returns:
            List of tuples containing (Document, embedding vector)

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot generate embeddings for empty chunks list")

        logger.info(f"Starting embedding generation for {len(chunks)} documents")
        logger.info(f"Model: {self.model_name}, Batch size: {self.batch_size}")

        start_time = time.time()
        results: List[Tuple[Document, List[float]]] = []

        # Process in batches
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(chunks), self.batch_size):
            batch_num = batch_idx // self.batch_size + 1
            batch_chunks = chunks[batch_idx : batch_idx + self.batch_size]

            # Check cache first
            cached_results, uncached_chunks = self._check_cache(batch_chunks)
            results.extend(cached_results)

            if uncached_chunks:
                # Extract texts from uncached documents
                texts = [doc.page_content for doc in uncached_chunks]

                try:
                    # Generate embeddings with retry logic
                    embeddings = await self._retry_with_backoff(
                        self._batch_embed, texts
                    )

                    # Combine documents with embeddings
                    batch_results = list(zip(uncached_chunks, embeddings))
                    results.extend(batch_results)

                    # Update cache
                    if self.enable_cache:
                        self._update_cache(batch_results)

                    self.stats["total_processed"] += len(uncached_chunks)

                except Exception as e:
                    logger.error(
                        f"Failed to generate embeddings for batch {batch_num} "
                        f"after max retries: {e}"
                    )
                    self.stats["failed_embeddings"] += len(uncached_chunks)

                    # Add placeholder embeddings for failed documents
                    # This prevents pipeline failure for single batch issues
                    for doc in uncached_chunks:
                        placeholder = [0.0] * self.expected_dimension
                        results.append((doc, placeholder))
                        logger.warning(
                            f"Using placeholder embedding for document: "
                            f"{doc.metadata.get('source', 'unknown')}"
                        )

            # Progress logging every 10 batches
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(
                    f"Progress: {batch_num}/{total_batches} batches "
                    f"({len(results)}/{len(chunks)} documents)"
                )

            # Rate limiting: 0.1s delay between batches
            if batch_idx + self.batch_size < len(chunks):
                await asyncio.sleep(0.1)

        elapsed = time.time() - start_time
        logger.info(
            f"Embedding generation complete: {len(results)} documents in "
            f"{elapsed:.2f}s ({len(results)/elapsed:.1f} docs/sec)"
        )
        logger.info(f"Stats: {self.stats}")

        return results

    async def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            APIError: If API call fails
        """
        response = await self.client.embeddings.create(
            model=self.model_name, input=texts, encoding_format="float"
        )

        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]

        return embeddings

    async def _retry_with_backoff(
        self, func, *args, max_retries: int = 3, **kwargs
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function execution

        Raises:
            Exception: If all retries are exhausted
        """
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except RateLimitError as e:
                if attempt == max_retries:
                    logger.error(f"Rate limit exceeded after {max_retries} retries")
                    raise

                # Exponential backoff: 2^attempt seconds
                wait_time = 2**attempt
                logger.warning(
                    f"Rate limit hit, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                self.stats["retries"] += 1
                await asyncio.sleep(wait_time)

            except APIError as e:
                if attempt == max_retries:
                    logger.error(f"API error after {max_retries} retries: {e}")
                    raise

                wait_time = 2**attempt
                logger.warning(
                    f"API error, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                self.stats["retries"] += 1
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Unexpected error in retry logic: {e}")
                raise

    def validate_embeddings(
        self, embeddings: List[Tuple[Document, List[float]]]
    ) -> Dict[str, Any]:
        """
        Validate embedding quality and consistency.

        Checks:
        - Embedding dimension matches expected
        - No NaN or infinite values
        - Embedding count matches document count

        Args:
            embeddings: List of (Document, embedding) tuples

        Returns:
            Dictionary with validation metrics and results
        """
        validation_results = {
            "total_embeddings": len(embeddings),
            "expected_dimension": self.expected_dimension,
            "dimension_valid": True,
            "contains_nan": False,
            "contains_inf": False,
            "dimension_mismatches": [],
            "invalid_embeddings": [],
            "validation_passed": True,
        }

        for idx, (doc, embedding) in enumerate(embeddings):
            # Check dimension
            if len(embedding) != self.expected_dimension:
                validation_results["dimension_valid"] = False
                validation_results["dimension_mismatches"].append(
                    {
                        "index": idx,
                        "source": doc.metadata.get("source", "unknown"),
                        "expected": self.expected_dimension,
                        "actual": len(embedding),
                    }
                )

            # Check for NaN or infinite values
            embedding_array = np.array(embedding)

            if np.any(np.isnan(embedding_array)):
                validation_results["contains_nan"] = True
                validation_results["invalid_embeddings"].append(
                    {
                        "index": idx,
                        "source": doc.metadata.get("source", "unknown"),
                        "issue": "Contains NaN values",
                    }
                )

            if np.any(np.isinf(embedding_array)):
                validation_results["contains_inf"] = True
                validation_results["invalid_embeddings"].append(
                    {
                        "index": idx,
                        "source": doc.metadata.get("source", "unknown"),
                        "issue": "Contains infinite values",
                    }
                )

        # Overall validation status
        validation_results["validation_passed"] = (
            validation_results["dimension_valid"]
            and not validation_results["contains_nan"]
            and not validation_results["contains_inf"]
        )

        # Log validation summary
        if validation_results["validation_passed"]:
            logger.info(
                f"✓ Validation passed: {validation_results['total_embeddings']} "
                f"embeddings, dimension {self.expected_dimension}"
            )
        else:
            logger.warning(
                f"✗ Validation issues found: "
                f"Dimension mismatches: {len(validation_results['dimension_mismatches'])}, "
                f"Invalid embeddings: {len(validation_results['invalid_embeddings'])}"
            )

        return validation_results

    def _get_content_hash(self, text: str) -> str:
        """
        Generate SHA256 hash of text content for caching.

        Args:
            text: Text content to hash

        Returns:
            Hex string of content hash
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _check_cache(
        self, chunks: List[Document]
    ) -> Tuple[List[Tuple[Document, List[float]]], List[Document]]:
        """
        Check cache for existing embeddings.

        Args:
            chunks: List of documents to check

        Returns:
            Tuple of (cached results, uncached documents)
        """
        if not self.enable_cache:
            return [], chunks

        cached_results = []
        uncached_chunks = []

        for doc in chunks:
            content_hash = self._get_content_hash(doc.page_content)
            cache_file = self.cache_dir / f"{content_hash}.json"

            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        cached_data = json.load(f)

                    # Verify model matches
                    if cached_data.get("model") == self.model_name:
                        embedding = cached_data["embedding"]
                        cached_results.append((doc, embedding))
                        self.stats["cache_hits"] += 1
                        continue

                except Exception as e:
                    logger.warning(f"Failed to load cache for {content_hash}: {e}")

            # Not in cache or cache load failed
            uncached_chunks.append(doc)
            self.stats["cache_misses"] += 1

        if cached_results:
            logger.debug(f"Cache hits: {len(cached_results)}/{len(chunks)}")

        return cached_results, uncached_chunks

    def _update_cache(self, results: List[Tuple[Document, List[float]]]) -> None:
        """
        Update cache with newly generated embeddings.

        Args:
            results: List of (Document, embedding) tuples to cache
        """
        if not self.enable_cache:
            return

        for doc, embedding in results:
            try:
                content_hash = self._get_content_hash(doc.page_content)
                cache_file = self.cache_dir / f"{content_hash}.json"

                cache_data = {
                    "model": self.model_name,
                    "embedding": embedding,
                    "dimension": len(embedding),
                    "timestamp": time.time(),
                    "source": doc.metadata.get("source", "unknown"),
                }

                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)

            except Exception as e:
                logger.warning(f"Failed to update cache: {e}")

    def clear_cache(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of cache files deleted
        """
        if not self.enable_cache or not self.cache_dir.exists():
            return 0

        deleted = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {deleted} cached embeddings")
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for embedding generation.

        Returns:
            Dictionary with statistics including cache performance
        """
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests if total_requests > 0 else 0.0
        )

        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.2%}",
            "model": self.model_name,
            "batch_size": self.batch_size,
            "cache_enabled": self.enable_cache,
        }


# Example usage
async def example_usage():
    """Example of how to use the EmbeddingGenerator."""
    from langchain_core.documents import Document

    # Create sample documents
    chunks = [
        Document(
            page_content=f"This is sample document {i}.",
            metadata={"source": f"doc_{i}.txt", "chunk_id": i},
        )
        for i in range(250)  # 250 docs = 3 batches
    ]

    # Initialize generator
    embedder = EmbeddingGenerator(model_name="text-embedding-3-small", batch_size=100)

    # Generate embeddings
    embedded_chunks = await embedder.generate_embeddings(chunks)

    # Validate results
    validation = embedder.validate_embeddings(embedded_chunks)
    print(f"\nValidation Results:")
    print(f"Total embeddings: {validation['total_embeddings']}")
    print(f"Embedding dimension: {validation['expected_dimension']}")
    print(f"Validation passed: {validation['validation_passed']}")

    # Get statistics
    stats = embedder.get_stats()
    print(f"\nGeneration Statistics:")
    print(f"Cache hit rate: {stats['cache_hit_rate']}")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Failed embeddings: {stats['failed_embeddings']}")
    print(f"Retries: {stats['retries']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
