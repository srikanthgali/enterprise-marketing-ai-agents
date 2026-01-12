"""Advanced retrieval system for RAG with semantic search, filtering, and reranking."""

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """
    Advanced retrieval system with semantic search, metadata filtering, and hybrid search.

    Features:
    - Semantic search using FAISS vector similarity
    - Metadata filtering (category, source, custom fields)
    - Result reranking with multiple strategies
    - Hybrid search (semantic + keyword)
    - Relevance scoring and deduplication
    """

    def __init__(self, vector_store: FAISS):
        """
        Initialize the advanced retriever.

        Args:
            vector_store: FAISS vector store containing embedded documents
        """
        self.vector_store = vector_store
        self.cache = {}  # Simple cache for repeated queries
        logger.info("AdvancedRetriever initialized with FAISS vector store")

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
    ) -> List[Document]:
        """
        Main retrieval method with semantic search, filtering, and optional reranking.

        Args:
            query: Search query string
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters (e.g., {"category": "API Reference"})
            rerank: Whether to apply reranking strategies

        Returns:
            List of Document objects ranked by relevance
        """
        logger.info(
            f"Retrieving documents for query: '{query}' (top_k={top_k}, rerank={rerank})"
        )

        # Check cache
        cache_key = f"{query}_{top_k}_{filter_metadata}_{rerank}"
        if cache_key in self.cache:
            logger.debug("Returning cached results")
            return self.cache[cache_key]

        # Step 1: Semantic search (fetch more candidates if reranking)
        fetch_k = top_k * 3 if rerank else top_k
        semantic_results = await self._semantic_search(query, k=fetch_k)

        # Extract documents and scores
        documents = [doc for doc, score in semantic_results]
        scores = {id(doc): score for doc, score in semantic_results}

        logger.debug(f"Semantic search returned {len(documents)} candidates")

        # Step 2: Apply metadata filters if provided
        if filter_metadata:
            documents = self._filter_by_metadata(documents, filter_metadata)
            logger.debug(f"After filtering: {len(documents)} documents remaining")

        # Step 3: Rerank results if enabled
        if rerank and documents:
            documents = await self._rerank_results(query, documents, scores)
            logger.debug("Reranking completed")

        # Step 4: Return top_k results
        final_results = documents[:top_k]

        # Cache results
        self.cache[cache_key] = final_results

        logger.info(f"Retrieved {len(final_results)} documents")
        return final_results

    async def _semantic_search(
        self, query: str, k: int
    ) -> List[Tuple[Document, float]]:
        """
        Perform semantic search using FAISS vector similarity.

        Args:
            query: Search query string
            k: Number of results to retrieve

        Returns:
            List of (Document, score) tuples ordered by similarity
        """
        try:
            # Use FAISS similarity search with scores
            # Note: FAISS returns distances, smaller is better for L2
            results = self.vector_store.similarity_search_with_score(query, k=k)

            # Convert distance to similarity score (normalize to 0-1 range)
            # For L2 distance: similarity = 1 / (1 + distance)
            scored_results = []
            for doc, distance in results:
                similarity_score = 1.0 / (1.0 + distance)
                scored_results.append((doc, similarity_score))

            logger.debug(f"Semantic search found {len(scored_results)} results")
            return scored_results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _filter_by_metadata(
        self, documents: List[Document], filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Filter documents by metadata conditions.

        Supports:
        - Exact match: {"category": "API Reference"}
        - Pattern match: {"source": "*stripe.com*"}
        - Multiple conditions (AND logic)

        Args:
            documents: List of documents to filter
            filters: Dictionary of metadata field -> value

        Returns:
            Filtered list of documents
        """
        if not filters:
            return documents

        filtered_docs = []

        for doc in documents:
            matches = True

            for key, value in filters.items():
                doc_value = doc.metadata.get(key)

                # Skip if metadata key doesn't exist
                if doc_value is None:
                    matches = False
                    break

                # Handle pattern matching with wildcards
                if isinstance(value, str) and "*" in value:
                    pattern = value.replace("*", ".*")
                    if not re.match(pattern, str(doc_value), re.IGNORECASE):
                        matches = False
                        break
                # Exact match
                elif doc_value != value:
                    matches = False
                    break

            if matches:
                filtered_docs.append(doc)

        logger.debug(
            f"Metadata filtering: {len(documents)} -> {len(filtered_docs)} documents"
        )
        return filtered_docs

    async def _rerank_results(
        self, query: str, documents: List[Document], semantic_scores: Dict[int, float]
    ) -> List[Document]:
        """
        Rerank documents using multiple strategies.

        Strategies:
        1. Query term overlap scoring
        2. Content length normalization
        3. Recency bonus (if timestamp available)
        4. Combined with semantic score (70% semantic, 30% rerank)

        Args:
            query: Original search query
            documents: List of documents to rerank
            semantic_scores: Dictionary mapping document id to semantic score

        Returns:
            Reranked list of documents
        """
        query_terms = set(query.lower().split())
        scored_docs = []

        for doc in documents:
            # Get base semantic score
            semantic_score = semantic_scores.get(id(doc), 0.5)

            # 1. Query term overlap
            content = doc.page_content.lower()
            term_matches = sum(1 for term in query_terms if term in content)
            term_score = term_matches / len(query_terms) if query_terms else 0

            # 2. Content length normalization (prefer moderate length)
            content_length = len(doc.page_content)
            ideal_length = 500  # Characters
            length_score = 1.0 - min(
                abs(content_length - ideal_length) / ideal_length, 1.0
            )

            # 3. Recency bonus (if timestamp metadata exists)
            recency_score = 0.0
            if "timestamp" in doc.metadata:
                try:
                    doc_time = datetime.fromisoformat(doc.metadata["timestamp"])
                    days_old = (datetime.now() - doc_time).days
                    # Decay over 365 days
                    recency_score = max(0, 1.0 - (days_old / 365))
                except Exception:
                    pass

            # Combine scores: 70% semantic, 15% term overlap, 10% length, 5% recency
            final_score = (
                0.70 * semantic_score
                + 0.15 * term_score
                + 0.10 * length_score
                + 0.05 * recency_score
            )

            scored_docs.append((doc, final_score))

        # Sort by final score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Log score distribution
        if scored_docs:
            avg_score = sum(score for _, score in scored_docs) / len(scored_docs)
            logger.debug(f"Reranking complete. Average score: {avg_score:.3f}")

        return [doc for doc, _ in scored_docs]

    async def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> List[Document]:
        """
        Hybrid search combining semantic and keyword search.

        Args:
            query: Search query string
            top_k: Number of top results to return
            alpha: Weight for semantic search (0-1)
                   0.0 = pure keyword, 1.0 = pure semantic, 0.5 = equal weight

        Returns:
            List of Document objects ranked by combined score
        """
        logger.info(
            f"Hybrid search for query: '{query}' (top_k={top_k}, alpha={alpha})"
        )

        # Validate alpha
        alpha = max(0.0, min(1.0, alpha))

        # Run both searches in parallel
        fetch_k = top_k * 2  # Fetch more candidates

        semantic_task = self._semantic_search(query, k=fetch_k)
        keyword_task = asyncio.create_task(
            asyncio.to_thread(self._keyword_search, query, fetch_k)
        )

        semantic_results = await semantic_task
        keyword_results = await keyword_task

        logger.debug(
            f"Semantic: {len(semantic_results)} results, Keyword: {len(keyword_results)} results"
        )

        # Merge and deduplicate results
        merged = self._merge_results(semantic_results, keyword_results, alpha)

        # Return top_k
        final_results = merged[:top_k]
        logger.info(f"Hybrid search returned {len(final_results)} documents")

        return final_results

    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """
        Simple keyword-based search using term matching.

        Scores documents by:
        - Term frequency
        - Normalized by document length

        Future: Integrate BM25 or Elasticsearch for production.

        Args:
            query: Search query string
            k: Number of results to retrieve

        Returns:
            List of (Document, score) tuples ordered by keyword relevance
        """
        query_terms = query.lower().split()
        scored_docs = []

        # Get all documents from vector store
        # Note: This is inefficient for large stores - use dedicated search engine in production
        try:
            # Fetch more documents for keyword search
            all_docs = self.vector_store.similarity_search(query, k=k * 5)
        except Exception as e:
            logger.warning(f"Keyword search fallback failed: {e}")
            return []

        for doc in all_docs:
            content = doc.page_content.lower()

            # Calculate term frequency
            term_freq = 0
            for term in query_terms:
                term_freq += content.count(term)

            # Normalize by document length (avoid division by zero)
            doc_length = len(content.split())
            normalized_score = term_freq / max(doc_length, 1) if doc_length > 0 else 0

            # Boost if query appears as phrase
            if query.lower() in content:
                normalized_score *= 1.5

            if normalized_score > 0:
                scored_docs.append((doc, normalized_score))

        # Sort by score (descending) and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        logger.debug(f"Keyword search found {len(scored_docs)} matching documents")

        return scored_docs[:k]

    def _merge_results(
        self,
        semantic: List[Tuple[Document, float]],
        keyword: List[Tuple[Document, float]],
        alpha: float,
    ) -> List[Document]:
        """
        Merge and deduplicate results from semantic and keyword search.

        Uses Reciprocal Rank Fusion (RRF) combined with weighted scores.

        Args:
            semantic: List of (Document, score) from semantic search
            keyword: List of (Document, score) from keyword search
            alpha: Weight for semantic search (1-alpha for keyword)

        Returns:
            Merged and deduplicated list of documents ordered by combined score
        """
        # Create score dictionaries using document content as key (for deduplication)
        doc_scores = defaultdict(lambda: {"semantic": 0.0, "keyword": 0.0, "doc": None})

        # Normalize and add semantic scores
        if semantic:
            max_sem_score = max(score for _, score in semantic) if semantic else 1.0
            for rank, (doc, score) in enumerate(semantic, 1):
                doc_key = doc.page_content  # Use content as unique key
                normalized_score = score / max_sem_score if max_sem_score > 0 else 0
                # Add reciprocal rank fusion
                rrf_score = 1.0 / (rank + 60)  # k=60 is standard
                doc_scores[doc_key]["semantic"] = (
                    normalized_score * 0.7 + rrf_score * 0.3
                )
                doc_scores[doc_key]["doc"] = doc

        # Normalize and add keyword scores
        if keyword:
            max_kw_score = max(score for _, score in keyword) if keyword else 1.0
            for rank, (doc, score) in enumerate(keyword, 1):
                doc_key = doc.page_content
                normalized_score = score / max_kw_score if max_kw_score > 0 else 0
                rrf_score = 1.0 / (rank + 60)
                doc_scores[doc_key]["keyword"] = (
                    normalized_score * 0.7 + rrf_score * 0.3
                )
                if doc_scores[doc_key]["doc"] is None:
                    doc_scores[doc_key]["doc"] = doc

        # Combine scores with alpha weighting
        combined_results = []
        for doc_key, scores in doc_scores.items():
            combined_score = (
                alpha * scores["semantic"] + (1 - alpha) * scores["keyword"]
            )
            combined_results.append((scores["doc"], combined_score))

        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Merged {len(semantic)} semantic + {len(keyword)} keyword -> "
            f"{len(combined_results)} unique documents"
        )

        return [doc for doc, _ in combined_results]

    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
        logger.debug("Query cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics.

        Returns:
            Dictionary with cache size and other metrics
        """
        return {
            "cache_size": len(self.cache),
            "vector_store_type": type(self.vector_store).__name__,
        }
