"""
Knowledge Base Search Tool for searching Stripe documentation using vector similarity.

Provides semantic search capabilities over the Stripe knowledge base.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.settings import get_settings
from src.marketing_agents.rag.retriever import AdvancedRetriever

logger = logging.getLogger(__name__)


class KnowledgeBaseSearchTool:
    """
    Knowledge base search tool using vector similarity.

    Capabilities:
    - Semantic search over Stripe documentation
    - Metadata filtering (category, source)
    - Relevance-ranked results
    - Contextual document retrieval
    """

    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize knowledge base search tool.

        Args:
            vector_store_path: Optional path to FAISS vector store (uses default if None)
        """
        self.settings = get_settings()

        # Set vector store path
        if vector_store_path:
            self.vector_store_path = Path(vector_store_path)
        else:
            self.vector_store_path = (
                self.settings.base_dir
                / self.settings.vector_store.persist_directory
                / "stripe_knowledge_base"
            )

        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[AdvancedRetriever] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None

        # Initialize on first use
        self._initialized = False

    def _initialize(self) -> bool:
        """
        Lazy initialization of vector store and retriever.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.vector_store.embedding_model
            )

            # Load FAISS vector store
            if self.vector_store_path.exists():
                logger.info(f"Loading vector store from {self.vector_store_path}")
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                # Initialize advanced retriever
                self.retriever = AdvancedRetriever(self.vector_store)

                self._initialized = True
                logger.info("Knowledge base search tool initialized successfully")
                return True
            else:
                logger.warning(
                    f"Vector store not found at {self.vector_store_path}. "
                    "Please run the document ingestion pipeline first."
                )
                return False

        except Exception as e:
            logger.error(f"Failed to initialize knowledge base search tool: {e}")
            return False

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Search knowledge base using semantic similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return (default: 5)
            filter_metadata: Optional metadata filters (e.g., {"category": "API"})
            min_score: Minimum relevance score threshold (0.0-1.0)

        Returns:
            Dictionary containing search results with metadata
        """
        if not self._initialize():
            return {
                "success": False,
                "error": "Knowledge base not initialized",
                "results": [],
            }

        try:
            # Use advanced retriever for semantic search
            documents = await self.retriever.retrieve(
                query=query, top_k=top_k, filter_metadata=filter_metadata, rerank=True
            )

            # Structure results
            results = []
            for doc in documents:
                # Calculate relevance score from metadata if available
                score = doc.metadata.get("score", 1.0)

                if score >= min_score:
                    results.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": score,
                            "source": doc.metadata.get("source", "unknown"),
                        }
                    )

            logger.info(
                f"Knowledge base search successful: {len(results)} results for '{query}'"
            )

            return {
                "success": True,
                "query": query,
                "num_results": len(results),
                "results": results,
                "filter_applied": filter_metadata is not None,
            }

        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    async def search_by_category(
        self, query: str, category: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search knowledge base filtered by category.

        Args:
            query: Search query string
            category: Category to filter by (e.g., "API Reference", "Guides")
            top_k: Number of results to return

        Returns:
            Dictionary with category-filtered search results
        """
        return await self.search(
            query=query, top_k=top_k, filter_metadata={"category": category}
        )

    async def search_multiple_queries(
        self, queries: List[str], top_k_per_query: int = 3, deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Search knowledge base with multiple queries and combine results.

        Args:
            queries: List of search queries
            top_k_per_query: Number of results per query
            deduplicate: Whether to remove duplicate results

        Returns:
            Dictionary with combined search results
        """
        if not self._initialize():
            return {
                "success": False,
                "error": "Knowledge base not initialized",
                "results": [],
            }

        all_results = []
        seen_sources = set() if deduplicate else None

        for query in queries:
            search_result = await self.search(query, top_k=top_k_per_query)

            if search_result["success"]:
                for result in search_result["results"]:
                    source = result.get("source", "")

                    # Deduplicate by source if enabled
                    if deduplicate:
                        if source not in seen_sources:
                            all_results.append(result)
                            seen_sources.add(source)
                    else:
                        all_results.append(result)

        return {
            "success": True,
            "queries": queries,
            "num_results": len(all_results),
            "results": all_results,
        }

    async def search_stripe_product_info(
        self, product: str, aspect: str = "overview"
    ) -> Dict[str, Any]:
        """
        Search for Stripe product-specific information.

        Args:
            product: Stripe product name (e.g., "Payments", "Billing", "Connect")
            aspect: Specific aspect to search for (e.g., "features", "pricing", "integration")

        Returns:
            Dictionary with product information
        """
        query = f"Stripe {product} {aspect}"

        # Search with product-specific filters if available
        result = await self.search(
            query=query,
            top_k=5,
            filter_metadata={"product": product} if product else None,
        )

        return {"product": product, "aspect": aspect, "search_result": result}

    async def get_context_for_campaign(
        self, campaign_type: str, target_audience: str, objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge base context for a marketing campaign.

        Args:
            campaign_type: Type of campaign (e.g., "product launch", "feature announcement")
            target_audience: Target audience description
            objectives: List of campaign objectives

        Returns:
            Dictionary with relevant contextual information
        """
        # Build comprehensive queries
        queries = [
            f"Stripe {campaign_type} best practices",
            f"marketing to {target_audience}",
        ]

        # Add objective-specific queries
        for objective in objectives[:2]:  # Limit to top 2 objectives
            queries.append(f"Stripe {objective}")

        # Search with multiple queries
        result = await self.search_multiple_queries(
            queries=queries, top_k_per_query=3, deduplicate=True
        )

        return {
            "campaign_type": campaign_type,
            "target_audience": target_audience,
            "objectives": objectives,
            "knowledge_base_context": result,
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        if not self._initialize():
            return {"initialized": False, "error": "Knowledge base not initialized"}

        try:
            # Get vector store statistics
            index_size = len(self.vector_store.index_to_docstore_id)

            return {
                "initialized": True,
                "index_size": index_size,
                "embedding_model": self.settings.vector_store.embedding_model,
                "vector_store_path": str(self.vector_store_path),
            }
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {"initialized": True, "error": str(e)}
