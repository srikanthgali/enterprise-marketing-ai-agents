"""
Example demonstrating the AdvancedRetriever for sophisticated document retrieval.

This example shows:
1. Basic semantic search with FAISS
2. Metadata filtering (category, source)
3. Result reranking with multiple strategies
4. Hybrid search (semantic + keyword)
5. Query caching and performance optimization
"""

import asyncio
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.marketing_agents.rag import AdvancedRetriever


async def create_sample_vector_store() -> FAISS:
    """
    Create a sample FAISS vector store with marketing documents.

    In production, this would be loaded from persisted storage.
    """
    print("Creating sample vector store...")

    # Sample documents with rich metadata
    documents = [
        Document(
            page_content=(
                "Stripe provides comprehensive APIs for payment processing. "
                "The Payment Intents API handles the lifecycle of a payment, "
                "from creation to completion. It supports multiple payment methods "
                "and currencies."
            ),
            metadata={
                "category": "API Reference",
                "source": "stripe.com/docs/api/payment_intents",
                "title": "Payment Intents API",
                "timestamp": "2025-01-01T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Handling refunds in Stripe is straightforward. You can issue full "
                "or partial refunds through the Refunds API. Refunds are processed "
                "asynchronously and typically complete within 5-10 business days "
                "depending on the bank."
            ),
            metadata={
                "category": "API Reference",
                "source": "stripe.com/docs/api/refunds",
                "title": "Refunds API",
                "timestamp": "2025-01-05T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Subscription billing allows you to charge customers on a recurring "
                "basis. Create subscription plans with different billing intervals "
                "(monthly, yearly, etc.). Stripe handles automated invoicing, "
                "payment retries, and customer notifications."
            ),
            metadata={
                "category": "Guide",
                "source": "stripe.com/docs/billing/subscriptions",
                "title": "Subscription Billing Guide",
                "timestamp": "2024-12-15T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Webhooks notify your application when events happen in your Stripe "
                "account. Set up webhook endpoints to receive real-time updates about "
                "payments, refunds, subscriptions, and more. Always verify webhook "
                "signatures for security."
            ),
            metadata={
                "category": "Guide",
                "source": "stripe.com/docs/webhooks",
                "title": "Webhooks Guide",
                "timestamp": "2024-11-20T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Customer Data Platforms (CDPs) help unify customer data from multiple "
                "sources. Segment is a leading CDP that collects, cleans, and routes "
                "customer data to various tools. It integrates with analytics, marketing "
                "automation, and data warehouses."
            ),
            metadata={
                "category": "Overview",
                "source": "segment.com/docs/intro",
                "title": "What is Segment?",
                "timestamp": "2025-01-10T00:00:00",
            },
        ),
        Document(
            page_content=(
                "Marketing automation streamlines repetitive tasks like email campaigns, "
                "social media posting, and lead nurturing. By automating workflows, "
                "marketing teams can focus on strategy and creative work. Integration "
                "with CRM systems enables personalized customer journeys."
            ),
            metadata={
                "category": "Overview",
                "source": "internal/marketing-automation",
                "title": "Marketing Automation Overview",
                "timestamp": "2024-10-01T00:00:00",
            },
        ),
    ]

    # Create embeddings (requires OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)

    print(f"✓ Created vector store with {len(documents)} documents\n")
    return vector_store


async def basic_semantic_search_example(retriever: AdvancedRetriever):
    """Demonstrate basic semantic search."""
    print("=" * 70)
    print("Example 1: Basic Semantic Search")
    print("=" * 70 + "\n")

    query = "How do I handle payment refunds?"
    print(f"Query: {query}\n")

    results = await retriever.retrieve(query, top_k=3, rerank=False)

    print(f"Found {len(results)} results:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print(f"   Content: {doc.page_content[:100]}...")
        print()


async def metadata_filtering_example(retriever: AdvancedRetriever):
    """Demonstrate metadata filtering."""
    print("=" * 70)
    print("Example 2: Semantic Search with Metadata Filtering")
    print("=" * 70 + "\n")

    query = "How does Stripe handle refunds?"
    filters = {"category": "API Reference"}

    print(f"Query: {query}")
    print(f"Filters: {filters}\n")

    results = await retriever.retrieve(
        query, top_k=5, filter_metadata=filters, rerank=False
    )

    print(f"Found {len(results)} results in 'API Reference' category:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
        print()


async def reranking_example(retriever: AdvancedRetriever):
    """Demonstrate reranking with multiple strategies."""
    print("=" * 70)
    print("Example 3: Semantic Search with Reranking")
    print("=" * 70 + "\n")

    query = "subscription billing recurring payments"

    print(f"Query: {query}\n")
    print("Comparing results with and without reranking...\n")

    # Without reranking
    print("Without Reranking:")
    results_no_rerank = await retriever.retrieve(query, top_k=3, rerank=False)
    for i, doc in enumerate(results_no_rerank, 1):
        print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")

    print("\nWith Reranking:")
    results_rerank = await retriever.retrieve(query, top_k=3, rerank=True)
    for i, doc in enumerate(results_rerank, 1):
        print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")

    print("\n✓ Reranking considers query term overlap, content length, and recency\n")


async def hybrid_search_example(retriever: AdvancedRetriever):
    """Demonstrate hybrid search (semantic + keyword)."""
    print("=" * 70)
    print("Example 4: Hybrid Search (Semantic + Keyword)")
    print("=" * 70 + "\n")

    query = "subscription billing"

    print(f"Query: {query}\n")

    # Pure semantic (alpha=1.0)
    print("Pure Semantic Search (alpha=1.0):")
    semantic_results = await retriever.hybrid_search(query, top_k=3, alpha=1.0)
    for i, doc in enumerate(semantic_results, 1):
        print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")

    # Balanced (alpha=0.5)
    print("\nBalanced Hybrid (alpha=0.5):")
    hybrid_results = await retriever.hybrid_search(query, top_k=3, alpha=0.5)
    for i, doc in enumerate(hybrid_results, 1):
        print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")

    # Pure keyword (alpha=0.0)
    print("\nPure Keyword Search (alpha=0.0):")
    keyword_results = await retriever.hybrid_search(query, top_k=3, alpha=0.0)
    for i, doc in enumerate(keyword_results, 1):
        print(f"  {i}. {doc.metadata.get('title', 'Untitled')}")

    print("\n✓ Alpha parameter controls semantic vs keyword weighting\n")


async def pattern_matching_example(retriever: AdvancedRetriever):
    """Demonstrate pattern matching in metadata filters."""
    print("=" * 70)
    print("Example 5: Pattern Matching in Metadata Filters")
    print("=" * 70 + "\n")

    query = "API documentation"
    filters = {"source": "*stripe.com*"}  # Wildcard pattern

    print(f"Query: {query}")
    print(f"Filters: {filters} (using wildcard pattern)\n")

    results = await retriever.retrieve(query, top_k=10, filter_metadata=filters)

    print(f"Found {len(results)} results from stripe.com:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
        print()


async def caching_example(retriever: AdvancedRetriever):
    """Demonstrate query caching."""
    print("=" * 70)
    print("Example 6: Query Caching")
    print("=" * 70 + "\n")

    query = "payment processing API"

    import time

    # First query (cold)
    print("First query (cold)...")
    start = time.time()
    results1 = await retriever.retrieve(query, top_k=3)
    time1 = time.time() - start
    print(f"✓ Retrieved {len(results1)} results in {time1:.3f}s\n")

    # Second query (cached)
    print("Second identical query (cached)...")
    start = time.time()
    results2 = await retriever.retrieve(query, top_k=3)
    time2 = time.time() - start
    print(f"✓ Retrieved {len(results2)} results in {time2:.3f}s\n")

    print(f"Speedup: {time1/time2:.1f}x faster with caching")

    # Show stats
    stats = retriever.get_stats()
    print(f"\nRetriever stats: {stats}\n")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Advanced Retriever Examples")
    print("=" * 70 + "\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set.")
        print("    Set your API key to run this example:")
        print("    export OPENAI_API_KEY='your-key-here'\n")
        return

    # Create vector store
    vector_store = await create_sample_vector_store()

    # Initialize retriever
    retriever = AdvancedRetriever(vector_store)

    # Run examples
    await basic_semantic_search_example(retriever)
    await metadata_filtering_example(retriever)
    await reranking_example(retriever)
    await hybrid_search_example(retriever)
    await pattern_matching_example(retriever)
    await caching_example(retriever)

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
