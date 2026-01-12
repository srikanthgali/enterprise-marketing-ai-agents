"""
Integration example: Complete RAG pipeline with AdvancedRetriever.

This example demonstrates:
1. Document ingestion from raw files
2. Chunking with semantic strategies
3. Embedding generation with caching
4. Vector store creation (FAISS)
5. Advanced retrieval with filtering and reranking

This shows how all RAG components work together.
"""

import asyncio
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.marketing_agents.rag import (
    AdvancedRetriever,
    ChunkingStrategy,
    DocumentIngestionPipeline,
    EmbeddingGenerator,
)


async def main():
    """Run the complete RAG pipeline integration example."""
    print("\n" + "=" * 70)
    print("RAG Pipeline Integration Example with AdvancedRetriever")
    print("=" * 70 + "\n")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set.")
        print("    Set your API key to run this example:")
        print("    export OPENAI_API_KEY='your-key-here'\n")
        return

    # Step 1: Document Ingestion
    print("Step 1: Document Ingestion")
    print("-" * 70)

    # Use the knowledge base if available, otherwise create sample docs
    kb_path = "data/raw/knowledge_base"
    if Path(kb_path).exists():
        print(f"Ingesting documents from: {kb_path}")
        pipeline = DocumentIngestionPipeline(source_dir=kb_path)
        documents = await pipeline.ingest_documents()
        print(f"✓ Ingested {len(documents)} documents\n")
    else:
        print("Knowledge base not found, using sample documents...")
        from langchain_core.documents import Document

        documents = [
            Document(
                page_content=(
                    "# Customer Data Platforms Overview\n\n"
                    "A Customer Data Platform (CDP) is a software that aggregates and "
                    "organizes customer data across a variety of touchpoints. It creates "
                    "a unified customer database that is accessible to other systems.\n\n"
                    "CDPs collect data from multiple sources including websites, mobile apps, "
                    "email, social media, and CRM systems. This unified view enables better "
                    "personalization and customer engagement."
                ),
                metadata={
                    "source": "internal/cdp-overview.md",
                    "category": "Overview",
                    "title": "Customer Data Platforms Overview",
                    "timestamp": "2025-01-01T00:00:00",
                },
            ),
            Document(
                page_content=(
                    "# Marketing Automation Best Practices\n\n"
                    "Marketing automation streamlines repetitive tasks and improves efficiency. "
                    "Key best practices include:\n\n"
                    "1. Segment your audience based on behavior and demographics\n"
                    "2. Create personalized email workflows\n"
                    "3. Use lead scoring to prioritize prospects\n"
                    "4. Integrate with your CRM for seamless data flow\n"
                    "5. Monitor and optimize campaign performance"
                ),
                metadata={
                    "source": "internal/marketing-automation.md",
                    "category": "Guide",
                    "title": "Marketing Automation Best Practices",
                    "timestamp": "2024-12-15T00:00:00",
                },
            ),
            Document(
                page_content=(
                    "# Segment Tracking API\n\n"
                    "The Segment Tracking API allows you to record events that customers "
                    "perform in your application. The API accepts various event types:\n\n"
                    "- **Track**: Record actions users perform\n"
                    "- **Identify**: Record traits about a user\n"
                    "- **Page**: Record page views\n"
                    "- **Screen**: Record mobile screen views\n"
                    "- **Group**: Associate users with groups\n\n"
                    "Example track call:\n"
                    "```\n"
                    "analytics.track('Product Viewed', {\n"
                    "  product_id: '123',\n"
                    "  category: 'Electronics'\n"
                    "});\n"
                    "```"
                ),
                metadata={
                    "source": "segment.com/docs/connections/spec/track",
                    "category": "API Reference",
                    "title": "Segment Tracking API",
                    "timestamp": "2025-01-10T00:00:00",
                },
            ),
        ]
        print(f"✓ Created {len(documents)} sample documents\n")

    # Step 2: Chunking
    print("Step 2: Chunking Documents")
    print("-" * 70)

    chunker = ChunkingStrategy(
        strategy="semantic",
        chunk_size=512,
        chunk_overlap=50,
    )

    chunks = []
    for doc in documents[:3]:  # Process first 3 docs for demo
        doc_chunks = chunker.chunk_document(doc)
        chunks.extend(doc_chunks)

    print(f"✓ Created {len(chunks)} chunks from {len(documents[:3])} documents")
    print(
        f"  Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars\n"
    )

    # Step 3: Generate Embeddings
    print("Step 3: Generating Embeddings")
    print("-" * 70)

    embedder = EmbeddingGenerator(
        model_name="text-embedding-3-small",
        batch_size=50,
        enable_cache=True,
    )

    embedded_chunks = await embedder.generate_embeddings(chunks)

    validation = embedder.validate_embeddings(embedded_chunks)
    print(f"✓ Generated {validation['total_embeddings']} embeddings")
    print(f"  All embeddings valid: {validation['all_valid']}")
    print(f"  Embedding dimension: {validation['dimension']}\n")

    # Step 4: Create Vector Store
    print("Step 4: Creating FAISS Vector Store")
    print("-" * 70)

    # Prepare data for FAISS
    texts = [doc.page_content for doc, _ in embedded_chunks]
    embeddings_list = [emb for _, emb in embedded_chunks]
    metadatas = [doc.metadata for doc, _ in embedded_chunks]

    # Create embeddings wrapper
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"✓ Created FAISS vector store with {len(chunks)} documents\n")

    # Step 5: Initialize AdvancedRetriever
    print("Step 5: Initializing AdvancedRetriever")
    print("-" * 70)

    retriever = AdvancedRetriever(vector_store)
    stats = retriever.get_stats()

    print(f"✓ Retriever initialized")
    print(f"  Vector store type: {stats['vector_store_type']}")
    print(f"  Cache size: {stats['cache_size']}\n")

    # Step 6: Example Queries
    print("Step 6: Running Example Queries")
    print("=" * 70 + "\n")

    # Query 1: Basic semantic search
    print("Query 1: Basic Semantic Search")
    print("-" * 70)
    query1 = "What is a Customer Data Platform?"
    print(f"Query: {query1}\n")

    results1 = await retriever.retrieve(query1, top_k=2, rerank=False)

    for i, doc in enumerate(results1, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print(f"   Preview: {doc.page_content[:100]}...")
        print()

    # Query 2: With metadata filtering
    print("\nQuery 2: Search with Metadata Filtering")
    print("-" * 70)
    query2 = "API documentation"
    filters = {"category": "API Reference"}
    print(f"Query: {query2}")
    print(f"Filters: {filters}\n")

    results2 = await retriever.retrieve(
        query2, top_k=3, filter_metadata=filters, rerank=False
    )

    print(f"Found {len(results2)} results in 'API Reference' category:")
    for i, doc in enumerate(results2, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print()

    # Query 3: With reranking
    print("Query 3: Search with Reranking")
    print("-" * 70)
    query3 = "marketing automation email campaigns"
    print(f"Query: {query3}\n")

    results3 = await retriever.retrieve(query3, top_k=3, rerank=True)

    print(f"Reranked results:")
    for i, doc in enumerate(results3, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print(f"   Category: {doc.metadata.get('category', 'N/A')}")
        print()

    # Query 4: Hybrid search
    print("Query 4: Hybrid Search")
    print("-" * 70)
    query4 = "tracking events"
    print(f"Query: {query4}")
    print(f"Alpha: 0.6 (60% semantic, 40% keyword)\n")

    results4 = await retriever.hybrid_search(query4, top_k=3, alpha=0.6)

    print(f"Hybrid search results:")
    for i, doc in enumerate(results4, 1):
        print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
        print()

    # Query 5: Caching demonstration
    print("Query 5: Caching Demonstration")
    print("-" * 70)

    import time

    # First query (cold)
    query5 = "customer engagement strategies"
    print(f"Query: {query5}\n")

    print("First execution (cold)...")
    start = time.time()
    results5a = await retriever.retrieve(query5, top_k=2)
    time5a = time.time() - start
    print(f"✓ Retrieved {len(results5a)} results in {time5a:.3f}s")

    # Second query (cached)
    print("\nSecond execution (cached)...")
    start = time.time()
    results5b = await retriever.retrieve(query5, top_k=2)
    time5b = time.time() - start
    print(f"✓ Retrieved {len(results5b)} results in {time5b:.3f}s")

    speedup = time5a / time5b if time5b > 0 else float("inf")
    print(f"\nSpeedup: {speedup:.1f}x faster with caching")

    # Final stats
    final_stats = retriever.get_stats()
    print(f"\nFinal cache size: {final_stats['cache_size']}")

    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Documents ingested: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings generated: {len(embedded_chunks)}")
    print(f"Queries executed: 5")
    print(f"Cache entries: {final_stats['cache_size']}")
    print("\n✓ RAG pipeline integration complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
