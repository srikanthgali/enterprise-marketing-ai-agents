#!/usr/bin/env python3
"""
Quick test script for RAG retrieval validation.

Usage:
    python scripts/test_rag.py
    python -c "from scripts.test_rag import test_retrieval; test_retrieval()"
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config.settings import Settings
from src.marketing_agents.rag import AdvancedRetriever


async def test_retrieval_async():
    """Test RAG retrieval with sample queries."""
    settings = Settings()
    index_path = settings.data_dir / "embeddings" / "stripe_knowledge_base"

    if not index_path.exists():
        print(f"❌ Vector store not found at: {index_path}")
        print("Run 'python scripts/initialize_rag_pipeline.py' first.")
        return False

    print("\n" + "=" * 70)
    print("🧪 Testing RAG Retrieval")
    print("=" * 70 + "\n")

    try:
        # Load vector store
        print(f"Loading index from: {index_path}")
        embeddings = OpenAIEmbeddings(model=settings.vector_store.embedding_model)
        vector_store = FAISS.load_local(
            str(index_path), embeddings, allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded\n")

        # Initialize retriever
        retriever = AdvancedRetriever(vector_store)

        # Test queries
        test_queries = [
            "How do I handle payment webhooks?",
            "What is Stripe Connect?",
            "How to implement subscription billing?",
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            print("-" * 70)

            results = await retriever.retrieve(query, top_k=3, rerank=False)

            if results:
                for j, doc in enumerate(results, 1):
                    title = doc.metadata.get("title", "Untitled")
                    category = doc.metadata.get("category", "N/A")
                    preview = doc.page_content[:100].replace("\n", " ")
                    print(f"  {j}. {title}")
                    print(f"     Category: {category}")
                    print(f"     Preview: {preview}...")
            else:
                print("  No results found")

            print()

        print("=" * 70)
        print("✅ RAG retrieval test completed successfully!")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_retrieval():
    """Synchronous wrapper for test_retrieval_async."""
    return asyncio.run(test_retrieval_async())


if __name__ == "__main__":
    success = test_retrieval()
    sys.exit(0 if success else 1)
