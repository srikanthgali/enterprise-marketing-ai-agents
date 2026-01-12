"""
Example of using the EmbeddingGenerator in a RAG pipeline.

This demonstrates:
1. Loading and chunking documents
2. Generating embeddings with batching and retries
3. Validating embedding quality
4. Integration with vector store
"""

import asyncio
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.marketing_agents.rag.embedding_generator import EmbeddingGenerator
from src.marketing_agents.rag.chunking import ChunkingStrategy
from src.marketing_agents.rag.document_ingestion import DocumentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def basic_embedding_example():
    """Basic example: Generate embeddings for sample documents."""
    print("\n" + "=" * 60)
    print("Basic Embedding Generation Example")
    print("=" * 60 + "\n")

    # Create sample documents
    documents = [
        Document(
            page_content="Twilio Segment is a Customer Data Platform (CDP) that helps "
            "businesses collect, clean, and activate customer data.",
            metadata={"source": "segment_overview.txt", "doc_type": "technical"},
        ),
        Document(
            page_content="Use the Analytics.js library to collect user events from "
            "your website and send them to Segment.",
            metadata={"source": "analytics_js.txt", "doc_type": "api_reference"},
        ),
        Document(
            page_content="Segment supports over 400 integrations including Salesforce, "
            "Marketo, Google Analytics, and Facebook Ads.",
            metadata={"source": "integrations.txt", "doc_type": "technical"},
        ),
    ]

    # Initialize embedding generator
    embedder = EmbeddingGenerator(
        model_name="text-embedding-3-small", batch_size=100, enable_cache=True
    )

    # Generate embeddings
    embedded_chunks = await embedder.generate_embeddings(documents)

    # Validate embeddings
    validation = embedder.validate_embeddings(embedded_chunks)

    print(f"Generated {validation['total_embeddings']} embeddings")
    print(f"Dimension: {validation['expected_dimension']}")
    print(f"Validation passed: ✓" if validation["validation_passed"] else "✗")

    # Display statistics
    stats = embedder.get_stats()
    print(f"\nCache hit rate: {stats['cache_hit_rate']}")
    print(f"Total processed: {stats['total_processed']}")


async def full_pipeline_example():
    """Full pipeline: Ingest, chunk, embed, and store in vector database."""
    print("\n" + "=" * 60)
    print("Full RAG Pipeline Example")
    print("=" * 60 + "\n")

    # Step 1: Load documents
    print("Step 1: Loading documents...")
    pipeline = DocumentIngestionPipeline()

    # Load sample markdown files
    docs_dir = Path("data/raw/knowledge_base")
    if docs_dir.exists():
        documents = pipeline.ingest_directory(str(docs_dir), file_pattern="*.md")
        print(f"Loaded {len(documents)} documents")
    else:
        # Use sample documents if directory doesn't exist
        documents = [
            Document(
                page_content="Sample technical documentation content " * 20,
                metadata={"source": "sample.txt"},
            )
            for _ in range(10)
        ]
        print(f"Using {len(documents)} sample documents")

    # Step 2: Chunk documents
    print("\nStep 2: Chunking documents...")
    chunker = ChunkingStrategy(chunk_size=500, chunk_overlap=100)
    chunks = []

    for doc in documents[:5]:  # Limit to 5 docs for example
        doc_chunks = chunker.chunk_document(doc)
        chunks.extend(doc_chunks)

    print(f"Created {len(chunks)} chunks")

    # Step 3: Generate embeddings
    print("\nStep 3: Generating embeddings...")
    embedder = EmbeddingGenerator(
        model_name="text-embedding-3-small", batch_size=50, enable_cache=True
    )

    embedded_chunks = await embedder.generate_embeddings(chunks)

    # Step 4: Validate
    print("\nStep 4: Validating embeddings...")
    validation = embedder.validate_embeddings(embedded_chunks)

    if validation["validation_passed"]:
        print("✓ All embeddings validated successfully")
    else:
        print(f"✗ Validation issues:")
        print(f"  - Dimension mismatches: {len(validation['dimension_mismatches'])}")
        print(f"  - Invalid embeddings: {len(validation['invalid_embeddings'])}")

    # Step 5: Store in vector database (FAISS)
    print("\nStep 5: Storing in vector database...")

    # Prepare data for FAISS
    texts = [doc.page_content for doc, _ in embedded_chunks]
    embeddings = [emb for _, emb in embedded_chunks]
    metadatas = [doc.metadata for doc, _ in embedded_chunks]

    # Note: In production, you'd use OpenAIEmbeddings wrapper
    # For this example, we'll show the structure
    print(f"Ready to store {len(texts)} documents with embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Display statistics
    stats = embedder.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']}")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Failed embeddings: {stats['failed_embeddings']}")
    print(f"  Retries: {stats['retries']}")


async def batch_processing_example():
    """Demonstrate batch processing with large document set."""
    print("\n" + "=" * 60)
    print("Batch Processing Example")
    print("=" * 60 + "\n")

    # Create large document set
    num_docs = 500
    documents = [
        Document(
            page_content=f"This is document {i} with some meaningful content about "
            f"marketing automation and customer data platforms. "
            f"It includes information about analytics, segmentation, and personalization.",
            metadata={"source": f"doc_{i}.txt", "doc_id": i, "category": "marketing"},
        )
        for i in range(num_docs)
    ]

    print(f"Processing {num_docs} documents...")

    # Initialize with smaller batch size to demonstrate progress logging
    embedder = EmbeddingGenerator(
        model_name="text-embedding-3-small", batch_size=50, enable_cache=True
    )

    # Generate embeddings (will show progress every 10 batches)
    embedded_chunks = await embedder.generate_embeddings(documents)

    # Validate
    validation = embedder.validate_embeddings(embedded_chunks)
    print(f"\n✓ Generated and validated {validation['total_embeddings']} embeddings")

    # Show cache performance
    stats = embedder.get_stats()
    print(f"Cache performance: {stats['cache_hit_rate']}")

    # Run again to demonstrate caching
    print("\nRunning again to test cache...")
    embedder2 = EmbeddingGenerator(
        model_name="text-embedding-3-small", batch_size=50, enable_cache=True
    )

    embedded_chunks2 = await embedder2.generate_embeddings(documents)
    stats2 = embedder2.get_stats()

    print(f"Second run cache hit rate: {stats2['cache_hit_rate']}")
    print(f"API calls saved: {stats2['cache_hits']}")


async def error_handling_example():
    """Demonstrate error handling and retry logic."""
    print("\n" + "=" * 60)
    print("Error Handling Example")
    print("=" * 60 + "\n")

    # Create documents with various edge cases
    documents = [
        Document(page_content="Normal document", metadata={"source": "normal.txt"}),
        Document(page_content="", metadata={"source": "empty.txt"}),  # Empty content
        Document(
            page_content="A" * 100000, metadata={"source": "huge.txt"}
        ),  # Very long
        Document(
            page_content="Special chars: 你好 🚀 ñ é",
            metadata={"source": "unicode.txt"},
        ),
    ]

    print(f"Processing {len(documents)} documents with edge cases...")

    embedder = EmbeddingGenerator(
        model_name="text-embedding-3-small", batch_size=10, enable_cache=False
    )

    try:
        embedded_chunks = await embedder.generate_embeddings(documents)
        validation = embedder.validate_embeddings(embedded_chunks)

        print(f"✓ Successfully processed all documents")
        print(f"Validation passed: {validation['validation_passed']}")

        if validation["invalid_embeddings"]:
            print("\nInvalid embeddings found:")
            for invalid in validation["invalid_embeddings"]:
                print(f"  - {invalid['source']}: {invalid['issue']}")

    except Exception as e:
        print(f"✗ Error during processing: {e}")

    # Show retry statistics
    stats = embedder.get_stats()
    if stats["retries"] > 0:
        print(f"\nRetries performed: {stats['retries']}")
    if stats["failed_embeddings"] > 0:
        print(f"Failed embeddings: {stats['failed_embeddings']}")


async def main():
    """Run all examples."""
    try:
        await basic_embedding_example()
        await full_pipeline_example()
        await batch_processing_example()
        await error_handling_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
