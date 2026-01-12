"""Example script demonstrating document chunking with ChunkingStrategy."""

import asyncio
import logging
from pathlib import Path

from marketing_agents.rag.chunking import ChunkingStrategy
from marketing_agents.rag.document_ingestion import DocumentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate document ingestion and intelligent chunking."""

    # Step 1: Ingest documents
    logger.info("=" * 80)
    logger.info("STEP 1: Document Ingestion")
    logger.info("=" * 80)

    pipeline = DocumentIngestionPipeline(
        source_dir="data/raw/knowledge_base",
        source_type="stripe_docs",
    )

    documents = await pipeline.ingest_documents()
    logger.info(f"\nIngested {len(documents)} documents")

    if not documents:
        logger.warning("No documents found to chunk. Exiting.")
        return

    # Step 2: Chunk documents with default strategy
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Default Chunking Strategy")
    logger.info("=" * 80)

    default_chunker = ChunkingStrategy(chunk_size=1000, chunk_overlap=200)
    default_chunks = default_chunker.chunk_documents(documents[:5])  # First 5 docs

    logger.info(f"\nCreated {len(default_chunks)} chunks with default strategy")

    # Validate chunks
    validation = default_chunker.validate_chunks(default_chunks)
    logger.info("\nValidation Results:")
    logger.info(f"  Total chunks: {validation['total_chunks']}")
    logger.info(f"  Avg chunk size: {validation['avg_chunk_size']:.0f} chars")
    logger.info(
        f"  Size range: [{validation['min_chunk_size']}, {validation['max_chunk_size']}]"
    )
    logger.info(f"  Within target range: {validation['within_target_pct']:.1f}%")
    logger.info(f"  Oversized chunks: {len(validation['oversized_chunks'])}")

    # Step 3: Try different document type strategies
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Document Type-Specific Strategies")
    logger.info("=" * 80)

    strategies = {
        "technical_docs": ChunkingStrategy.from_document_type("technical_docs"),
        "api_reference": ChunkingStrategy.from_document_type("api_reference"),
        "tutorials": ChunkingStrategy.from_document_type("tutorials"),
        "faq": ChunkingStrategy.from_document_type("faq"),
    }

    for doc_type, chunker in strategies.items():
        chunks = chunker.chunk_documents(documents[:2])  # First 2 docs
        validation = chunker.validate_chunks(chunks)

        logger.info(f"\n{doc_type.upper()} Strategy:")
        logger.info(
            f"  Chunk size: {chunker.chunk_size}, Overlap: {chunker.chunk_overlap}"
        )
        logger.info(f"  Total chunks: {validation['total_chunks']}")
        logger.info(f"  Avg size: {validation['avg_chunk_size']:.0f} chars")
        logger.info(f"  Within target: {validation['within_target_pct']:.1f}%")

    # Step 4: Examine sample chunks
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Sample Chunk Inspection")
    logger.info("=" * 80)

    if default_chunks:
        sample_chunk = default_chunks[0]
        logger.info(f"\nSample Chunk Metadata:")
        logger.info(f"  Chunk ID: {sample_chunk.metadata['chunk_id']}")
        logger.info(f"  Chunk Index: {sample_chunk.metadata['chunk_index']}")
        logger.info(f"  Total Chunks: {sample_chunk.metadata['total_chunks']}")
        logger.info(f"  Source: {sample_chunk.metadata.get('source', 'unknown')}")
        logger.info(f"  Content Length: {sample_chunk.metadata['chunk_size']} chars")
        logger.info(f"\nFirst 200 chars of content:")
        logger.info(f"  {sample_chunk.page_content[:200]}...")

    # Step 5: Check for oversized chunks
    if validation["oversized_chunks"]:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Oversized Chunks Report")
        logger.info("=" * 80)

        for oversized in validation["oversized_chunks"][:5]:  # Show first 5
            logger.info(f"\n  Chunk ID: {oversized['chunk_id']}")
            logger.info(f"  Size: {oversized['size']} chars")
            logger.info(f"  Source: {oversized['source']}")

    logger.info("\n" + "=" * 80)
    logger.info("Chunking demonstration complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
