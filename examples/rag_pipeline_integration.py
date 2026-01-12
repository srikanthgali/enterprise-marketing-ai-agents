"""
Integration example: Complete RAG pipeline with chunking.

This demonstrates how ChunkingStrategy integrates with:
1. DocumentIngestionPipeline (input)
2. EmbeddingGenerator (output - placeholder for now)
"""

import asyncio
import json
import logging
from pathlib import Path

from src.marketing_agents.rag.chunking import ChunkingStrategy
from src.marketing_agents.rag.document_ingestion import DocumentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Complete RAG pipeline integration."""

    # ========================================================================
    # STEP 1: Document Ingestion
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Ingesting Documents")
    logger.info("=" * 80)

    pipeline = DocumentIngestionPipeline(
        source_dir="data/raw/knowledge_base",
        source_type="stripe_docs",
    )

    documents = await pipeline.ingest_documents()
    logger.info(f"\n✓ Ingested {len(documents)} documents")

    # ========================================================================
    # STEP 2: Document Type Classification (Simplified)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Classifying Documents by Type")
    logger.info("=" * 80)

    # Group documents by type based on content/metadata
    doc_groups = {
        "api_reference": [],
        "tutorials": [],
        "technical_docs": [],
        "faq": [],
    }

    for doc in documents:
        # Simple classification based on content
        content = doc.page_content.lower()
        source = doc.metadata.get("source", "").lower()

        if "api" in source or "reference" in content[:200]:
            doc_groups["api_reference"].append(doc)
        elif "tutorial" in source or "how to" in content[:200]:
            doc_groups["tutorials"].append(doc)
        elif "faq" in source or "question" in content[:200]:
            doc_groups["faq"].append(doc)
        else:
            doc_groups["technical_docs"].append(doc)

    logger.info("\nDocument classification:")
    for doc_type, docs in doc_groups.items():
        logger.info(f"  {doc_type}: {len(docs)} documents")

    # ========================================================================
    # STEP 3: Chunk Documents with Type-Specific Strategies
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Chunking with Type-Specific Strategies")
    logger.info("=" * 80)

    all_chunks = []
    chunk_stats = {}

    for doc_type, docs in doc_groups.items():
        if not docs:
            continue

        # Create chunking strategy for this document type
        chunker = ChunkingStrategy.from_document_type(doc_type)

        # Chunk documents
        chunks = chunker.chunk_documents(docs)
        all_chunks.extend(chunks)

        # Validate and collect stats
        validation = chunker.validate_chunks(chunks)
        chunk_stats[doc_type] = validation

        logger.info(f"\n{doc_type}:")
        logger.info(
            f"  Strategy: chunk_size={chunker.chunk_size}, overlap={chunker.chunk_overlap}"
        )
        logger.info(f"  Chunks created: {validation['total_chunks']}")
        logger.info(f"  Avg chunk size: {validation['avg_chunk_size']:.0f} chars")
        logger.info(f"  Within target: {validation['within_target_pct']:.1f}%")
        if validation["oversized_chunks"]:
            logger.info(f"  ⚠️  Oversized chunks: {len(validation['oversized_chunks'])}")

    logger.info(f"\n✓ Total chunks created: {len(all_chunks)}")

    # ========================================================================
    # STEP 4: Analyze Chunk Distribution
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Chunk Distribution Analysis")
    logger.info("=" * 80)

    # Group chunks by size ranges
    size_ranges = {
        "0-500": 0,
        "501-1000": 0,
        "1001-1500": 0,
        "1501-2000": 0,
        "2000+": 0,
    }

    for chunk in all_chunks:
        size = len(chunk.page_content)
        if size <= 500:
            size_ranges["0-500"] += 1
        elif size <= 1000:
            size_ranges["501-1000"] += 1
        elif size <= 1500:
            size_ranges["1001-1500"] += 1
        elif size <= 2000:
            size_ranges["1501-2000"] += 1
        else:
            size_ranges["2000+"] += 1

    logger.info("\nChunk size distribution:")
    for range_name, count in size_ranges.items():
        pct = (count / len(all_chunks)) * 100
        logger.info(f"  {range_name} chars: {count} ({pct:.1f}%)")

    # ========================================================================
    # STEP 5: Save Sample Chunks for Inspection
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving Sample Chunks")
    logger.info("=" * 80)

    output_dir = Path("data/processed/sample_chunks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save 5 random chunks from each type
    import random

    samples = {}
    for doc_type in doc_groups.keys():
        type_chunks = [
            c
            for c in all_chunks
            if c.metadata.get("source", "").lower().find(doc_type.split("_")[0]) != -1
        ]
        if type_chunks:
            samples[doc_type] = random.sample(type_chunks, min(5, len(type_chunks)))

    for doc_type, sample_chunks in samples.items():
        output_file = output_dir / f"{doc_type}_samples.json"

        sample_data = [
            {
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_index": chunk.metadata["chunk_index"],
                "total_chunks": chunk.metadata["total_chunks"],
                "source": chunk.metadata.get("source", "unknown"),
                "content_length": len(chunk.page_content),
                "content_preview": chunk.page_content[:200] + "...",
            }
            for chunk in sample_chunks
        ]

        with open(output_file, "w") as f:
            json.dump(sample_data, f, indent=2)

        logger.info(f"  ✓ Saved {len(sample_data)} samples to {output_file}")

    # ========================================================================
    # STEP 6: Prepare for Embedding (Next Pipeline Stage)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Ready for Embedding Generation")
    logger.info("=" * 80)

    logger.info(f"\nChunks ready for embedding: {len(all_chunks)}")
    logger.info("\nNext steps:")
    logger.info("  1. Generate embeddings for all chunks")
    logger.info("  2. Store chunks + embeddings in vector store")
    logger.info("  3. Build retrieval index")
    logger.info("  4. Enable semantic search")

    # Calculate estimated embedding cost (OpenAI text-embedding-ada-002)
    total_tokens = sum(len(c.page_content) / 4 for c in all_chunks)  # Rough estimate
    estimated_cost = (total_tokens / 1000) * 0.0001  # $0.0001 per 1K tokens

    logger.info(f"\nEstimated embedding cost: ${estimated_cost:.4f}")
    logger.info(f"(Based on ~{total_tokens:.0f} tokens)")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n📄 Documents ingested: {len(documents)}")
    logger.info(f"✂️  Chunks created: {len(all_chunks)}")
    logger.info(f"📊 Document types: {len([g for g in doc_groups.values() if g])}")
    logger.info(f"💾 Sample chunks saved to: {output_dir}")

    logger.info("\nChunk quality metrics:")
    avg_size = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
    logger.info(f"  Average chunk size: {avg_size:.0f} chars")

    total_within_target = sum(s["within_target"] for s in chunk_stats.values())
    total_chunks_validated = sum(s["total_chunks"] for s in chunk_stats.values())
    within_target_pct = (total_within_target / total_chunks_validated) * 100
    logger.info(f"  Within target range: {within_target_pct:.1f}%")

    logger.info("\n✅ RAG Pipeline Integration Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
