#!/usr/bin/env python3
"""
Simple Incremental Marketing Content Embedder

Adds marketing templates to existing vector store without re-embedding existing content.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config.settings import Settings
from src.marketing_agents.rag import ChunkingStrategy, EmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main execution function."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("Incremental Marketing Content Embedder")
    logger.info("=" * 80)

    # Initialize
    settings = Settings()
    marketing_dir = PROJECT_ROOT / "data" / "raw" / "knowledge_base" / "marketing"
    vector_store_dir = PROJECT_ROOT / "data" / "embeddings" / "stripe_knowledge_base"

    # Step 1: Check for marketing files
    logger.info("\n[1/5] Checking for marketing content files...")

    if not marketing_dir.exists():
        logger.error(f"Marketing directory not found: {marketing_dir}")
        return 1

    marketing_files = list(marketing_dir.glob("*.md"))
    logger.info(f"Found {len(marketing_files)} marketing files:")
    for f in marketing_files:
        logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    if not marketing_files:
        logger.error("No marketing files found!")
        return 1

    # Step 2: Load and chunk marketing content
    logger.info("\n[2/5] Processing marketing content into chunks...")

    chunker = ChunkingStrategy(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for file in marketing_files:
        content = file.read_text(encoding="utf-8")
        doc = Document(
            page_content=content,
            metadata={
                "source": f"marketing/{file.name}",
                "type": "marketing_guide",
                "file_name": file.name,
                "category": "marketing",
            },
        )

        chunks = chunker.chunk_documents([doc])
        logger.info(f"  {file.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Generate embeddings (cache-aware)
    logger.info("\n[3/5] Generating embeddings (cache-aware)...")
    logger.info(f"Model: {settings.vector_store.embedding_model}")
    logger.info(f"Cache enabled: {settings.vector_store.embedding_cache_enabled}")

    embedder = EmbeddingGenerator(
        model_name=settings.vector_store.embedding_model,
        batch_size=settings.vector_store.embedding_batch_size,
        enable_cache=settings.vector_store.embedding_cache_enabled,
    )

    embedded_chunks = await embedder.generate_embeddings(all_chunks)

    # Get stats
    stats = embedder.get_stats()
    logger.info(f"Embedding generation complete:")
    logger.info(f"  - Cache hits: {stats['cache_hits']}")
    logger.info(f"  - Cache misses (new): {stats['cache_misses']}")
    logger.info(f"  - Cache hit rate: {stats['cache_hit_rate']}")

    # Step 4: Load existing vector store
    logger.info("\n[4/5] Loading existing vector store...")

    embeddings_model = OpenAIEmbeddings(
        model=settings.vector_store.embedding_model,
        openai_api_key=settings.api.openai_api_key.get_secret_value(),
    )

    if vector_store_dir.exists():
        try:
            vector_store = FAISS.load_local(
                str(vector_store_dir),
                embeddings_model,
                allow_dangerous_deserialization=True,
            )
            existing_count = vector_store.index.ntotal
            logger.info(f"Loaded existing index with {existing_count:,} documents")
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            logger.info("Creating new vector store...")
            docs = [chunk for chunk, _ in embedded_chunks]
            embs = [emb for _, emb in embedded_chunks]
            vector_store = FAISS.from_embeddings(
                list(zip([d.page_content for d in docs], embs)),
                embeddings_model,
                metadatas=[d.metadata for d in docs],
            )
            existing_count = 0
    else:
        logger.info("No existing index found, creating new one...")
        docs = [chunk for chunk, _ in embedded_chunks]
        embs = [emb for _, emb in embedded_chunks]
        vector_store = FAISS.from_embeddings(
            list(zip([d.page_content for d in docs], embs)),
            embeddings_model,
            metadatas=[d.metadata for d in docs],
        )
        existing_count = 0

    # Step 5: Add new documents
    logger.info("\n[5/5] Adding new documents to vector store...")

    if existing_count > 0:
        # Add to existing index
        docs = [chunk for chunk, _ in embedded_chunks]
        embs = [emb for _, emb in embedded_chunks]

        vector_store.add_embeddings(
            list(zip([d.page_content for d in docs], embs)),
            metadatas=[d.metadata for d in docs],
        )

    # Save updated vector store
    logger.info("Saving updated vector store...")
    vector_store.save_local(str(vector_store_dir))

    new_count = vector_store.index.ntotal
    logger.info(f"Vector store updated successfully:")
    logger.info(f"  - Previous documents: {existing_count:,}")
    logger.info(f"  - New documents added: {len(embedded_chunks)}")
    logger.info(f"  - Total documents: {new_count:,}")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✓ SUCCESS: Marketing content added successfully!")
    logger.info(f"Total execution time: {elapsed:.2f}s")
    logger.info("=" * 80)

    logger.info("\nNext Steps:")
    logger.info("1. Restart services to load new embeddings:")
    logger.info("   ./stop_all.sh && ./start_all.sh")
    logger.info("\n2. Test marketing queries:")
    logger.info("   - 'Create a marketing strategy for fintech startups'")
    logger.info("   - 'Plan a product launch campaign with $25K budget'")
    logger.info("   - 'What are B2B SaaS buyer personas?'")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
