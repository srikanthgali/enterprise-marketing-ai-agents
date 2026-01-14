#!/usr/bin/env python3
"""
Incremental Marketing Content Addition and Embedding Script

This script:
1. Verifies existing embedding cache (6,239 chunks from Stripe docs)
2. Processes new marketing templates (only new content)
3. Generates embeddings ONLY for new content (cache prevents re-embedding)
4. Updates FAISS index incrementally
5. Validates cache performance and chunk coverage

Usage:
    python scripts/add_marketing_content_incremental.py --embed
    python scripts/add_marketing_content_incremental.py --validate-only
    python scripts/add_marketing_content_incremental.py --no-embed
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from src.marketing_agents.rag import (
    ChunkingStrategy,
    EmbeddingGenerator,
)
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IncrementalMarketingEmbedder:
    """
    Incrementally add marketing content and embeddings without re-processing existing data.
    """

    def __init__(self):
        """Initialize with settings and components."""
        self.settings = get_settings()
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingGenerator(enable_cache=True)  # Cache enabled
        self.vector_store = None

        # Paths
        self.marketing_dir = (
            PROJECT_ROOT / "data" / "raw" / "knowledge_base" / "marketing"
        )
        self.cache_dir = PROJECT_ROOT / "data" / "embeddings" / "cache"
        self.vector_store_dir = (
            PROJECT_ROOT / "data" / "embeddings" / "stripe_knowledge_base"
        )

        # Statistics
        self.stats = {
            "existing_cache_files": 0,
            "existing_chunks": 0,
            "new_files_found": 0,
            "new_chunks_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embeddings_generated": 0,
            "vector_store_updated": False,
            "total_time": 0,
        }

    def validate_existing_cache(self) -> Dict[str, Any]:
        """
        Validate existing embedding cache before adding new content.

        Returns:
            Dictionary with cache validation results
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Validating Existing Embedding Cache")
        logger.info("=" * 80)

        if not self.cache_dir.exists():
            logger.warning(f"Cache directory not found: {self.cache_dir}")
            return {
                "cache_exists": False,
                "cache_files": 0,
                "cache_size_mb": 0,
                "message": "No existing cache - will create fresh embeddings",
            }

        # Count cache files
        cache_files = list(self.cache_dir.glob("*.json"))
        cache_count = len(cache_files)

        # Calculate cache size
        cache_size = sum(f.stat().st_size for f in cache_files)
        cache_size_mb = cache_size / (1024 * 1024)

        self.stats["existing_cache_files"] = cache_count

        validation = {
            "cache_exists": True,
            "cache_files": cache_count,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_dir": str(self.cache_dir),
            "message": f"✓ Found {cache_count} cached embeddings ({cache_size_mb:.1f} MB)",
        }

        logger.info(f"✓ Cache Status: {validation['message']}")
        logger.info(f"  - Cache directory: {self.cache_dir}")
        logger.info(f"  - Cache files: {cache_count:,}")
        logger.info(f"  - Cache size: {cache_size_mb:.1f} MB")
        logger.info("")

        return validation

    def check_marketing_files(self) -> List[Path]:
        """
        Check for new marketing content files.

        Returns:
            List of marketing file paths
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Checking for New Marketing Content")
        logger.info("=" * 80)

        if not self.marketing_dir.exists():
            logger.warning(f"Marketing directory not found: {self.marketing_dir}")
            logger.info("Expected location for marketing templates:")
            logger.info(f"  {self.marketing_dir}/")
            return []

        # Find all markdown files in marketing directory
        marketing_files = list(self.marketing_dir.glob("*.md"))
        self.stats["new_files_found"] = len(marketing_files)

        logger.info(f"✓ Found {len(marketing_files)} marketing content files:")
        for i, file in enumerate(marketing_files, 1):
            file_size = file.stat().st_size / 1024  # KB
            logger.info(f"  {i}. {file.name} ({file_size:.1f} KB)")
        logger.info("")

        return marketing_files

    async def process_new_content(self, marketing_files: List[Path]) -> List[Any]:
        """
        Process new marketing files into chunks.

        Args:
            marketing_files: List of marketing content file paths

        Returns:
            List of document chunks
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Processing New Marketing Content into Chunks")
        logger.info("=" * 80)

        if not marketing_files:
            logger.warning("No marketing files to process")
            return []

        all_chunks = []

        for file in marketing_files:
            logger.info(f"Processing: {file.name}")

            try:
                # Read file content
                content = file.read_text(encoding="utf-8")

                # Create document
                from langchain_core.documents import Document

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": f"marketing/{file.name}",
                        "type": "marketing_guide",
                        "file_name": file.name,
                        "category": "marketing",
                    },
                )

                # Chunk document
                chunks = await self.chunker.chunk_documents([doc])

                logger.info(f"  ✓ Created {len(chunks)} chunks from {file.name}")
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"  ✗ Failed to process {file.name}: {e}")

        self.stats["new_chunks_created"] = len(all_chunks)

        logger.info("")
        logger.info(f"✓ Total new chunks created: {len(all_chunks)}")
        logger.info("")

        return all_chunks

    async def generate_embeddings_incremental(
        self, chunks: List[Any]
    ) -> List[Tuple[Any, List[float]]]:
        """
        Generate embeddings for new chunks (existing cached chunks skipped automatically).

        Args:
            chunks: List of document chunks

        Returns:
            List of (chunk, embedding) tuples
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Generating Embeddings (Cache-Aware)")
        logger.info("=" * 80)

        if not chunks:
            logger.warning("No chunks to embed")
            return []

        logger.info(f"Processing {len(chunks)} chunks...")
        logger.info(f"Cache enabled: {self.embedder.enable_cache}")
        logger.info(f"Model: {self.embedder.model_name}")
        logger.info("")
        logger.info("⚡ Cache mechanism:")
        logger.info("  - Existing content → Cache hit (instant, no API call)")
        logger.info("  - New content → Cache miss (generate embedding)")
        logger.info("")

        start_time = time.time()

        # Generate embeddings (cache handles deduplication automatically)
        embedded_chunks = await self.embedder.generate_embeddings(chunks)

        elapsed = time.time() - start_time

        # Get statistics from embedder
        embedder_stats = self.embedder.get_stats()

        self.stats["cache_hits"] = embedder_stats["cache_hits"]
        self.stats["cache_misses"] = embedder_stats["cache_misses"]
        self.stats["embeddings_generated"] = embedder_stats["total_processed"]

        logger.info("")
        logger.info(f"✓ Embedding generation complete in {elapsed:.2f}s")
        logger.info(
            f"  - Cache hits: {embedder_stats['cache_hits']} (existing content, skipped)"
        )
        logger.info(
            f"  - Cache misses: {embedder_stats['cache_misses']} (new embeddings generated)"
        )
        logger.info(f"  - Cache hit rate: {embedder_stats['cache_hit_rate']}")
        logger.info(f"  - Total processed: {len(embedded_chunks)}")
        logger.info("")

        return embedded_chunks

    async def update_vector_store(
        self, embedded_chunks: List[Tuple[Any, List[float]]]
    ) -> bool:
        """
        Update FAISS vector store with new embeddings.

        Args:
            embedded_chunks: List of (chunk, embedding) tuples

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Updating Vector Store (Incremental)")
        logger.info("=" * 80)

        if not embedded_chunks:
            logger.warning("No embedded chunks to add to vector store")
            return False

        try:
            # Initialize vector store
            logger.info("Loading existing vector store...")
            self.vector_store = VectorStore()

            # Load existing index
            if self.vector_store_dir.exists():
                await self.vector_store.load()
                existing_count = self.vector_store.get_statistics()["total_documents"]
                logger.info(
                    f"  ✓ Loaded existing index with {existing_count:,} documents"
                )
            else:
                logger.info("  - No existing index found, will create new one")
                existing_count = 0

            # Add new embeddings
            logger.info("")
            logger.info(f"Adding {len(embedded_chunks)} new documents...")

            docs = [chunk for chunk, _ in embedded_chunks]
            embeddings = [emb for _, emb in embedded_chunks]

            await self.vector_store.add_documents(docs, embeddings)

            # Save updated index
            logger.info("Saving updated vector store...")
            await self.vector_store.save()

            # Get updated statistics
            new_stats = self.vector_store.get_statistics()
            new_count = new_stats["total_documents"]

            self.stats["existing_chunks"] = existing_count
            self.stats["vector_store_updated"] = True

            logger.info("")
            logger.info(f"✓ Vector store updated successfully")
            logger.info(f"  - Previous document count: {existing_count:,}")
            logger.info(f"  - New documents added: {len(embedded_chunks)}")
            logger.info(f"  - Total document count: {new_count:,}")
            logger.info(f"  - Index type: {new_stats['index_type']}")
            logger.info("")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to update vector store: {e}")
            import traceback

            traceback.print_exc()
            return False

    def validate_results(self) -> Dict[str, Any]:
        """
        Validate the incremental embedding results.

        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Validation and Results")
        logger.info("=" * 80)

        # Calculate metrics
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / total_cache_requests * 100)
            if total_cache_requests > 0
            else 0
        )

        expected_new_chunks = self.stats["new_chunks_created"]
        actual_new_embeddings = self.stats["cache_misses"]

        # Validation checks
        cache_working = (
            cache_hit_rate == 0 or cache_hit_rate > 90
        )  # Either no cache or high hit rate
        chunks_match = expected_new_chunks == actual_new_embeddings
        vector_store_ok = self.stats["vector_store_updated"]

        all_valid = cache_working and chunks_match and vector_store_ok

        validation = {
            "success": all_valid,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_working": cache_working,
            "chunks_processed": self.stats["new_chunks_created"],
            "embeddings_generated": self.stats["cache_misses"],
            "chunks_match": chunks_match,
            "vector_store_updated": vector_store_ok,
            "total_documents": self.stats["existing_chunks"]
            + self.stats["cache_misses"],
        }

        # Print results
        logger.info("📊 Summary Statistics:")
        logger.info(f"  - Marketing files processed: {self.stats['new_files_found']}")
        logger.info(f"  - New chunks created: {self.stats['new_chunks_created']}")
        logger.info(
            f"  - Existing cached embeddings: {self.stats['existing_cache_files']:,}"
        )
        logger.info(f"  - New embeddings generated: {self.stats['cache_misses']}")
        logger.info(f"  - Cache hit rate: {cache_hit_rate:.1f}%")
        logger.info(
            f"  - Total documents in vector store: {validation['total_documents']:,}"
        )
        logger.info("")

        logger.info("✅ Validation Checks:")
        logger.info(
            f"  {'✓' if cache_working else '✗'} Cache mechanism working: {cache_working}"
        )
        logger.info(
            f"  {'✓' if chunks_match else '✗'} Chunks match embeddings: {chunks_match}"
        )
        logger.info(
            f"  {'✓' if vector_store_ok else '✗'} Vector store updated: {vector_store_ok}"
        )
        logger.info("")

        if all_valid:
            logger.info("🎉 SUCCESS: Marketing content added successfully!")
            logger.info("")
            logger.info("Next Steps:")
            logger.info("  1. Restart services to load new embeddings:")
            logger.info("     ./stop_all.sh && ./start_all.sh")
            logger.info("")
            logger.info("  2. Test marketing queries:")
            logger.info("     - 'Create a marketing strategy for fintech startups'")
            logger.info("     - 'Plan a product launch campaign with $25K budget'")
            logger.info("     - 'What are B2B SaaS buyer personas?'")
        else:
            logger.warning("⚠️  Some validation checks failed - review output above")

        logger.info("")

        return validation

    async def run(
        self, embed: bool = True, validate_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the incremental marketing content addition workflow.

        Args:
            embed: Whether to generate embeddings (default: True)
            validate_only: Only validate existing cache, don't process new content

        Returns:
            Dictionary with results
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Incremental Marketing Content Embedder")
        logger.info("=" * 80)
        logger.info("")

        try:
            # Step 1: Validate existing cache
            cache_validation = self.validate_existing_cache()

            if validate_only:
                logger.info("Validate-only mode: Skipping content processing")
                return {
                    "status": "validation_complete",
                    "cache_validation": cache_validation,
                    "stats": self.stats,
                }

            # Step 2: Check for marketing files
            marketing_files = self.check_marketing_files()

            if not marketing_files:
                logger.error("✗ No marketing files found - cannot proceed")
                logger.info("")
                logger.info("Expected marketing content files:")
                logger.info(f"  {self.marketing_dir}/marketing_strategy_framework.md")
                logger.info(f"  {self.marketing_dir}/campaign_planning_guide.md")
                logger.info(f"  {self.marketing_dir}/audience_segmentation_guide.md")
                logger.info("")
                logger.info("These files should have been created automatically.")
                logger.info(
                    "Check if the marketing directory exists and contains .md files."
                )
                return {
                    "status": "error",
                    "message": "No marketing files found",
                    "stats": self.stats,
                }

            # Step 3: Process new content into chunks
            chunks = await self.process_new_content(marketing_files)

            if not chunks:
                logger.error("✗ Failed to create chunks from marketing content")
                return {
                    "status": "error",
                    "message": "Chunk creation failed",
                    "stats": self.stats,
                }

            if not embed:
                logger.info("Embed disabled: Skipping embedding generation")
                return {
                    "status": "chunks_created",
                    "chunks_count": len(chunks),
                    "stats": self.stats,
                }

            # Step 4: Generate embeddings (cache-aware)
            embedded_chunks = await self.generate_embeddings_incremental(chunks)

            if not embedded_chunks:
                logger.error("✗ Failed to generate embeddings")
                return {
                    "status": "error",
                    "message": "Embedding generation failed",
                    "stats": self.stats,
                }

            # Step 5: Update vector store
            vector_store_ok = await self.update_vector_store(embedded_chunks)

            if not vector_store_ok:
                logger.error("✗ Failed to update vector store")
                return {
                    "status": "error",
                    "message": "Vector store update failed",
                    "stats": self.stats,
                }

            # Step 6: Validate results
            validation = self.validate_results()

            self.stats["total_time"] = time.time() - start_time

            logger.info(f"⏱️  Total execution time: {self.stats['total_time']:.2f}s")
            logger.info("=" * 80)

            return {
                "status": (
                    "success" if validation["success"] else "completed_with_warnings"
                ),
                "validation": validation,
                "stats": self.stats,
                "execution_time": self.stats["total_time"],
            }

        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": str(e), "stats": self.stats}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Add marketing content and generate embeddings incrementally"
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        default=True,
        help="Generate embeddings for new content (default: True)",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding generation (only create chunks)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing cache, don't process new content",
    )

    args = parser.parse_args()

    # Handle conflicting flags
    embed = args.embed and not args.no_embed

    # Run the embedder
    embedder = IncrementalMarketingEmbedder()
    result = asyncio.run(embedder.run(embed=embed, validate_only=args.validate_only))

    # Exit with appropriate code
    if result["status"] in ["success", "validation_complete"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
