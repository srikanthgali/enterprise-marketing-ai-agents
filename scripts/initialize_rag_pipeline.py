#!/usr/bin/env python3
"""
Complete RAG Pipeline Initialization Script

This script initializes the entire RAG (Retrieval-Augmented Generation) pipeline:
1. Document Ingestion - Loads documents from data/raw/knowledge_base/
2. Chunking - Applies intelligent chunking strategy
3. Embedding Generation - Generates embeddings in batches with progress tracking
4. Vector Store Creation - Builds FAISS index
5. Validation - Tests retrieval with sample queries

Usage:
    python scripts/initialize_rag_pipeline.py

Expected time: 4-6 minutes for ~1,000 documents
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.marketing_agents.rag import (
    AdvancedRetriever,
    ChunkingStrategy,
    DocumentIngestionPipeline,
    EmbeddingGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/rag_initialization.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RAGPipelineInitializer:
    """
    Orchestrates complete RAG pipeline initialization with progress tracking,
    error handling, and detailed reporting.
    """

    def __init__(self):
        """Initialize the RAG pipeline initializer."""
        self.settings = Settings()
        self.start_time = None
        self.report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": {},
            "summary": {},
            "errors": [],
        }

        # Create necessary directories
        self.embeddings_dir = self.settings.data_dir / "embeddings"
        self.reports_dir = self.settings.data_dir / "reports"
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def log_step(
        self, step_name: str, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a pipeline step."""
        self.report["steps"][step_name] = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

    async def initialize_rag_pipeline(self) -> Dict[str, Any]:
        """
        Main orchestration function for RAG pipeline initialization.

        Returns:
            Dictionary with initialization results and statistics
        """
        print("\n" + "=" * 80)
        print("🚀 Starting RAG Pipeline Initialization...")
        print("=" * 80 + "\n")

        self.start_time = time.time()

        try:
            # Step 1: Document Ingestion
            documents = await self._step_1_document_ingestion()

            # Step 2: Chunking
            chunks = await self._step_2_chunking(documents)

            # Step 3: Embedding Generation
            embedded_chunks = await self._step_3_embedding_generation(chunks)

            # Step 4: Vector Store Creation
            vector_store = await self._step_4_vector_store_creation(
                chunks, embedded_chunks
            )

            # Step 5: Validation
            await self._step_5_validation(vector_store)

            # Calculate total time
            total_time = time.time() - self.start_time

            # Build final summary
            self._build_summary(
                len(documents), len(chunks), len(embedded_chunks), total_time
            )

            # Save report
            self._save_report()

            # Print completion message
            self._print_completion()

            return {
                "status": "success",
                "documents": len(documents),
                "chunks": len(chunks),
                "embeddings": len(embedded_chunks),
                "total_time": total_time,
                "report": self.report,
            }

        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
            self.report["errors"].append(
                {"step": "initialization", "error": str(e), "type": type(e).__name__}
            )

            # Save partial report
            self._save_report()

            print("\n" + "=" * 80)
            print("❌ RAG Pipeline Initialization Failed")
            print("=" * 80)
            print(f"\nError: {e}")
            print(f"\nCheck logs at: logs/rag_initialization.log")
            print(f"Partial report saved to: {self._get_report_path()}")
            print("\n" + "=" * 80 + "\n")

            raise

    async def _step_1_document_ingestion(self) -> List[Document]:
        """
        Step 1: Document Ingestion

        Loads documents from data/raw/knowledge_base/ and extracts metadata.
        """
        print("[1/5] Document Ingestion...")
        step_start = time.time()

        try:
            # Check if knowledge base exists
            kb_path = self.settings.data_dir / "raw" / "knowledge_base"
            if not kb_path.exists():
                raise FileNotFoundError(
                    f"Knowledge base not found at: {kb_path}\n"
                    f"Run 'python scripts/run_data_extraction.py' first to scrape Stripe docs."
                )

            # Initialize ingestion pipeline
            pipeline = DocumentIngestionPipeline(
                source_dir=str(kb_path), source_type="stripe_docs"
            )

            # Ingest documents
            documents = await pipeline.ingest_documents()

            if not documents:
                raise ValueError(
                    "No documents were ingested. Check the knowledge base directory."
                )

            # Get statistics
            stats = pipeline.get_statistics()

            # Calculate time
            duration = time.time() - step_start

            # Log step
            self.log_step(
                "document_ingestion",
                "success",
                {
                    "total_documents": len(documents),
                    "total_files": stats.get("total_files", 0),
                    "successful": stats.get("successful", 0),
                    "failed": stats.get("failed", 0),
                    "total_words": stats.get("total_words", 0),
                    "duration_seconds": duration,
                },
            )

            print(f"✓ ({len(documents):,} documents in {duration:.1f}s)\n")
            return documents

        except Exception as e:
            self.log_step("document_ingestion", "failed", {"error": str(e)})
            raise

    async def _step_2_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Step 2: Chunking

        Applies intelligent chunking strategy to break documents into optimal sizes.
        """
        print("[2/5] Chunking...")
        step_start = time.time()

        try:
            # Initialize chunking strategy
            chunker = ChunkingStrategy(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " "],
            )

            # Chunk all documents (the method handles progress logging internally)
            chunks = chunker.chunk_documents(documents)

            if not chunks:
                raise ValueError("No chunks were created. Check chunking strategy.")

            # Calculate statistics
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
            chunk_sizes = [len(c.page_content) for c in chunks]
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)

            # Calculate time
            duration = time.time() - step_start

            # Log step
            self.log_step(
                "chunking",
                "success",
                {
                    "total_chunks": len(chunks),
                    "avg_chunk_size": int(avg_chunk_size),
                    "min_chunk_size": min_size,
                    "max_chunk_size": max_size,
                    "duration_seconds": duration,
                },
            )

            print(f"✓ ({len(chunks):,} chunks in {duration:.1f}s)\n")
            return chunks

        except Exception as e:
            self.log_step("chunking", "failed", {"error": str(e)})
            raise

    async def _step_3_embedding_generation(
        self, chunks: List[Document]
    ) -> List[tuple[Document, List[float]]]:
        """
        Step 3: Embedding Generation

        Generates embeddings for all chunks with batching and progress tracking.
        """
        print("[3/5] Embedding Generation...")
        step_start = time.time()

        try:
            # Initialize embedding generator
            embedder = EmbeddingGenerator(
                model_name=self.settings.vector_store.embedding_model,
                batch_size=self.settings.vector_store.embedding_batch_size,
                enable_cache=self.settings.vector_store.embedding_cache_enabled,
            )

            # Generate embeddings with custom progress tracking
            print(
                f"Generating embeddings (model: {self.settings.vector_store.embedding_model})"
            )
            print(f"Batch size: {self.settings.vector_store.embedding_batch_size}")

            embedded_chunks = []
            total_batches = (
                len(chunks) + self.settings.vector_store.embedding_batch_size - 1
            ) // self.settings.vector_store.embedding_batch_size

            with tqdm(
                total=len(chunks), desc="Generating embeddings", unit="chunk"
            ) as pbar:
                for i in range(
                    0, len(chunks), self.settings.vector_store.embedding_batch_size
                ):
                    batch = chunks[
                        i : i + self.settings.vector_store.embedding_batch_size
                    ]
                    try:
                        batch_embeddings = await embedder.generate_embeddings(batch)
                        embedded_chunks.extend(batch_embeddings)
                        pbar.update(len(batch))
                    except Exception as e:
                        logger.warning(f"Failed to generate embeddings for batch: {e}")
                        continue

            if not embedded_chunks:
                raise ValueError("No embeddings were generated.")

            # Validate embeddings
            validation = embedder.validate_embeddings(embedded_chunks)

            # Calculate time
            duration = time.time() - step_start

            # Estimate API cost (rough approximation)
            # OpenAI text-embedding-3-small: ~$0.02 per 1M tokens
            # Approx 1 token per 4 characters
            total_chars = sum(len(c.page_content) for c in chunks)
            estimated_tokens = total_chars // 4
            estimated_cost = (estimated_tokens / 1_000_000) * 0.02

            # Log step
            self.log_step(
                "embedding_generation",
                "success",
                {
                    "total_embeddings": len(embedded_chunks),
                    "embedding_dimension": validation["expected_dimension"],
                    "all_valid": validation["validation_passed"],
                    "duration_seconds": duration,
                    "estimated_api_cost_usd": round(estimated_cost, 4),
                    "estimated_tokens": estimated_tokens,
                },
            )

            print(f"✓ ({len(embedded_chunks):,} embeddings in {duration:.1f}s)")
            print(f"  Estimated cost: ${estimated_cost:.4f} USD\n")

            return embedded_chunks

        except Exception as e:
            self.log_step("embedding_generation", "failed", {"error": str(e)})
            raise

    async def _step_4_vector_store_creation(
        self,
        chunks: List[Document],
        embedded_chunks: List[tuple[Document, List[float]]],
    ) -> FAISS:
        """
        Step 4: Vector Store Creation

        Builds FAISS index and saves to disk.
        """
        print("[4/5] Vector Store Indexing...")
        step_start = time.time()

        try:
            # Create embeddings wrapper
            embeddings = OpenAIEmbeddings(
                model=self.settings.vector_store.embedding_model
            )

            # Create FAISS vector store
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Save to disk
            index_path = self.embeddings_dir / "stripe_knowledge_base"
            vector_store.save_local(str(index_path))

            # Calculate index size
            index_size_bytes = sum(
                f.stat().st_size for f in index_path.iterdir() if f.is_file()
            )
            index_size_mb = index_size_bytes / (1024 * 1024)

            # Calculate time
            duration = time.time() - step_start

            # Log step
            self.log_step(
                "vector_store_creation",
                "success",
                {
                    "index_path": str(index_path),
                    "index_size_bytes": index_size_bytes,
                    "index_size_mb": round(index_size_mb, 2),
                    "total_vectors": len(chunks),
                    "duration_seconds": duration,
                },
            )

            print(f"✓ ({index_size_mb:.1f} MB in {duration:.1f}s)")
            print(f"  Saved to: {index_path}\n")

            return vector_store

        except Exception as e:
            self.log_step("vector_store_creation", "failed", {"error": str(e)})
            raise

    async def _step_5_validation(self, vector_store: FAISS) -> None:
        """
        Step 5: Validation

        Tests retrieval with sample queries to verify the RAG pipeline works.
        """
        print("[5/5] Validation...")
        step_start = time.time()

        try:
            # Initialize retriever
            retriever = AdvancedRetriever(vector_store)

            # Sample test queries
            test_queries = [
                "How do I handle payment webhooks?",
                "What is Stripe Connect?",
                "How to implement subscription billing?",
            ]

            validation_results = []

            for query in test_queries:
                try:
                    results = await retriever.retrieve(query, top_k=3, rerank=False)
                    avg_relevance = (
                        sum(doc.metadata.get("relevance_score", 0.0) for doc in results)
                        / len(results)
                        if results
                        else 0.0
                    )

                    validation_results.append(
                        {
                            "query": query,
                            "results_found": len(results),
                            "avg_relevance": avg_relevance,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Validation query failed: {query} - {e}")
                    validation_results.append(
                        {
                            "query": query,
                            "results_found": 0,
                            "error": str(e),
                        }
                    )

            # Calculate time
            duration = time.time() - step_start

            # Check if validation passed
            all_passed = all(r.get("results_found", 0) > 0 for r in validation_results)

            # Log step
            self.log_step(
                "validation",
                "success" if all_passed else "partial",
                {
                    "test_queries": len(test_queries),
                    "queries_passed": sum(
                        1 for r in validation_results if r.get("results_found", 0) > 0
                    ),
                    "validation_results": validation_results,
                    "duration_seconds": duration,
                },
            )

            print(f"✓ ({len(test_queries)} test queries in {duration:.1f}s)\n")

        except Exception as e:
            self.log_step("validation", "failed", {"error": str(e)})
            raise

    def _build_summary(
        self,
        num_documents: int,
        num_chunks: int,
        num_embeddings: int,
        total_time: float,
    ) -> None:
        """Build final summary for the report."""
        index_path = self.embeddings_dir / "stripe_knowledge_base"

        # Calculate index size
        index_size_bytes = 0
        if index_path.exists():
            index_size_bytes = sum(
                f.stat().st_size for f in index_path.iterdir() if f.is_file()
            )

        self.report["summary"] = {
            "total_documents": num_documents,
            "total_chunks": num_chunks,
            "total_embeddings": num_embeddings,
            "index_size_bytes": index_size_bytes,
            "index_size_mb": round(index_size_bytes / (1024 * 1024), 2),
            "total_time_seconds": round(total_time, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "index_path": str(index_path),
            "status": "completed",
        }

    def _save_report(self) -> None:
        """Save the initialization report to JSON."""
        report_path = self._get_report_path()
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)
        logger.info(f"Report saved to: {report_path}")

    def _get_report_path(self) -> Path:
        """Get the report file path with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.reports_dir / f"rag_initialization_{timestamp}.json"

    def _print_completion(self) -> None:
        """Print completion message with summary."""
        summary = self.report["summary"]

        print("\n" + "=" * 80)
        print("✅ RAG Pipeline Initialization Complete!")
        print("=" * 80)
        print(f"\n📊 Summary:")
        print(f"  - Documents:  {summary['total_documents']:,}")
        print(f"  - Chunks:     {summary['total_chunks']:,}")
        print(f"  - Embeddings: {summary['total_embeddings']:,}")
        print(f"  - Index Size: {summary['index_size_mb']:.1f} MB")
        print(
            f"  - Total Time: {summary['total_time_minutes']:.1f}m ({summary['total_time_seconds']:.1f}s)"
        )
        print(f"\n💾 Index Path: {summary['index_path']}")
        print(f"📄 Report: {self._get_report_path()}")
        print("\n🧪 Run test query:")
        print(
            '   python -c "from scripts.test_rag import test_retrieval; test_retrieval()"'
        )
        print("\n" + "=" * 80 + "\n")


async def main():
    """Main entry point for RAG pipeline initialization."""
    initializer = RAGPipelineInitializer()

    try:
        result = await initializer.initialize_rag_pipeline()
        return result
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
