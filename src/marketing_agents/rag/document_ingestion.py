"""Document ingestion pipeline for RAG system."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document

from .utils import (
    calculate_reading_time,
    categorize_document_content,
    clean_content,
    extract_title_from_content,
)

# Configure logging
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """
    Pipeline for ingesting documents into the RAG system.

    Supports multiple file formats (.md, .html, .txt) and extracts rich metadata
    for improved retrieval and context understanding.
    """

    SUPPORTED_EXTENSIONS = {".md", ".html", ".txt"}

    def __init__(
        self,
        source_dir: str = "data/raw/knowledge_base",
        source_type: str = "stripe_docs",
        encoding: str = "utf-8",
    ):
        """
        Initialize the document ingestion pipeline.

        Args:
            source_dir: Directory containing documents to ingest
            source_type: Type of source (e.g., "stripe_docs", "internal_docs")
            encoding: File encoding (default: utf-8)
        """
        self.source_dir = Path(source_dir)
        self.source_type = source_type
        self.encoding = encoding

        # Statistics
        self.stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "total_words": 0,
            "total_chars": 0,
        }

        # Validate source directory
        if not self.source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        if not self.source_dir.is_dir():
            raise ValueError(f"Source path is not a directory: {source_dir}")

    async def ingest_documents(self) -> list[Document]:
        """
        Main entry point for document ingestion.

        Discovers all supported files, loads and cleans content, extracts metadata,
        and returns a list of LangChain Document objects ready for vector store.

        Returns:
            List of Document objects with content and metadata
        """
        logger.info(f"Starting document ingestion from: {self.source_dir}")
        start_time = datetime.now()

        # Discover all files
        file_paths = self._discover_files()
        self.stats["total_files"] = len(file_paths)
        logger.info(f"Discovered {len(file_paths)} files to ingest")

        # Process files
        documents = []
        for idx, file_path in enumerate(file_paths, 1):
            try:
                document = await self._process_file(file_path)
                if document:
                    documents.append(document)
                    self.stats["successful"] += 1

                    # Log progress every 10 documents
                    if idx % 10 == 0:
                        logger.info(
                            f"Progress: {idx}/{len(file_paths)} files processed "
                            f"({self.stats['successful']} successful, "
                            f"{self.stats['failed']} failed)"
                        )
            except Exception as e:
                self.stats["failed"] += 1
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

        # Calculate elapsed time
        elapsed = (datetime.now() - start_time).total_seconds()

        # Log final statistics
        self._log_ingestion_stats(elapsed)

        return documents

    def _discover_files(self) -> list[str]:
        """
        Recursively discover all supported files in source directory.

        Returns:
            List of absolute file paths
        """
        discovered_files = []

        for root, dirs, files in os.walk(self.source_dir):
            for file_name in files:
                file_path = Path(root) / file_name

                # Check if file has supported extension
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    discovered_files.append(str(file_path))
                else:
                    self.stats["skipped"] += 1
                    logger.debug(f"Skipping unsupported file: {file_path}")

        return sorted(discovered_files)

    async def _process_file(self, file_path: str) -> Optional[Document]:
        """
        Process a single file: load, clean, extract metadata.

        Args:
            file_path: Path to file to process

        Returns:
            Document object or None if processing fails
        """
        # Load file content
        content = self._load_file(file_path)
        if not content:
            logger.warning(f"Empty content for file: {file_path}")
            return None

        # Extract metadata
        metadata = self._extract_metadata(file_path, content)

        # Update statistics
        self.stats["total_words"] += metadata["word_count"]
        self.stats["total_chars"] += metadata["char_count"]

        # Create LangChain Document
        document = Document(
            page_content=content,
            metadata=metadata,
        )

        return document

    def _load_file(self, file_path: str) -> str:
        """
        Load and clean file content.

        Args:
            file_path: Path to file to load

        Returns:
            Cleaned file content
        """
        path = Path(file_path)

        try:
            # Try primary encoding
            with open(path, "r", encoding=self.encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            logger.warning(
                f"Failed to decode {file_path} with {self.encoding}, " f"trying latin-1"
            )
            try:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return ""
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return ""

        # Clean content based on file type
        cleaned_content = clean_content(content, path.suffix)

        return cleaned_content

    def _extract_metadata(self, file_path: str, content: str) -> dict:
        """
        Extract rich metadata from file and content.

        Args:
            file_path: Path to source file
            content: File content

        Returns:
            Dictionary of metadata
        """
        path = Path(file_path)

        # Extract title
        title = self._extract_title(content)
        if not title:
            # Fallback to filename without extension
            title = path.stem.replace("_", " ").replace("-", " ").title()

        # Categorize document
        category = self._categorize_document(content, file_path)

        # Calculate metrics
        word_count = len(content.split())
        char_count = len(content)
        reading_time = calculate_reading_time(word_count)

        # Get relative path from source_dir
        try:
            relative_path = path.relative_to(self.source_dir)
        except ValueError:
            relative_path = path

        # Build metadata dictionary
        metadata = {
            "source": str(relative_path),
            "source_type": self.source_type,
            "file_name": path.name,
            "file_extension": path.suffix,
            "title": title,
            "category": category,
            "word_count": word_count,
            "char_count": char_count,
            "reading_time_minutes": reading_time,
            "ingested_at": datetime.now().isoformat(),
            "language": "en",
        }

        return metadata

    def _extract_title(self, content: str) -> str:
        """
        Extract title from document content.

        Args:
            content: Document content

        Returns:
            Extracted title or empty string
        """
        title = extract_title_from_content(content)
        return title if title else ""

    def _categorize_document(self, content: str, file_path: str = "") -> str:
        """
        Classify document type based on content and path.

        Args:
            content: Document content
            file_path: Optional file path for additional context

        Returns:
            Document category (API Reference, Tutorial, Guide, FAQ, Conceptual)
        """
        return categorize_document_content(content, file_path)

    def _log_ingestion_stats(self, elapsed_seconds: float):
        """
        Log ingestion statistics.

        Args:
            elapsed_seconds: Time taken for ingestion
        """
        logger.info("=" * 60)
        logger.info("Document Ingestion Complete")
        logger.info("=" * 60)
        logger.info(f"Total files discovered: {self.stats['total_files']}")
        logger.info(f"Successfully ingested: {self.stats['successful']}")
        logger.info(f"Failed: {self.stats['failed']}")
        logger.info(f"Skipped (unsupported): {self.stats['skipped']}")
        logger.info(f"Total words: {self.stats['total_words']:,}")
        logger.info(f"Total characters: {self.stats['total_chars']:,}")
        logger.info(f"Time elapsed: {elapsed_seconds:.2f} seconds")

        if self.stats["successful"] > 0:
            avg_time = elapsed_seconds / self.stats["successful"]
            logger.info(f"Average time per document: {avg_time:.3f} seconds")

        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """
        Get ingestion statistics.

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
