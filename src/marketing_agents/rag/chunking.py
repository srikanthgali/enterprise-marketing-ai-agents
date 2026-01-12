"""Intelligent document chunking for RAG system."""

import logging
import re
import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """
    Intelligent document chunking strategy with special handling for code blocks,
    markdown headers, and semantic boundaries.

    Uses LangChain's RecursiveCharacterTextSplitter as base and extends it with
    custom logic for preserving code blocks, headers, and optimizing chunk boundaries.
    """

    # Predefined chunking parameters by document type
    DOCUMENT_TYPE_CONFIGS = {
        "technical_docs": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "separators": ["\n\n", "\n", ". ", " "],
        },
        "api_reference": {
            "chunk_size": 800,
            "chunk_overlap": 150,
            "separators": ["\n\n", "###", "##"],
        },
        "tutorials": {
            "chunk_size": 1200,
            "chunk_overlap": 250,
            "separators": ["\n\n", "\n"],
        },
        "faq": {
            "chunk_size": 600,
            "chunk_overlap": 100,
            "separators": ["\n\n", "?", ". "],
        },
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[list] = None,
    ):
        """
        Initialize the chunking strategy.

        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        # Initialize the base text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized ChunkingStrategy with chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={self.separators}"
        )

    @classmethod
    def from_document_type(cls, doc_type: str) -> "ChunkingStrategy":
        """
        Create a ChunkingStrategy configured for a specific document type.

        Args:
            doc_type: Document type (technical_docs, api_reference, tutorials, faq)

        Returns:
            ChunkingStrategy instance configured for the document type
        """
        if doc_type not in cls.DOCUMENT_TYPE_CONFIGS:
            logger.warning(
                f"Unknown document type '{doc_type}', using default configuration"
            )
            return cls()

        config = cls.DOCUMENT_TYPE_CONFIGS[doc_type]
        logger.info(f"Creating chunking strategy for document type: {doc_type}")
        return cls(**config)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk a list of documents with intelligent boundary handling.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects with enhanced metadata
        """
        logger.info(f"Starting chunking of {len(documents)} documents")
        all_chunks = []

        for doc_idx, document in enumerate(documents, 1):
            try:
                chunks = self._chunk_single_document(document)
                all_chunks.extend(chunks)

                if doc_idx % 10 == 0:
                    logger.info(
                        f"Chunked {doc_idx}/{len(documents)} documents, "
                        f"total chunks: {len(all_chunks)}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to chunk document {doc_idx} "
                    f"(source: {document.metadata.get('source', 'unknown')}): {e}",
                    exc_info=True,
                )

        logger.info(
            f"Chunking complete: {len(documents)} documents -> {len(all_chunks)} chunks"
        )
        return all_chunks

    def _chunk_single_document(self, document: Document) -> list[Document]:
        """
        Chunk a single document with intelligent handling.

        Args:
            document: Document to chunk

        Returns:
            List of chunked Document objects
        """
        content = document.page_content
        metadata = document.metadata

        # Step 1: Preserve code blocks
        content_with_markers, code_blocks = self._preserve_code_blocks(content)

        # Step 2: Split by sections if applicable
        sections = self._split_by_sections(content_with_markers)

        # Step 3: Chunk each section
        all_text_chunks = []
        for section in sections:
            # Use the base text splitter
            section_chunks = self.text_splitter.split_text(section)
            all_text_chunks.extend(section_chunks)

        # Step 4: Optimize chunk boundaries
        optimized_chunks = self._optimize_chunk_boundaries(all_text_chunks)

        # Step 5: Restore code blocks
        restored_chunks = self._restore_code_blocks(optimized_chunks, code_blocks)

        # Step 6: Create Document objects with enhanced metadata
        chunk_documents = []
        total_chunks = len(restored_chunks)

        for idx, chunk_text in enumerate(restored_chunks):
            # Generate unique chunk ID
            source_file = metadata.get("source", "unknown")
            chunk_id = self._generate_chunk_id(source_file, idx)

            # Create enhanced metadata
            chunk_metadata = {
                **metadata,  # Inherit all parent metadata
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_id": chunk_id,
                "chunk_size": len(chunk_text),
            }

            chunk_doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            chunk_documents.append(chunk_doc)

        return chunk_documents

    def _preserve_code_blocks(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Preserve code blocks by replacing them with markers.

        Args:
            text: Text content with potential code blocks

        Returns:
            Tuple of (text with markers, dictionary of marker -> code block)
        """
        code_blocks = {}
        marker_template = "<<<CODE_BLOCK_{}>>>"

        # Match code blocks with triple backticks
        pattern = r"```[\w]*\n.*?\n```"
        matches = list(re.finditer(pattern, text, re.DOTALL))

        # Replace code blocks with markers (in reverse to preserve indices)
        modified_text = text
        for idx, match in enumerate(reversed(matches)):
            marker = marker_template.format(len(matches) - idx - 1)
            code_blocks[marker] = match.group(0)
            modified_text = (
                modified_text[: match.start()] + marker + modified_text[match.end() :]
            )

        return modified_text, code_blocks

    def _restore_code_blocks(
        self, chunks: list[str], code_blocks: dict[str, str]
    ) -> list[str]:
        """
        Restore code blocks from markers.

        Args:
            chunks: List of text chunks with markers
            code_blocks: Dictionary of marker -> code block

        Returns:
            List of chunks with restored code blocks
        """
        restored_chunks = []
        for chunk in chunks:
            restored = chunk
            for marker, code_block in code_blocks.items():
                restored = restored.replace(marker, code_block)
            restored_chunks.append(restored)

        return restored_chunks

    def _split_by_sections(self, text: str) -> list[str]:
        """
        Split text by markdown headers while keeping headers with their content.

        Args:
            text: Text to split

        Returns:
            List of text sections
        """
        # Match markdown headers (# Header, ## Header, etc.)
        header_pattern = r"^(#{1,6}\s+.+)$"
        lines = text.split("\n")

        sections = []
        current_section = []

        for line in lines:
            if re.match(header_pattern, line):
                # If we have a current section, save it
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []

            current_section.append(line)

        # Add the last section
        if current_section:
            sections.append("\n".join(current_section))

        # If no sections were created (no headers), return original text
        return sections if sections else [text]

    def _optimize_chunk_boundaries(self, chunks: list[str]) -> list[str]:
        """
        Optimize chunk boundaries to avoid breaking mid-sentence or mid-list.

        Args:
            chunks: Raw text chunks

        Returns:
            Optimized chunks with cleaner boundaries
        """
        optimized = []

        for chunk in chunks:
            # Strip leading/trailing whitespace
            chunk = chunk.strip()

            # Skip empty chunks
            if not chunk:
                continue

            # Check if chunk starts mid-sentence (lowercase letter, not at list)
            if chunk and chunk[0].islower() and not re.match(r"^\s*[-*•]\s", chunk):
                # Try to find a better starting point
                sentences = re.split(r"([.!?]\s+)", chunk)
                if len(sentences) > 1:
                    # Start from first complete sentence
                    chunk = "".join(sentences[2:])

            # Check if chunk ends mid-sentence
            if chunk and not re.search(r"[.!?]\s*$", chunk):
                # Try to find a better ending point
                # Look for last sentence boundary
                last_boundary = max(
                    chunk.rfind(". "),
                    chunk.rfind("! "),
                    chunk.rfind("? "),
                )
                if last_boundary > len(chunk) * 0.7:  # Only if near the end
                    chunk = chunk[: last_boundary + 1]

            # Preserve list item continuity - don't break in the middle of a list
            lines = chunk.split("\n")
            cleaned_lines = []
            in_list = False

            for line in lines:
                # Check if line is a list item
                is_list_item = bool(re.match(r"^\s*[-*•]\s", line))

                if is_list_item:
                    in_list = True
                elif in_list and not line.strip():
                    # Blank line might end the list
                    in_list = False

                cleaned_lines.append(line)

            chunk = "\n".join(cleaned_lines)

            optimized.append(chunk.strip())

        return optimized

    def _generate_chunk_id(self, source: str, index: int) -> str:
        """
        Generate a unique chunk ID.

        Args:
            source: Source file path or identifier
            index: Chunk index

        Returns:
            Unique chunk identifier
        """
        # Extract filename from path
        if "/" in source:
            filename = source.split("/")[-1]
        else:
            filename = source

        # Remove extension
        if "." in filename:
            filename = filename.rsplit(".", 1)[0]

        # Create ID with filename, index, and short UUID
        short_uuid = str(uuid.uuid4())[:8]
        return f"{filename}_{index}_{short_uuid}"

    def validate_chunks(self, chunks: list[Document]) -> dict:
        """
        Validate chunks and return statistics.

        Args:
            chunks: List of chunked Document objects

        Returns:
            Dictionary with validation metrics:
                - total_chunks: Total number of chunks
                - avg_chunk_size: Average chunk size in characters
                - min_chunk_size: Minimum chunk size
                - max_chunk_size: Maximum chunk size
                - within_target: Count of chunks within target size range
                - oversized_chunks: List of chunks exceeding max size
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "within_target": 0,
                "oversized_chunks": [],
            }

        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        # Define acceptable range (target +/- 20%)
        min_acceptable = self.chunk_size * 0.8
        max_acceptable = self.chunk_size * 1.2

        within_target = sum(
            1 for size in chunk_sizes if min_acceptable <= size <= max_acceptable
        )

        # Find oversized chunks (more than 50% over target)
        max_threshold = self.chunk_size * 1.5
        oversized_chunks = [
            {
                "chunk_id": chunk.metadata.get("chunk_id", "unknown"),
                "size": len(chunk.page_content),
                "source": chunk.metadata.get("source", "unknown"),
            }
            for chunk in chunks
            if len(chunk.page_content) > max_threshold
        ]

        validation_results = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "within_target": within_target,
            "within_target_pct": (within_target / len(chunks)) * 100,
            "oversized_chunks": oversized_chunks,
        }

        # Log validation summary
        logger.info(
            f"Chunk validation: {validation_results['total_chunks']} chunks, "
            f"avg size: {validation_results['avg_chunk_size']:.0f}, "
            f"range: [{validation_results['min_chunk_size']}, "
            f"{validation_results['max_chunk_size']}], "
            f"within target: {validation_results['within_target_pct']:.1f}%"
        )

        if oversized_chunks:
            logger.warning(
                f"Found {len(oversized_chunks)} oversized chunks (>{max_threshold} chars)"
            )

        return validation_results
