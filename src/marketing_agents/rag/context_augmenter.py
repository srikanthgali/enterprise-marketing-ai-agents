"""Context augmentation for RAG with citation support and intelligent truncation."""

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

# Configure logging
logger = logging.getLogger(__name__)


class ContextAugmenter:
    """
    Context augmentation system for RAG pipelines.

    Formats retrieved documents into prompts with:
    - Numbered source citations
    - Source metadata (title, URL)
    - Intelligent truncation when context is too long
    - Citation extraction and validation from LLM responses

    Features:
    - Configurable maximum context length
    - Automatic truncation of older/less relevant sources
    - Citation tracking and validation
    - Clean formatting for LLM consumption
    """

    def __init__(self, max_context_length: int = 4000):
        """
        Initialize the context augmenter.

        Args:
            max_context_length: Maximum character count for context portion (default: 4000)
        """
        self.max_context_length = max_context_length
        logger.info(
            f"ContextAugmenter initialized with max_context_length={max_context_length}"
        )

    def augment_prompt(
        self, query: str, retrieved_docs: List[Document], include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Augment a query with retrieved document context.

        Args:
            query: User's question/query
            retrieved_docs: List of retrieved Document objects
            include_citations: Whether to include source metadata for citations

        Returns:
            Dictionary containing:
                - augmented_prompt: Full prompt ready for LLM
                - context: Just the context portion
                - sources: List of source metadata dictionaries
                - context_length: Character count of context

        Example:
            >>> augmenter = ContextAugmenter(max_context_length=4000)
            >>> result = augmenter.augment_prompt(
            ...     query="How does Stripe handle refunds?",
            ...     retrieved_docs=docs,
            ...     include_citations=True
            ... )
            >>> # result["augmented_prompt"] ready for LLM
        """
        if not retrieved_docs:
            logger.warning("No documents provided for augmentation")
            return {
                "augmented_prompt": f"QUESTION: {query}\nANSWER:",
                "context": "",
                "sources": [],
                "context_length": 0,
            }

        # Format each document with source markers
        formatted_sources = []
        sources_metadata = []

        for idx, doc in enumerate(retrieved_docs, start=1):
            formatted_doc = self._format_document(doc, idx, include_citations)
            formatted_sources.append(formatted_doc)

            # Extract metadata for citation tracking
            metadata = doc.metadata or {}
            source_info = {
                "index": idx,
                "title": metadata.get("title", f"Source {idx}"),
                "url": metadata.get("source", metadata.get("url", "")),
                "category": metadata.get("category", ""),
                "chunk_id": metadata.get("chunk_id", ""),
            }
            sources_metadata.append(source_info)

        # Join all sources
        context = "\n\n".join(formatted_sources)

        # Truncate if necessary
        if len(context) > self.max_context_length:
            logger.info(
                f"Context length {len(context)} exceeds max {self.max_context_length}, truncating"
            )
            context = self._truncate_context(context, self.max_context_length)

            # Update sources metadata to reflect truncation
            # Count how many complete sources remain after truncation
            remaining_sources = []
            for source_meta in sources_metadata:
                if f"[Source {source_meta['index']}]" in context:
                    remaining_sources.append(source_meta)
            sources_metadata = remaining_sources

        # Build the augmented prompt
        augmented_prompt = f"CONTEXT: {context}\n\nQUESTION: {query}\n\nANSWER:"

        result = {
            "augmented_prompt": augmented_prompt,
            "context": context,
            "sources": sources_metadata,
            "context_length": len(context),
        }

        logger.debug(
            f"Augmented prompt with {len(sources_metadata)} sources, context length: {len(context)}"
        )
        return result

    def _format_document(
        self, doc: Document, index: int, include_citations: bool
    ) -> str:
        """
        Format a single document with source markers and metadata.

        Args:
            doc: Document to format
            index: Source number (1-indexed)
            include_citations: Whether to include title and URL metadata

        Returns:
            Formatted string with source marker and content

        Example output:
            [Source 1] Title: Stripe Refunds Guide
            URL: https://stripe.com/docs/refunds

            Content about refunds...
        """
        metadata = doc.metadata or {}
        parts = [f"[Source {index}]"]

        # Add title if available and citations enabled
        if include_citations:
            title = metadata.get("title", "")
            if title:
                parts.append(f"Title: {title}")

            # Add URL if available
            url = metadata.get("source", metadata.get("url", ""))
            if url:
                parts.append(f"URL: {url}")

        # Add a blank line before content if we added metadata
        if len(parts) > 1:
            parts.append("")  # Empty string creates blank line when joined

        # Add the actual content
        content = doc.page_content.strip()
        parts.append(content)

        return "\n".join(parts)

    def _truncate_context(self, context: str, max_length: int) -> str:
        """
        Intelligently truncate context if it exceeds maximum length.

        Strategy:
        - Truncates older/earlier sources first
        - Keeps most recent/relevant sources intact
        - Ensures no mid-word truncation
        - Adds "[truncated]" indicator

        Args:
            context: Full context string
            max_length: Maximum allowed character count

        Returns:
            Truncated context string
        """
        if len(context) <= max_length:
            return context

        # Find all source boundaries
        # Pattern matches [Source N] markers
        source_pattern = r"\[Source \d+\]"
        source_matches = list(re.finditer(source_pattern, context))

        if not source_matches:
            # No source markers found, do simple truncation
            truncated = context[:max_length].rsplit(" ", 1)[0]  # Avoid mid-word cut
            return truncated + "... [truncated]"

        # Try to keep as many complete sources as possible, starting from most recent
        # Work backwards from the last source
        reserve_space = len("\n\n[Earlier sources truncated for brevity]")
        available_length = max_length - reserve_space

        # Find the latest complete source that fits
        for i in range(len(source_matches) - 1, -1, -1):
            if i == len(source_matches) - 1:
                # Last source - take from its start to end of context
                candidate = context[source_matches[i].start() :]
            else:
                # Take from this source to end
                candidate = context[source_matches[i].start() :]

            if len(candidate) <= available_length:
                # This source and all following sources fit
                if i == 0:
                    # All sources fit (shouldn't happen, but handle it)
                    return context[:max_length]
                else:
                    # Add truncation notice before the kept sources
                    truncation_notice = "[Earlier sources truncated for brevity]\n\n"
                    return truncation_notice + candidate

        # If even the last source is too long, truncate it
        last_source_start = source_matches[-1].start()
        last_source = context[last_source_start : last_source_start + available_length]

        # Avoid mid-word truncation
        last_space = last_source.rfind(" ")
        if last_space > len(last_source) * 0.8:  # Only if we're not losing too much
            last_source = last_source[:last_space]

        truncation_notice = "[Earlier sources truncated for brevity]\n\n"
        return truncation_notice + last_source + "... [truncated]"

    def extract_citations(
        self, response: str, sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract and validate citations from an LLM response.

        Parses the response for citation markers like:
        - [1], [2], etc.
        - [Source 1], [Source 2], etc.

        Args:
            response: LLM's generated response text
            sources: List of source metadata dictionaries from augment_prompt()

        Returns:
            Dictionary containing:
                - cited_sources: List of source metadata dicts that were cited
                - citation_indices: List of cited source indices
                - total_citations: Total count of citation markers found
                - uncited_sources: List of sources that weren't cited

        Example:
            >>> citations = augmenter.extract_citations(
            ...     llm_response,
            ...     result["sources"]
            ... )
            >>> print(f"Cited {len(citations['cited_sources'])} sources")
        """
        if not response or not sources:
            return {
                "cited_sources": [],
                "citation_indices": [],
                "total_citations": 0,
                "uncited_sources": sources if sources else [],
            }

        # Pattern to match [1], [2], [Source 1], [Source 2], etc.
        citation_pattern = r"\[(?:Source )?(\d+)\]"
        matches = re.findall(citation_pattern, response)

        # Convert to integers and get unique indices
        cited_indices = sorted(set(int(m) for m in matches))

        # Validate that cited indices exist in sources
        valid_indices = self._validate_citations(cited_indices, sources)

        # Get the actual source metadata for cited sources
        cited_sources = []
        for idx in valid_indices:
            # Find source with this index
            for source in sources:
                if source.get("index") == idx:
                    cited_sources.append(source)
                    break

        # Find uncited sources
        cited_indices_set = set(valid_indices)
        uncited_sources = [
            source for source in sources if source.get("index") not in cited_indices_set
        ]

        result = {
            "cited_sources": cited_sources,
            "citation_indices": valid_indices,
            "total_citations": len(matches),  # Total mentions, including duplicates
            "uncited_sources": uncited_sources,
        }

        logger.debug(
            f"Extracted {len(valid_indices)} unique citations from {len(matches)} total mentions"
        )

        return result

    def _validate_citations(
        self, cited_indices: List[int], sources: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Validate that cited indices exist in the sources list.

        Args:
            cited_indices: List of citation indices extracted from response
            sources: List of source metadata dictionaries

        Returns:
            List of valid citation indices (those that exist in sources)
        """
        if not sources:
            return []

        # Get valid index range from sources
        valid_indices = {source.get("index") for source in sources if "index" in source}

        # Filter to only valid citations
        validated = [idx for idx in cited_indices if idx in valid_indices]

        # Log any invalid citations
        invalid = set(cited_indices) - set(validated)
        if invalid:
            logger.warning(f"Found invalid citation indices: {invalid}")

        return validated


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create sample documents
    sample_docs = [
        Document(
            page_content="Stripe allows you to issue full or partial refunds for charges. Refunds are processed asynchronously and can take 5-10 business days to appear on a customer's statement.",
            metadata={
                "title": "Stripe Refunds Guide",
                "source": "https://stripe.com/docs/refunds",
                "category": "payment_processing",
            },
        ),
        Document(
            page_content="To create a refund, use the Refunds API endpoint. You can specify the amount to refund, or omit it to refund the full charge amount. Refunds can be created via API, dashboard, or webhooks.",
            metadata={
                "title": "Refunds API Reference",
                "source": "https://stripe.com/docs/api/refunds",
                "category": "api_reference",
            },
        ),
        Document(
            page_content="Subscription refunds work differently than one-time charge refunds. When you refund a subscription payment, you can choose whether to prorate the refund amount based on unused time.",
            metadata={
                "title": "Subscription Refunds",
                "source": "https://stripe.com/docs/billing/subscriptions/refunds",
                "category": "subscriptions",
            },
        ),
    ]

    # Initialize augmenter
    augmenter = ContextAugmenter(max_context_length=4000)

    # Augment prompt
    result = augmenter.augment_prompt(
        query="How does Stripe handle subscription refunds?",
        retrieved_docs=sample_docs,
        include_citations=True,
    )

    print("=" * 80)
    print("AUGMENTED PROMPT:")
    print("=" * 80)
    print(result["augmented_prompt"])
    print("\n" + "=" * 80)
    print(f"Context length: {result['context_length']} characters")
    print(f"Number of sources: {len(result['sources'])}")
    print("\n" + "=" * 80)
    print("SOURCES METADATA:")
    print("=" * 80)
    for source in result["sources"]:
        print(f"[{source['index']}] {source['title']} - {source['url']}")

    # Simulate LLM response with citations
    llm_response = """
    Stripe handles subscription refunds with special considerations [Source 3].
    Unlike regular charge refunds [1], subscription refunds can be prorated based
    on unused time. You can create refunds using the Refunds API [2].
    """

    # Extract citations
    citations = augmenter.extract_citations(llm_response, result["sources"])

    print("\n" + "=" * 80)
    print("CITATION ANALYSIS:")
    print("=" * 80)
    print(f"Total citation mentions: {citations['total_citations']}")
    print(f"Unique sources cited: {len(citations['cited_sources'])}")
    print("\nCited sources:")
    for source in citations["cited_sources"]:
        print(f"  [{source['index']}] {source['title']}")

    if citations["uncited_sources"]:
        print("\nUncited sources:")
        for source in citations["uncited_sources"]:
            print(f"  [{source['index']}] {source['title']}")
