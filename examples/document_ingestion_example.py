"""Example script demonstrating document ingestion pipeline usage."""

import asyncio
import logging

from src.marketing_agents.rag.document_ingestion import DocumentIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def main():
    """Main function to demonstrate document ingestion."""

    # Initialize the pipeline
    print("Initializing document ingestion pipeline...")
    pipeline = DocumentIngestionPipeline(
        source_dir="data/raw/knowledge_base", source_type="stripe_docs"
    )

    # Ingest documents
    print("\nStarting document ingestion...")
    documents = await pipeline.ingest_documents()

    # Display results
    print(f"\n{'='*60}")
    print(f"Ingestion Results")
    print(f"{'='*60}")
    print(f"Total documents ingested: {len(documents)}")

    # Get statistics
    stats = pipeline.get_statistics()
    print(f"\nDetailed Statistics:")
    print(f"  Total files discovered: {stats['total_files']}")
    print(f"  Successfully processed: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped (unsupported): {stats['skipped']}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Total characters: {stats['total_chars']:,}")

    # Display sample documents
    if documents:
        print(f"\n{'='*60}")
        print("Sample Documents")
        print(f"{'='*60}")

        for i, doc in enumerate(documents[:3], 1):
            print(f"\nDocument {i}:")
            print(f"  Title: {doc.metadata['title']}")
            print(f"  Category: {doc.metadata['category']}")
            print(f"  Source: {doc.metadata['source']}")
            print(f"  Word Count: {doc.metadata['word_count']}")
            print(f"  Reading Time: {doc.metadata['reading_time_minutes']} min")
            print(f"  Content Preview: {doc.page_content[:200]}...")

        if len(documents) > 3:
            print(f"\n... and {len(documents) - 3} more documents")

    print(f"\n{'='*60}")
    print("Ingestion complete!")
    print(f"{'='*60}")

    return documents


if __name__ == "__main__":
    # Run the async main function
    documents = asyncio.run(main())
