"""Example of integrating ContextAugmenter with AdvancedRetriever in a RAG pipeline."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from src.marketing_agents.rag.context_augmenter import ContextAugmenter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def rag_pipeline_with_context_augmentation():
    """
    Demonstrate a complete RAG pipeline with context augmentation.

    Pipeline flow:
    1. User query
    2. Retrieval (using AdvancedRetriever)
    3. Context augmentation (using ContextAugmenter)
    4. LLM generation (simulated)
    5. Citation extraction
    """

    # Step 1: User query
    user_query = "How does Stripe handle webhook authentication?"
    logger.info(f"User query: {user_query}")

    # Step 2: Simulate retrieval (in real use, would use AdvancedRetriever)
    # from src.marketing_agents.rag.retriever import AdvancedRetriever
    # retriever = AdvancedRetriever(vector_store)
    # retrieved_docs = await retriever.retrieve(user_query, top_k=5)

    # For this example, create mock retrieved documents
    retrieved_docs = [
        Document(
            page_content=(
                "Stripe uses webhook signatures to verify that webhook events are sent by Stripe. "
                "Each webhook request includes a Stripe-Signature header that contains a timestamp "
                "and one or more signatures."
            ),
            metadata={
                "title": "Webhook Security",
                "source": "https://stripe.com/docs/webhooks/signatures",
                "category": "security",
                "chunk_id": "webhook_sec_001",
            },
        ),
        Document(
            page_content=(
                "To verify webhook signatures, you need your webhook signing secret from the "
                "Stripe Dashboard. Use this secret with Stripe's SDK to validate the signature "
                "and timestamp to prevent replay attacks."
            ),
            metadata={
                "title": "Verifying Webhook Signatures",
                "source": "https://stripe.com/docs/webhooks/signatures#verify",
                "category": "security",
                "chunk_id": "webhook_sec_002",
            },
        ),
        Document(
            page_content=(
                "Webhook endpoints should be protected with HTTPS and should validate the "
                "Stripe signature before processing events. Never trust webhook events without "
                "signature verification."
            ),
            metadata={
                "title": "Webhook Best Practices",
                "source": "https://stripe.com/docs/webhooks/best-practices",
                "category": "best_practices",
                "chunk_id": "webhook_bp_001",
            },
        ),
        Document(
            page_content=(
                "When testing webhooks locally, you can use the Stripe CLI to forward webhook "
                "events to your local server. The CLI automatically handles signature generation."
            ),
            metadata={
                "title": "Testing Webhooks",
                "source": "https://stripe.com/docs/webhooks/test",
                "category": "testing",
                "chunk_id": "webhook_test_001",
            },
        ),
    ]

    logger.info(f"Retrieved {len(retrieved_docs)} documents")

    # Step 3: Context augmentation
    augmenter = ContextAugmenter(max_context_length=4000)
    augmentation_result = augmenter.augment_prompt(
        query=user_query, retrieved_docs=retrieved_docs, include_citations=True
    )

    logger.info(
        f"Context augmented: {augmentation_result['context_length']} characters"
    )
    logger.info(f"Sources included: {len(augmentation_result['sources'])}")

    # Display the augmented prompt
    print("\n" + "=" * 80)
    print("AUGMENTED PROMPT FOR LLM:")
    print("=" * 80)
    print(augmentation_result["augmented_prompt"])

    # Step 4: LLM generation (simulated)
    # In real use: llm_response = await llm.generate(augmentation_result["augmented_prompt"])
    simulated_llm_response = """
    Stripe handles webhook authentication through signature verification [1]. Each webhook
    request includes a Stripe-Signature header containing a timestamp and signatures.

    To verify webhooks, you need your webhook signing secret from the Stripe Dashboard [2].
    This secret is used with Stripe's SDK to validate signatures and prevent replay attacks.

    Best practices include using HTTPS for webhook endpoints and always validating signatures
    before processing events [3]. For local testing, the Stripe CLI can forward events to your
    local server with automatic signature handling [4].
    """

    logger.info("LLM response generated")

    # Step 5: Citation extraction
    citations = augmenter.extract_citations(
        response=simulated_llm_response, sources=augmentation_result["sources"]
    )

    # Display results
    print("\n" + "=" * 80)
    print("LLM RESPONSE:")
    print("=" * 80)
    print(simulated_llm_response)

    print("\n" + "=" * 80)
    print("CITATION ANALYSIS:")
    print("=" * 80)
    print(f"Total citation mentions: {citations['total_citations']}")
    print(f"Unique sources cited: {len(citations['cited_sources'])}")

    print("\n📚 Cited Sources:")
    for source in citations["cited_sources"]:
        print(f"  [{source['index']}] {source['title']}")
        print(f"      {source['url']}")

    if citations["uncited_sources"]:
        print("\n⚠️  Uncited Sources (retrieved but not mentioned):")
        for source in citations["uncited_sources"]:
            print(f"  [{source['index']}] {source['title']}")

    # Calculate citation coverage
    coverage = (
        len(citations["cited_sources"]) / len(augmentation_result["sources"]) * 100
    )
    print(f"\n📊 Citation Coverage: {coverage:.1f}%")

    return {
        "query": user_query,
        "retrieved_docs": len(retrieved_docs),
        "context_length": augmentation_result["context_length"],
        "llm_response": simulated_llm_response,
        "citations": citations,
        "coverage": coverage,
    }


def demonstrate_truncation():
    """Demonstrate context truncation when content is too long."""

    print("\n" + "=" * 80)
    print("TRUNCATION DEMONSTRATION:")
    print("=" * 80)

    # Create documents with large content
    large_docs = []
    for i in range(10):
        content = f"This is source {i+1}. " + "Lorem ipsum dolor sit amet. " * 100
        large_docs.append(
            Document(
                page_content=content,
                metadata={
                    "title": f"Large Document {i+1}",
                    "source": f"https://example.com/doc{i+1}",
                    "category": "example",
                },
            )
        )

    # Use a small max_context_length to force truncation
    augmenter = ContextAugmenter(max_context_length=2000)
    result = augmenter.augment_prompt(
        query="Test query", retrieved_docs=large_docs, include_citations=True
    )

    print(f"\nOriginal documents: 10")
    print(f"Documents after truncation: {len(result['sources'])}")
    print(f"Context length: {result['context_length']} characters (max: 2000)")
    print(f"\nTruncated context preview:")
    print(result["context"][:500] + "...")


if __name__ == "__main__":
    # Run the main RAG pipeline example
    print("\n🚀 RAG Pipeline with Context Augmentation\n")
    asyncio.run(rag_pipeline_with_context_augmentation())

    # Demonstrate truncation
    demonstrate_truncation()

    print("\n" + "=" * 80)
    print("✅ Integration example completed successfully!")
    print("=" * 80)
