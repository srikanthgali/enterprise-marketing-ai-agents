#!/usr/bin/env python3
"""
Add Support Tickets and Campaign Data to Knowledge Base

This script adds synthetic support tickets and marketing campaign data
to the vector store to enhance agent capabilities with historical context.
"""

import asyncio
import json
import logging
import pandas as pd
import sys
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


async def add_support_tickets():
    """Load and add support tickets to vector store."""
    logger.info("\n" + "=" * 80)
    logger.info("Adding Support Tickets to Knowledge Base")
    logger.info("=" * 80)

    # Load all tickets
    tickets_dir = PROJECT_ROOT / "data" / "raw" / "support_tickets"
    all_tickets = []

    for file in sorted(tickets_dir.glob("tickets_2024_*.json")):
        logger.info(f"Loading {file.name}...")
        tickets = json.load(open(file))
        all_tickets.extend(tickets)
        logger.info(f"  Loaded {len(tickets)} tickets")

    logger.info(f"\nTotal tickets loaded: {len(all_tickets)}")

    # Transform to documents
    logger.info("\nTransforming tickets to documents...")
    documents = []

    for ticket in all_tickets:
        # Create rich document with searchable content
        doc = Document(
            page_content=f"""Support Ticket: {ticket['subject']}

Category: {ticket['category']} - {ticket['subcategory']}
Priority: {ticket['priority']}
Customer Tier: {ticket['customer_tier']}

Problem Description:
{ticket['description']}

Resolution Information:
- Resolution Time: {ticket['resolution_time_hours']:.1f} hours
- Customer Satisfaction: {ticket['satisfaction_score']}/5 stars
- Status: {ticket['status']}
- Agent Handoffs: {ticket['agent_handoffs']}
- Tags: {', '.join(ticket['tags'])}

Customer Context:
- Plan: {ticket['customer_context']['plan']}
- Monthly Events: {ticket['customer_context']['monthly_events']:,}
- Data Sources: {', '.join(ticket['customer_context']['sources'])}
- Account Age: {ticket['customer_context']['account_age_months']} months

This ticket demonstrates a {ticket['priority']} priority {ticket['category']} issue
that was resolved in {ticket['resolution_time_hours']:.1f} hours with a satisfaction
score of {ticket['satisfaction_score']}/5.
""",
            metadata={
                "source": f"support_tickets/{ticket['ticket_id']}",
                "type": "support_ticket",
                "ticket_id": ticket["ticket_id"],
                "category": ticket["category"],
                "subcategory": ticket["subcategory"],
                "priority": ticket["priority"],
                "customer_tier": ticket["customer_tier"],
                "resolution_time_hours": ticket["resolution_time_hours"],
                "satisfaction_score": ticket["satisfaction_score"],
                "status": ticket["status"],
                "created_at": ticket["created_at"],
                "tags": ticket["tags"],
            },
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} documents from tickets")

    return documents


async def add_marketing_campaigns():
    """Load and add marketing campaign data to vector store."""
    logger.info("\n" + "=" * 80)
    logger.info("Adding Marketing Campaigns to Knowledge Base")
    logger.info("=" * 80)

    # Load campaigns
    campaigns_file = (
        PROJECT_ROOT / "data" / "raw" / "marketing_data" / "campaigns_2024.csv"
    )
    logger.info(f"Loading {campaigns_file.name}...")

    df = pd.read_csv(campaigns_file)
    logger.info(f"Loaded {len(df)} campaigns")

    # Transform to documents
    logger.info("\nTransforming campaigns to documents...")
    documents = []

    for _, campaign in df.iterrows():
        # Create rich document with campaign details and performance
        doc = Document(
            page_content=f"""Marketing Campaign: {campaign['campaign_name']}

Campaign Details:
- Type: {campaign['campaign_type']}
- Channel: {campaign['channel']}
- Audience Segment: {campaign['audience_segment']}
- Geography: {campaign['geography']}
- Device Type: {campaign['device_type']}
- Age Group: {campaign['age_group']}
- Duration: {campaign['start_date']} to {campaign['end_date']}
- Status: {campaign['status']}

Budget & Investment:
- Campaign Budget: ${campaign['budget']:,.2f}
- Customer Acquisition Cost (CAC): ${campaign['cac']:.2f}

Performance Metrics:
- Impressions: {campaign['impressions']:,}
- Clicks: {campaign['clicks']:,}
- Click-Through Rate (CTR): {campaign['ctr']:.2f}%
- Conversions: {campaign['conversions']:,}
- Conversion Rate: {campaign['conversion_rate']:.2f}%
- Revenue Generated: ${campaign['revenue']:,.2f}
- Return on Investment (ROI): {campaign['roi']:.2f}%
- Customer Lifetime Value (LTV): ${campaign['ltv']:.2f}

Campaign Summary:
This {campaign['campaign_type']} campaign ran via {campaign['channel']} targeting
{campaign['audience_segment']} in {campaign['geography']}. With a budget of
${campaign['budget']:,.2f}, it achieved {campaign['conversions']:,} conversions
at a {campaign['conversion_rate']:.2f}% conversion rate, generating
${campaign['revenue']:,.2f} in revenue for an ROI of {campaign['roi']:.2f}%.

Key Insights:
- CTR of {campaign['ctr']:.2f}% indicates {
    'strong' if campaign['ctr'] > 4 else 'good' if campaign['ctr'] > 3 else 'moderate'
} audience engagement
- Conversion rate of {campaign['conversion_rate']:.2f}% is {
    'excellent' if campaign['conversion_rate'] > 6 else 'good' if campaign['conversion_rate'] > 4 else 'moderate'
}
- ROI of {campaign['roi']:.2f}% demonstrates {
    'exceptional' if campaign['roi'] > 20000 else 'strong' if campaign['roi'] > 10000 else 'solid'
} campaign performance
- CAC of ${campaign['cac']:.2f} with LTV of ${campaign['ltv']:.2f} shows {
    'healthy' if campaign['ltv'] / campaign['cac'] > 3 else 'acceptable'
} unit economics
""",
            metadata={
                "source": f"marketing_campaigns/{campaign['campaign_id']}",
                "type": "marketing_campaign",
                "campaign_id": campaign["campaign_id"],
                "campaign_type": campaign["campaign_type"],
                "channel": campaign["channel"],
                "audience_segment": campaign["audience_segment"],
                "geography": campaign["geography"],
                "status": campaign["status"],
                "budget": float(campaign["budget"]),
                "roi": float(campaign["roi"]),
                "conversion_rate": float(campaign["conversion_rate"]),
                "ctr": float(campaign["ctr"]),
                "revenue": float(campaign["revenue"]),
            },
        )
        documents.append(doc)

    logger.info(f"Created {len(documents)} documents from campaigns")

    return documents


async def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Synthetic Data Integration Script")
    logger.info("=" * 80)

    settings = Settings()

    # Step 1: Load support tickets
    ticket_docs = await add_support_tickets()

    # Step 2: Load marketing campaigns
    campaign_docs = await add_marketing_campaigns()

    # Combine all documents
    all_documents = ticket_docs + campaign_docs
    logger.info(f"\n Total documents to add: {len(all_documents)}")
    logger.info(f"  - Support tickets: {len(ticket_docs)}")
    logger.info(f"  - Marketing campaigns: {len(campaign_docs)}")

    # Step 3: Chunk documents (if needed for very long content)
    logger.info("\n" + "=" * 80)
    logger.info("Chunking Documents")
    logger.info("=" * 80)

    chunker = ChunkingStrategy(chunk_size=1200, chunk_overlap=200)
    chunks = chunker.chunk_documents(all_documents)
    logger.info(f"Created {len(chunks)} chunks from {len(all_documents)} documents")

    # Step 4: Generate embeddings
    logger.info("\n" + "=" * 80)
    logger.info("Generating Embeddings")
    logger.info("=" * 80)

    embedder = EmbeddingGenerator(
        model_name=settings.vector_store.embedding_model,
        batch_size=settings.vector_store.embedding_batch_size,
        enable_cache=settings.vector_store.embedding_cache_enabled,
    )

    embedded_chunks = await embedder.generate_embeddings(chunks)

    stats = embedder.get_stats()
    logger.info(f"\nEmbedding Statistics:")
    logger.info(f"  - Total processed: {len(embedded_chunks)}")
    logger.info(f"  - Cache hits: {stats['cache_hits']}")
    logger.info(f"  - Cache misses (new): {stats['cache_misses']}")
    logger.info(f"  - Cache hit rate: {stats['cache_hit_rate']}")

    # Step 5: Load existing vector store and add new documents
    logger.info("\n" + "=" * 80)
    logger.info("Updating Vector Store")
    logger.info("=" * 80)

    embeddings_model = OpenAIEmbeddings(
        model=settings.vector_store.embedding_model,
        openai_api_key=settings.api.openai_api_key.get_secret_value(),
    )

    vector_store_dir = PROJECT_ROOT / "data" / "embeddings" / "stripe_knowledge_base"

    vector_store = FAISS.load_local(
        str(vector_store_dir), embeddings_model, allow_dangerous_deserialization=True
    )

    before_count = vector_store.index.ntotal
    logger.info(f"Existing documents in vector store: {before_count:,}")

    # Add new embeddings
    docs = [chunk for chunk, _ in embedded_chunks]
    embs = [emb for _, emb in embedded_chunks]

    vector_store.add_embeddings(
        list(zip([d.page_content for d in docs], embs)),
        metadatas=[d.metadata for d in docs],
    )

    # Save updated vector store
    logger.info("Saving updated vector store...")
    vector_store.save_local(str(vector_store_dir))

    after_count = vector_store.index.ntotal

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ SUCCESS: Synthetic Data Added Successfully!")
    logger.info("=" * 80)
    logger.info(f"\nVector Store Update:")
    logger.info(f"  - Previous document count: {before_count:,}")
    logger.info(f"  - Support tickets added: {len(ticket_docs)}")
    logger.info(f"  - Campaign records added: {len(campaign_docs)}")
    logger.info(f"  - Total chunks added: {len(chunks)}")
    logger.info(f"  - New document count: {after_count:,}")
    logger.info(f"  - Net increase: +{after_count - before_count:,} documents")

    logger.info("\n📊 Data Breakdown:")
    logger.info(f"  - Technical Documentation: {6352} (original)")
    logger.info(f"  - Support Ticket History: {len(ticket_docs)}")
    logger.info(f"  - Marketing Campaigns: {len(campaign_docs)}")
    logger.info(f"  - Total Knowledge Base: {after_count:,}")

    logger.info("\n🎯 New Capabilities Enabled:")
    logger.info("  ✓ Customer Support: Search 500 historical tickets")
    logger.info("  ✓ Marketing Strategy: Analyze 150 campaign performances")
    logger.info("  ✓ Data-Driven Insights: Historical trends and patterns")
    logger.info("  ✓ Contextual Responses: Real-world examples and outcomes")

    logger.info("\n📋 Next Steps:")
    logger.info("  1. Restart services: ./stop_all.sh && ./start_all.sh")
    logger.info("  2. Test support query: 'How do others solve webhook issues?'")
    logger.info("  3. Test marketing query: 'What email campaigns performed best?'")
    logger.info("  4. Test analytics query: 'Show me Q4 campaign performance trends'")

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
