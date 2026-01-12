"""
Example script demonstrating RAG evaluation usage with different scenarios.

This script shows how to:
1. Initialize the RAGEvaluator
2. Run evaluations with different configurations
3. Compare retrieval strategies
4. Generate and export reports
"""

import asyncio
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.marketing_agents.rag.retriever import AdvancedRetriever
from tests.evaluation.test_rag_pipeline import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_basic_evaluation():
    """Run basic evaluation with default settings."""
    print("\n" + "=" * 70)
    print("BASIC EVALUATION - Default Settings")
    print("=" * 70)

    # Load vector store (assumes it's already been created)
    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings=None,  # Will be loaded from saved index
            allow_dangerous_deserialization=True,
        )
        logger.info(f"Loaded vector store from {vector_store_path}")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        logger.info("Please run the embedding generation pipeline first")
        return

    # Initialize retriever and evaluator
    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(
        retriever=retriever, test_cases_path="data/evaluation/rag_test_cases.json"
    )

    # Run evaluation with default settings (top_k=5, rerank=True)
    results = await evaluator.evaluate_retrieval(top_k=5, rerank=True)

    # Print summary metrics
    metrics = results["metrics"]
    print(f"\n📊 RESULTS:")
    print(f"  Precision@5: {metrics['avg_precision']:.3f} (target: > 0.80)")
    print(f"  Recall@5:    {metrics['avg_recall']:.3f} (target: > 0.70)")
    print(f"  MRR:         {metrics['avg_mrr']:.3f} (target: > 0.85)")
    print(f"  NDCG:        {metrics['avg_ndcg']:.3f} (target: > 0.80)")
    print(f"  F1 Score:    {metrics['avg_f1']:.3f}")
    print(f"  Avg Time:    {metrics['avg_response_time']:.3f}s")
    print(
        f"  Failed:      {metrics['failed_queries_count']}/{metrics['total_queries']}"
    )

    # Export results
    output_files = evaluator.export_results(results)
    print(f"\n📁 Results exported to:")
    print(f"  JSON: {output_files['json']}")
    print(f"  MD:   {output_files['markdown']}")

    return results


async def compare_top_k_values():
    """Compare evaluation metrics across different top_k values."""
    print("\n" + "=" * 70)
    print("COMPARISON - Different top_k Values")
    print("=" * 70)

    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=None, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(
        retriever=retriever, test_cases_path="data/evaluation/rag_test_cases.json"
    )

    # Test different top_k values
    top_k_values = [3, 5, 10]
    comparison_results = {}

    for k in top_k_values:
        logger.info(f"Evaluating with top_k={k}")
        results = await evaluator.evaluate_retrieval(top_k=k, rerank=True)
        comparison_results[k] = results["metrics"]

    # Display comparison table
    print("\n📊 COMPARISON TABLE:")
    print(f"{'Metric':<20} {'K=3':<12} {'K=5':<12} {'K=10':<12}")
    print("-" * 56)

    metrics_to_compare = [
        ("Precision@K", "avg_precision"),
        ("Recall@K", "avg_recall"),
        ("MRR", "avg_mrr"),
        ("NDCG", "avg_ndcg"),
        ("F1 Score", "avg_f1"),
        ("Avg Time (s)", "avg_response_time"),
    ]

    for metric_name, metric_key in metrics_to_compare:
        values = [comparison_results[k][metric_key] for k in top_k_values]
        print(
            f"{metric_name:<20} {values[0]:<12.3f} {values[1]:<12.3f} {values[2]:<12.3f}"
        )

    print("\n💡 Insights:")
    print("  - Higher K typically improves recall but may reduce precision")
    print("  - MRR is less affected by K since it only considers first relevant doc")
    print("  - Response time increases with K due to more processing")


async def compare_retrieval_strategies():
    """Compare semantic search vs. hybrid search."""
    print("\n" + "=" * 70)
    print("COMPARISON - Retrieval Strategies")
    print("=" * 70)

    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=None, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(
        retriever=retriever, test_cases_path="data/evaluation/rag_test_cases.json"
    )

    # Scenario 1: Semantic search with reranking
    logger.info("Evaluating: Semantic + Reranking")
    semantic_results = await evaluator.evaluate_retrieval(top_k=5, rerank=True)

    # Scenario 2: Semantic search without reranking
    logger.info("Evaluating: Semantic only")
    retriever.clear_cache()  # Clear cache to ensure fresh results
    no_rerank_results = await evaluator.evaluate_retrieval(top_k=5, rerank=False)

    # Display comparison
    print("\n📊 STRATEGY COMPARISON:")
    print(
        f"{'Metric':<20} {'Semantic+Rerank':<18} {'Semantic Only':<18} {'Difference':<12}"
    )
    print("-" * 68)

    metrics_to_compare = [
        ("Precision@5", "avg_precision"),
        ("Recall@5", "avg_recall"),
        ("MRR", "avg_mrr"),
        ("NDCG", "avg_ndcg"),
        ("F1 Score", "avg_f1"),
    ]

    for metric_name, metric_key in metrics_to_compare:
        val1 = semantic_results["metrics"][metric_key]
        val2 = no_rerank_results["metrics"][metric_key]
        diff = val1 - val2
        sign = "+" if diff >= 0 else ""
        print(f"{metric_name:<20} {val1:<18.3f} {val2:<18.3f} {sign}{diff:<12.3f}")

    print("\n💡 Insights:")
    print("  - Reranking typically improves precision and NDCG")
    print("  - May slightly increase response time due to scoring overhead")


async def evaluate_by_category():
    """Evaluate performance across different query categories."""
    print("\n" + "=" * 70)
    print("ANALYSIS - Performance by Category")
    print("=" * 70)

    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=None, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(
        retriever=retriever, test_cases_path="data/evaluation/rag_test_cases.json"
    )

    # Run full evaluation
    results = await evaluator.evaluate_retrieval(top_k=5, rerank=True)

    # Display category breakdown
    category_metrics = results["metrics"]["category_metrics"]

    print("\n📊 CATEGORY PERFORMANCE:")
    print(f"{'Category':<25} {'Count':<8} {'Precision':<12} {'Recall':<12} {'MRR':<12}")
    print("-" * 69)

    # Sort categories by precision (descending)
    sorted_categories = sorted(
        category_metrics.items(), key=lambda x: x[1]["precision"], reverse=True
    )

    for category, metrics in sorted_categories:
        print(
            f"{category:<25} {metrics['count']:<8} "
            f"{metrics['precision']:<12.3f} "
            f"{metrics['recall']:<12.3f} "
            f"{metrics['mrr']:<12.3f}"
        )

    # Identify best and worst performing categories
    if sorted_categories:
        best_cat, best_metrics = sorted_categories[0]
        worst_cat, worst_metrics = sorted_categories[-1]

        print(f"\n✅ Best performing: {best_cat}")
        print(
            f"   Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}"
        )

        print(f"\n⚠️  Needs improvement: {worst_cat}")
        print(
            f"   Precision: {worst_metrics['precision']:.3f}, Recall: {worst_metrics['recall']:.3f}"
        )


async def analyze_failed_queries():
    """Analyze queries that failed to retrieve relevant documents."""
    print("\n" + "=" * 70)
    print("ANALYSIS - Failed Queries")
    print("=" * 70)

    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=None, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(
        retriever=retriever, test_cases_path="data/evaluation/rag_test_cases.json"
    )

    # Run evaluation
    results = await evaluator.evaluate_retrieval(top_k=5, rerank=True)

    # Get failed queries
    failed_queries = results["metrics"]["failed_queries"]

    if not failed_queries:
        print(
            "\n✅ No failed queries! All queries retrieved at least one relevant document."
        )
        return

    print(f"\n⚠️  Found {len(failed_queries)} failed queries:\n")

    for i, failed in enumerate(failed_queries, 1):
        print(f"{i}. [{failed['id']}] ({failed['category']})")
        print(f"   Query: {failed['query']}")
        print()

    # Analyze failure patterns
    category_failures = {}
    for failed in failed_queries:
        cat = failed["category"]
        category_failures[cat] = category_failures.get(cat, 0) + 1

    if category_failures:
        print("📊 Failures by Category:")
        for cat, count in sorted(
            category_failures.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  - {cat}: {count}")

    print("\n💡 Recommendations:")
    print("  - Review chunk size and overlap for affected categories")
    print("  - Consider adding more training data for weak categories")
    print("  - Verify embedding model performance on domain-specific terminology")


async def test_with_metadata_filters():
    """Test retrieval with metadata filtering."""
    print("\n" + "=" * 70)
    print("EVALUATION - With Metadata Filters")
    print("=" * 70)

    vector_store_path = "data/embeddings/faiss_index"

    try:
        vector_store = FAISS.load_local(
            vector_store_path, embeddings=None, allow_dangerous_deserialization=True
        )
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    retriever = AdvancedRetriever(vector_store)

    # Filter to only test queries of a specific category
    all_test_cases_path = "data/evaluation/rag_test_cases.json"
    evaluator = RAGEvaluator(retriever, all_test_cases_path)

    # Get only subscription_billing queries
    subscription_queries = [
        tc
        for tc in evaluator.test_cases
        if tc.get("category") == "subscription_billing"
    ]

    if not subscription_queries:
        print("\n⚠️  No subscription_billing test cases found")
        return

    # Evaluate with metadata filter
    logger.info(f"Evaluating {len(subscription_queries)} subscription_billing queries")
    results = await evaluator.evaluate_retrieval(
        test_queries=subscription_queries,
        top_k=5,
        filter_metadata={"category": "subscription_billing"},
    )

    metrics = results["metrics"]
    print(f"\n📊 FILTERED RESULTS (subscription_billing only):")
    print(f"  Precision@5: {metrics['avg_precision']:.3f}")
    print(f"  Recall@5:    {metrics['avg_recall']:.3f}")
    print(f"  MRR:         {metrics['avg_mrr']:.3f}")
    print(f"  NDCG:        {metrics['avg_ndcg']:.3f}")


async def run_all_scenarios():
    """Run all evaluation scenarios."""
    print("\n" + "🚀" * 35)
    print("RAG PIPELINE - COMPREHENSIVE EVALUATION")
    print("🚀" * 35)

    scenarios = [
        ("Basic Evaluation", run_basic_evaluation),
        ("Top-K Comparison", compare_top_k_values),
        ("Strategy Comparison", compare_retrieval_strategies),
        ("Category Analysis", evaluate_by_category),
        ("Failed Query Analysis", analyze_failed_queries),
        ("Metadata Filtering", test_with_metadata_filters),
    ]

    for i, (name, func) in enumerate(scenarios, 1):
        try:
            await func()
        except Exception as e:
            logger.error(f"Scenario '{name}' failed: {e}")
            print(f"\n❌ {name} failed: {e}")

        # Add separator between scenarios
        if i < len(scenarios):
            print("\n" + "-" * 70)

    print("\n" + "✅" * 35)
    print("EVALUATION COMPLETE")
    print("✅" * 35)


if __name__ == "__main__":
    # Check if vector store exists
    vector_store_path = Path("data/embeddings/faiss_index")

    if not vector_store_path.exists():
        print("⚠️  Vector store not found!")
        print(f"Expected location: {vector_store_path}")
        print("\nPlease run the following steps first:")
        print("  1. Generate embeddings: python scripts/initialize_rag_pipeline.py")
        print("  2. Then run this evaluation script")
        exit(1)

    # Run all evaluation scenarios
    asyncio.run(run_all_scenarios())
