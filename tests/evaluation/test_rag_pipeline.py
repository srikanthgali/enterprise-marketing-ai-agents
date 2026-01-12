"""RAG Pipeline Evaluation Tests with comprehensive metrics and reporting."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from langchain_core.documents import Document

from src.marketing_agents.rag.retriever import AdvancedRetriever

# Configure logging
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG retrieval systems.

    Calculates multiple metrics to assess retrieval quality:
    - Precision@K: Percentage of retrieved documents that are relevant
    - Recall@K: Percentage of relevant documents that were retrieved
    - MRR (Mean Reciprocal Rank): Quality of ranking (1/rank of first relevant doc)
    - NDCG (Normalized Discounted Cumulative Gain): Quality with position discount
    - Response time: Average retrieval latency
    """

    def __init__(self, retriever: AdvancedRetriever, test_cases_path: str):
        """
        Initialize the RAG evaluator.

        Args:
            retriever: AdvancedRetriever instance to evaluate
            test_cases_path: Path to JSON file containing test cases
        """
        self.retriever = retriever
        self.test_cases_path = test_cases_path
        self.test_cases = self._load_test_cases()
        logger.info(f"RAGEvaluator initialized with {len(self.test_cases)} test cases")

    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """
        Load test cases from JSON file.

        Returns:
            List of test case dictionaries

        Raises:
            FileNotFoundError: If test cases file doesn't exist
            json.JSONDecodeError: If file contains invalid JSON
        """
        path = Path(self.test_cases_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Test cases file not found: {self.test_cases_path}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        logger.debug(f"Loaded {len(test_cases)} test cases from {self.test_cases_path}")
        return test_cases

    async def evaluate_retrieval(
        self,
        test_queries: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        rerank: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance across test queries.

        Args:
            test_queries: Optional list of test queries (uses loaded test cases if None)
            top_k: Number of documents to retrieve per query
            rerank: Whether to enable reranking
            filter_metadata: Optional metadata filters to apply

        Returns:
            Dictionary containing aggregated metrics and per-query results
        """
        queries = test_queries if test_queries is not None else self.test_cases

        if not queries:
            raise ValueError("No test queries provided for evaluation")

        logger.info(
            f"Starting evaluation with {len(queries)} queries (top_k={top_k}, rerank={rerank})"
        )

        results = []
        response_times = []

        for test_case in queries:
            query = test_case["query"]
            relevant_doc_ids = set(test_case["relevant_doc_ids"])
            category = test_case.get("category", "general")
            relevance_scores = test_case.get("relevance_scores", None)

            # Retrieve documents and measure time
            start_time = time.time()
            retrieved_docs = await self.retriever.retrieve(
                query=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
                rerank=rerank,
            )
            response_time = time.time() - start_time
            response_times.append(response_time)

            # Extract document IDs from retrieved documents
            retrieved_doc_ids = [
                doc.metadata.get("doc_id", doc.metadata.get("chunk_id", f"doc_{i}"))
                for i, doc in enumerate(retrieved_docs)
            ]

            # Calculate metrics for this query
            precision = self._precision_at_k(retrieved_doc_ids, list(relevant_doc_ids))
            recall = self._recall_at_k(retrieved_doc_ids, list(relevant_doc_ids))
            mrr = self._mean_reciprocal_rank(retrieved_doc_ids, list(relevant_doc_ids))
            ndcg = self._ndcg(
                retrieved_doc_ids, list(relevant_doc_ids), relevance_scores
            )

            # Calculate F1 score
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            result = {
                "query_id": test_case.get("id", "unknown"),
                "query": query,
                "category": category,
                "precision": precision,
                "recall": recall,
                "mrr": mrr,
                "ndcg": ndcg,
                "f1": f1,
                "response_time": response_time,
                "retrieved_count": len(retrieved_docs),
                "relevant_count": len(relevant_doc_ids),
                "retrieved_doc_ids": retrieved_doc_ids,
            }

            results.append(result)

            logger.debug(
                f"Query '{query[:50]}...': P={precision:.2f}, R={recall:.2f}, "
                f"MRR={mrr:.2f}, NDCG={ndcg:.2f}, Time={response_time:.3f}s"
            )

        # Calculate aggregate metrics
        metrics = self._calculate_aggregate_metrics(results, response_times)

        logger.info(
            f"Evaluation complete. Avg Precision: {metrics['avg_precision']:.3f}, "
            f"Avg Recall: {metrics['avg_recall']:.3f}, Avg MRR: {metrics['avg_mrr']:.3f}"
        )

        return {
            "metrics": metrics,
            "per_query_results": results,
            "config": {
                "top_k": top_k,
                "rerank": rerank,
                "filter_metadata": filter_metadata,
                "num_queries": len(queries),
            },
        }

    def _precision_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Precision@K: percentage of retrieved docs that are relevant.

        Precision@K = (# relevant docs in top K) / K

        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs

        Returns:
            Precision score between 0.0 and 1.0
        """
        if not retrieved:
            return 0.0

        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_set)

        return relevant_retrieved / len(retrieved)

    def _recall_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Recall@K: percentage of relevant docs that were retrieved.

        Recall@K = (# relevant docs in top K) / (total # relevant docs)

        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs

        Returns:
            Recall score between 0.0 and 1.0
        """
        if not relevant:
            return 0.0

        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_set)

        return relevant_retrieved / len(relevant)

    def _mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank of first relevant document
        Returns 0 if no relevant documents are retrieved.

        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs

        Returns:
            MRR score (0.0 to 1.0, where 1.0 means first doc is relevant)
        """
        relevant_set = set(relevant)

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    def _ndcg(
        self,
        retrieved: List[str],
        relevant: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).

        NDCG measures ranking quality with position discount:
        - DCG = sum(relevance / log2(position + 1))
        - NDCG = DCG / IDCG (ideal DCG)

        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: List of relevant document IDs
            relevance_scores: Optional list of relevance scores (0-1) for each relevant doc
                             If not provided, binary relevance (1 for relevant, 0 otherwise) is used

        Returns:
            NDCG score between 0.0 and 1.0
        """
        if not retrieved or not relevant:
            return 0.0

        # Create relevance mapping
        if relevance_scores is not None and len(relevance_scores) == len(relevant):
            relevance_map = dict(zip(relevant, relevance_scores))
        else:
            # Binary relevance: 1 for relevant, 0 for not relevant
            relevance_map = {doc_id: 1.0 for doc_id in relevant}

        # Calculate DCG for retrieved documents
        dcg = 0.0
        for position, doc_id in enumerate(retrieved, start=1):
            relevance = relevance_map.get(doc_id, 0.0)
            dcg += relevance / np.log2(position + 1)

        # Calculate ideal DCG (IDCG) - best possible ranking
        ideal_relevances = sorted(
            [relevance_map.get(doc_id, 0.0) for doc_id in relevant], reverse=True
        )
        idcg = 0.0
        for position, relevance in enumerate(
            ideal_relevances[: len(retrieved)], start=1
        ):
            idcg += relevance / np.log2(position + 1)

        # Normalize
        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def _calculate_aggregate_metrics(
        self, results: List[Dict[str, Any]], response_times: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all test queries.

        Args:
            results: List of per-query results
            response_times: List of response times

        Returns:
            Dictionary of aggregate metrics
        """
        if not results:
            return {}

        # Calculate averages
        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])
        avg_mrr = np.mean([r["mrr"] for r in results])
        avg_ndcg = np.mean([r["ndcg"] for r in results])
        avg_f1 = np.mean([r["f1"] for r in results])
        avg_response_time = np.mean(response_times)

        # Calculate per-category metrics
        category_metrics = {}
        categories = set(r["category"] for r in results)
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            category_metrics[category] = {
                "count": len(cat_results),
                "precision": np.mean([r["precision"] for r in cat_results]),
                "recall": np.mean([r["recall"] for r in cat_results]),
                "mrr": np.mean([r["mrr"] for r in cat_results]),
                "ndcg": np.mean([r["ndcg"] for r in cat_results]),
                "f1": np.mean([r["f1"] for r in cat_results]),
            }

        # Identify failed queries (zero recall)
        failed_queries = [r for r in results if r["recall"] == 0.0]

        return {
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_mrr": float(avg_mrr),
            "avg_ndcg": float(avg_ndcg),
            "avg_f1": float(avg_f1),
            "avg_response_time": float(avg_response_time),
            "total_queries": len(results),
            "failed_queries_count": len(failed_queries),
            "category_metrics": category_metrics,
            "failed_queries": [
                {"id": q["query_id"], "query": q["query"], "category": q["category"]}
                for q in failed_queries
            ],
        }

    def generate_report(
        self, metrics: Dict[str, Any], output_format: str = "markdown"
    ) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            metrics: Metrics dictionary from evaluate_retrieval()
            output_format: Output format ("markdown" or "text")

        Returns:
            Formatted report string
        """
        if output_format == "markdown":
            return self._generate_markdown_report(metrics)
        else:
            return self._generate_text_report(metrics)

    def _generate_markdown_report(self, metrics: Dict[str, Any]) -> str:
        """Generate evaluation report in Markdown format."""
        report = []
        report.append("# RAG Pipeline Evaluation Report\n")
        report.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        # Configuration
        config = metrics.get("config", {})
        report.append("## Configuration\n")
        report.append(f"- **Top K:** {config.get('top_k', 'N/A')}")
        report.append(f"- **Reranking:** {config.get('rerank', 'N/A')}")
        report.append(f"- **Filter Metadata:** {config.get('filter_metadata', 'None')}")
        report.append(f"- **Number of Queries:** {config.get('num_queries', 'N/A')}\n")

        # Overall metrics
        overall = metrics.get("metrics", {})
        report.append("## Overall Metrics\n")
        report.append("| Metric | Value | Target |")
        report.append("|--------|-------|--------|")
        report.append(
            f"| **Precision@{config.get('top_k', 'K')}** | "
            f"{overall.get('avg_precision', 0):.3f} | > 0.80 |"
        )
        report.append(
            f"| **Recall@{config.get('top_k', 'K')}** | "
            f"{overall.get('avg_recall', 0):.3f} | > 0.70 |"
        )
        report.append(f"| **MRR** | {overall.get('avg_mrr', 0):.3f} | > 0.85 |")
        report.append(f"| **NDCG** | {overall.get('avg_ndcg', 0):.3f} | > 0.80 |")
        report.append(f"| **F1 Score** | {overall.get('avg_f1', 0):.3f} | > 0.75 |")
        report.append(
            f"| **Avg Response Time** | "
            f"{overall.get('avg_response_time', 0):.3f}s | < 1.0s |\n"
        )

        # Category breakdown
        category_metrics = overall.get("category_metrics", {})
        if category_metrics:
            report.append("## Per-Category Performance\n")
            report.append("| Category | Count | Precision | Recall | MRR | NDCG | F1 |")
            report.append(
                "|----------|-------|-----------|--------|-----|------|-----|"
            )
            for category, cat_metrics in sorted(category_metrics.items()):
                report.append(
                    f"| {category} | {cat_metrics['count']} | "
                    f"{cat_metrics['precision']:.3f} | {cat_metrics['recall']:.3f} | "
                    f"{cat_metrics['mrr']:.3f} | {cat_metrics['ndcg']:.3f} | "
                    f"{cat_metrics['f1']:.3f} |"
                )
            report.append("")

        # Failed queries
        failed = overall.get("failed_queries", [])
        if failed:
            report.append(f"## Failed Queries ({len(failed)} total)\n")
            report.append("Queries that retrieved no relevant documents:\n")
            for query in failed[:10]:  # Show first 10
                report.append(
                    f"- **[{query['id']}]** ({query['category']}): {query['query']}"
                )
            if len(failed) > 10:
                report.append(f"\n*...and {len(failed) - 10} more*\n")
        else:
            report.append("## Failed Queries\n")
            report.append(
                "✅ No failed queries - all queries retrieved at least one relevant document!\n"
            )

        # Performance summary
        report.append("## Summary\n")
        precision = overall.get("avg_precision", 0)
        recall = overall.get("avg_recall", 0)
        mrr = overall.get("avg_mrr", 0)

        meets_targets = precision > 0.80 and recall > 0.70 and mrr > 0.85
        if meets_targets:
            report.append("✅ **System meets all target metrics!**\n")
        else:
            report.append("⚠️ **System does not meet all target metrics:**\n")
            if precision <= 0.80:
                report.append(f"  - Precision@K: {precision:.3f} (target: > 0.80)")
            if recall <= 0.70:
                report.append(f"  - Recall@K: {recall:.3f} (target: > 0.70)")
            if mrr <= 0.85:
                report.append(f"  - MRR: {mrr:.3f} (target: > 0.85)")
            report.append("")

        return "\n".join(report)

    def _generate_text_report(self, metrics: Dict[str, Any]) -> str:
        """Generate evaluation report in plain text format."""
        report = []
        report.append("=" * 70)
        report.append("RAG PIPELINE EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)

        overall = metrics.get("metrics", {})
        report.append("\nOVERALL METRICS:")
        report.append(f"  Precision@K: {overall.get('avg_precision', 0):.3f}")
        report.append(f"  Recall@K:    {overall.get('avg_recall', 0):.3f}")
        report.append(f"  MRR:         {overall.get('avg_mrr', 0):.3f}")
        report.append(f"  NDCG:        {overall.get('avg_ndcg', 0):.3f}")
        report.append(f"  F1 Score:    {overall.get('avg_f1', 0):.3f}")
        report.append(f"  Avg Time:    {overall.get('avg_response_time', 0):.3f}s")

        failed = overall.get("failed_queries", [])
        report.append(f"\nFAILED QUERIES: {len(failed)}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)

    def export_results(
        self, metrics: Dict[str, Any], output_dir: str = "data/evaluation"
    ):
        """
        Export evaluation results to JSON and Markdown files.

        Args:
            metrics: Metrics dictionary from evaluate_retrieval()
            output_dir: Directory to save output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export JSON
        json_file = output_path / f"rag_evaluation_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported JSON results to {json_file}")

        # Export Markdown report
        md_file = output_path / f"rag_evaluation_{timestamp}.md"
        report = self.generate_report(metrics, output_format="markdown")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Exported Markdown report to {md_file}")

        return {"json": str(json_file), "markdown": str(md_file)}


# ============================================================================
# Pytest Test Cases
# ============================================================================


@pytest.mark.asyncio
class TestRAGPipeline:
    """Test suite for RAG pipeline evaluation."""

    @pytest.fixture
    def mock_retriever(self, tmp_path):
        """Create a mock retriever for testing."""

        # This would normally be a real retriever with a vector store
        # For testing purposes, we'll create a mock
        class MockRetriever:
            async def retrieve(self, query, top_k=5, filter_metadata=None, rerank=True):
                # Return mock documents
                docs = [
                    Document(
                        page_content=f"Mock content for {query}",
                        metadata={"doc_id": f"chunk_{i}", "category": "test"},
                    )
                    for i in range(top_k)
                ]
                return docs

            def clear_cache(self):
                pass

        return MockRetriever()

    @pytest.fixture
    def test_cases_file(self, tmp_path):
        """Create a temporary test cases file."""
        test_cases = {
            "test_cases": [
                {
                    "id": "test_001",
                    "query": "How does Stripe handle subscription refunds?",
                    "relevant_doc_ids": ["chunk_0", "chunk_1", "chunk_2"],
                    "category": "subscription_billing",
                },
                {
                    "id": "test_002",
                    "query": "What is the Segment CDP?",
                    "relevant_doc_ids": ["chunk_1", "chunk_3"],
                    "category": "conceptual",
                },
            ]
        }

        test_file = tmp_path / "test_cases.json"
        with open(test_file, "w") as f:
            json.dump(test_cases, f)

        return str(test_file)

    @pytest.mark.asyncio
    async def test_evaluator_initialization(self, mock_retriever, test_cases_file):
        """Test RAGEvaluator initialization."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)
        assert evaluator.retriever is not None
        assert len(evaluator.test_cases) == 2

    @pytest.mark.asyncio
    async def test_precision_calculation(self, mock_retriever, test_cases_file):
        """Test precision@k metric calculation."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)

        # Test perfect precision
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        assert evaluator._precision_at_k(retrieved, relevant) == 1.0

        # Test partial precision
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc4"]
        assert evaluator._precision_at_k(retrieved, relevant) == pytest.approx(
            1 / 3, rel=1e-2
        )

    @pytest.mark.asyncio
    async def test_recall_calculation(self, mock_retriever, test_cases_file):
        """Test recall@k metric calculation."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)

        # Test perfect recall
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2"]
        assert evaluator._recall_at_k(retrieved, relevant) == 1.0

        # Test partial recall
        retrieved = ["doc1", "doc2"]
        relevant = ["doc1", "doc3", "doc4"]
        assert evaluator._recall_at_k(retrieved, relevant) == pytest.approx(
            1 / 3, rel=1e-2
        )

    @pytest.mark.asyncio
    async def test_mrr_calculation(self, mock_retriever, test_cases_file):
        """Test MRR metric calculation."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)

        # Test first position
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        assert evaluator._mean_reciprocal_rank(retrieved, relevant) == 1.0

        # Test third position
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc3"]
        assert evaluator._mean_reciprocal_rank(retrieved, relevant) == pytest.approx(
            1 / 3, rel=1e-2
        )

        # Test no relevant docs
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc4"]
        assert evaluator._mean_reciprocal_rank(retrieved, relevant) == 0.0

    @pytest.mark.asyncio
    async def test_ndcg_calculation(self, mock_retriever, test_cases_file):
        """Test NDCG metric calculation."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)

        # Test perfect ranking
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        ndcg = evaluator._ndcg(retrieved, relevant)
        assert ndcg == pytest.approx(1.0, rel=1e-2)

        # Test with relevance scores
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        relevance_scores = [1.0, 0.8, 0.5]
        ndcg = evaluator._ndcg(retrieved, relevant, relevance_scores)
        assert 0.0 <= ndcg <= 1.0

    @pytest.mark.asyncio
    async def test_full_evaluation(self, mock_retriever, test_cases_file):
        """Test full evaluation pipeline."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)
        results = await evaluator.evaluate_retrieval(top_k=5)

        assert "metrics" in results
        assert "per_query_results" in results
        assert "config" in results

        metrics = results["metrics"]
        assert "avg_precision" in metrics
        assert "avg_recall" in metrics
        assert "avg_mrr" in metrics
        assert "avg_ndcg" in metrics

    @pytest.mark.asyncio
    async def test_report_generation(self, mock_retriever, test_cases_file):
        """Test report generation."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)
        results = await evaluator.evaluate_retrieval(top_k=5)

        # Test markdown report
        md_report = evaluator.generate_report(results, output_format="markdown")
        assert "# RAG Pipeline Evaluation Report" in md_report
        assert "Precision" in md_report

        # Test text report
        text_report = evaluator.generate_report(results, output_format="text")
        assert "EVALUATION REPORT" in text_report

    @pytest.mark.asyncio
    async def test_export_results(self, mock_retriever, test_cases_file, tmp_path):
        """Test exporting results to files."""
        evaluator = RAGEvaluator(mock_retriever, test_cases_file)
        results = await evaluator.evaluate_retrieval(top_k=5)

        output_dir = str(tmp_path / "evaluation_output")
        files = evaluator.export_results(results, output_dir=output_dir)

        assert Path(files["json"]).exists()
        assert Path(files["markdown"]).exists()


if __name__ == "__main__":
    """Example usage of RAGEvaluator."""
    # This would normally use a real retriever with a vector store
    print("RAG Evaluation Module")
    print("=" * 70)
    print("To run evaluations, use:")
    print("  pytest tests/evaluation/test_rag_pipeline.py -v")
    print("\nOr programmatically:")
    print(
        """
    from src.marketing_agents.rag.retriever import AdvancedRetriever
    from tests.evaluation.test_rag_pipeline import RAGEvaluator

    # Initialize with your retriever
    retriever = AdvancedRetriever(vector_store)
    evaluator = RAGEvaluator(retriever, "data/evaluation/rag_test_cases.json")

    # Run evaluation
    results = await evaluator.evaluate_retrieval(top_k=5)

    # Print metrics
    print(f"Precision@5: {results['metrics']['avg_precision']:.2f}")
    print(f"Recall@5: {results['metrics']['avg_recall']:.2f}")
    print(f"MRR: {results['metrics']['avg_mrr']:.2f}")

    # Generate and export report
    evaluator.export_results(results)
    """
    )
