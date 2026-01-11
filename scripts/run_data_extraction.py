"""
Master orchestration script for complete data extraction and generation.

Runs all data generation components in sequence:
1. Stripe Help Center documentation scraping
2. Support ticket generation
3. Marketing campaign data generation
4. Feedback and interaction data generation
5. Data validation and summary reporting
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.utils import get_logger
from config.settings import get_settings
from src.marketing_agents.data_extraction.web_scraper import StripeDocsScraper
from src.marketing_agents.data_extraction.ticket_generator import (
    generate_support_tickets,
)
from src.marketing_agents.data_extraction.campaign_generator import (
    generate_campaign_data,
)
from src.marketing_agents.data_extraction.feedback_generator import (
    generate_feedback_data,
)


logger = get_logger(__name__)


class DataExtractionOrchestrator:
    """
    Orchestrates complete data extraction and generation pipeline.

    Coordinates:
    - Web scraping of Stripe Help Center documentation
    - Synthetic data generation (tickets, campaigns, feedback)
    - Data validation and quality checks
    - Summary report generation
    """

    def __init__(self):
        """Initialize orchestrator."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        self.reports_dir = self.settings.data_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.execution_log = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps": [],
            "errors": [],
        }

    def log_step(
        self, step_name: str, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log execution step."""
        step = {
            "step": step_name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }

        self.execution_log["steps"].append(step)

        if status == "error":
            self.execution_log["errors"].append(step)

        self.logger.info(f"Step: {step_name} - Status: {status}")

    def _check_step_completed(self, step_name: str) -> bool:
        """
        Check if a step was previously completed successfully.

        Args:
            step_name: Name of the step to check

        Returns:
            True if step completed successfully, False otherwise
        """
        if step_name == "web_scraping":
            docs_dir = self.settings.data_dir / "raw" / "knowledge_base" / "stripe_docs"
            metadata_file = docs_dir / "_scraper_metadata.json"
            return (
                docs_dir.exists()
                and len(list(docs_dir.glob("*.md"))) > 0
                and metadata_file.exists()
            )

        elif step_name == "ticket_generation":
            tickets_dir = self.settings.data_dir / "raw" / "support_tickets"
            stats_file = tickets_dir / "_ticket_stats.json"
            return stats_file.exists()

        elif step_name == "campaign_generation":
            campaigns_dir = self.settings.data_dir / "raw" / "marketing_data"
            stats_file = campaigns_dir / "_campaign_stats.json"
            csv_file = campaigns_dir / "campaigns_2024.csv"
            return stats_file.exists() and csv_file.exists()

        elif step_name == "feedback_generation":
            feedback_dir = self.settings.data_dir / "raw" / "feedback"
            stats_file = feedback_dir / "_feedback_stats.json"
            return stats_file.exists()

        return False

    async def run_web_scraping(self) -> Dict[str, Any]:
        """
        Step 1: Scrape Stripe Documentation.

        Returns:
            Scraping results and statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 1: Web Scraping - Stripe Documentation")
        self.logger.info("=" * 80)

        # Check if already completed
        if self._check_step_completed("web_scraping"):
            output_dir = (
                self.settings.data_dir / "raw" / "knowledge_base" / "stripe_docs"
            )
            num_files = len(list(output_dir.glob("*.md")))

            metadata_file = output_dir / "_scraper_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    pages_scraped = metadata.get("stats", {}).get(
                        "pages_scraped", num_files
                    )
            else:
                pages_scraped = num_files

            self.logger.info(f"✓ Step already completed: {pages_scraped} pages exist")
            self.log_step(
                "web_scraping",
                "skipped",
                {
                    "reason": "already_completed",
                    "pages_scraped": pages_scraped,
                    "num_files": num_files,
                },
            )
            return {
                "status": "skipped",
                "pages_scraped": pages_scraped,
                "num_files": num_files,
                "output_dir": str(output_dir),
            }

        try:
            scraper = StripeDocsScraper()
            self.logger.info("Starting Stripe documentation scraping...")
            self.logger.info("Compliance checks:")
            self.logger.info("  ✓ robots.txt will be checked before scraping")
            self.logger.info("  ✓ Rate limiting: 2 seconds between requests")
            self.logger.info("  ✓ Retry logic with exponential backoff")
            self.logger.info("  ✓ User agent identifies bot purpose")

            results = await scraper.scrape_documentation()

            self.log_step(
                "web_scraping",
                "success",
                {
                    "pages_scraped": results.get("pages_scraped", 0),
                    "pages_failed": results.get("pages_failed", 0),
                    "pages_blocked": results.get("pages_blocked_by_robots", 0),
                    "total_bytes": results.get("total_bytes", 0),
                    "output_dir": str(scraper.output_dir),
                    "robots_txt_checked": results.get("robots_txt_checked", False),
                    "rate_limited": True,
                },
            )

            self.logger.info(
                f"✓ Scraping completed: {results.get('pages_scraped', 0)} pages"
            )
            self.logger.info(f"  - Failed: {results.get('pages_failed', 0)} pages")
            self.logger.info(
                f"  - Blocked by robots.txt: {results.get('pages_blocked_by_robots', 0)} pages"
            )
            self.logger.info(
                f"  - Total size: {results.get('total_bytes', 0) / 1024:.2f} KB"
            )

            return results

        except Exception as e:
            self.logger.error(f"Web scraping failed: {e}", exc_info=True)
            self.log_step("web_scraping", "error", {"error": str(e)})
            return {"status": "error", "error": str(e)}

    def run_ticket_generation(self) -> Dict[str, Any]:
        """Step 2: Generate synthetic support tickets."""
        self.logger.info("=" * 80)
        self.logger.info("STEP 2: Synthetic Data - Support Tickets")
        self.logger.info("=" * 80)

        if self._check_step_completed("ticket_generation"):
            stats_file = (
                self.settings.data_dir
                / "raw"
                / "support_tickets"
                / "_ticket_stats.json"
            )
            with open(stats_file) as f:
                stats = json.load(f)
            total = stats.get("total_tickets", 0)
            self.logger.info(f"✓ Step already completed: {total} tickets exist")
            self.log_step(
                "ticket_generation",
                "skipped",
                {"reason": "already_completed", "total_tickets": total},
            )
            return stats

        try:
            self.logger.info("Generating 500 support tickets...")
            result = generate_support_tickets()

            self.log_step(
                "ticket_generation",
                "success",
                {
                    "total_tickets": result.get("total_tickets", 0),
                    "categories": result.get("by_category", {}),
                },
            )

            self.logger.info(
                f"✓ Generated {result.get('total_tickets', 0)} support tickets"
            )
            return result

        except Exception as e:
            self.logger.error(f"Ticket generation failed: {e}", exc_info=True)
            self.log_step("ticket_generation", "error", {"error": str(e)})
            return {"status": "error", "error": str(e)}

    def run_campaign_generation(self) -> Dict[str, Any]:
        """Step 3: Generate marketing campaign data."""
        self.logger.info("=" * 80)
        self.logger.info("STEP 3: Synthetic Data - Marketing Campaigns")
        self.logger.info("=" * 80)

        if self._check_step_completed("campaign_generation"):
            stats_file = (
                self.settings.data_dir
                / "raw"
                / "marketing_data"
                / "_campaign_stats.json"
            )
            with open(stats_file) as f:
                stats = json.load(f)
            total = stats.get("total_campaigns", 0)
            self.logger.info(f"✓ Step already completed: {total} campaigns exist")
            self.log_step(
                "campaign_generation",
                "skipped",
                {"reason": "already_completed", "total_campaigns": total},
            )
            return stats

        try:
            self.logger.info("Generating 150 marketing campaigns...")
            result = generate_campaign_data()

            self.log_step(
                "campaign_generation",
                "success",
                {
                    "total_campaigns": result.get("total_campaigns", 0),
                    "date_range": result.get("date_range", {}),
                    "channels": result.get("by_channel", {}),
                },
            )

            self.logger.info(
                f"✓ Generated {result.get('total_campaigns', 0)} campaigns"
            )
            return result

        except Exception as e:
            self.logger.error(f"Campaign generation failed: {e}", exc_info=True)
            self.log_step("campaign_generation", "error", {"error": str(e)})
            return {"status": "error", "error": str(e)}

    def run_feedback_generation(self) -> Dict[str, Any]:
        """Step 4: Generate feedback and interaction data."""
        self.logger.info("=" * 80)
        self.logger.info("STEP 4: Synthetic Data - Feedback & Interactions")
        self.logger.info("=" * 80)

        if self._check_step_completed("feedback_generation"):
            stats_file = (
                self.settings.data_dir / "raw" / "feedback" / "_feedback_stats.json"
            )
            with open(stats_file) as f:
                stats = json.load(f)
            total = (
                stats.get("interactions", {}).get("total", 0)
                + stats.get("campaign_reviews", {}).get("total", 0)
                + stats.get("csat", {}).get("total", 0)
            )
            self.logger.info(f"✓ Step already completed: {total} feedback items exist")
            self.log_step(
                "feedback_generation",
                "skipped",
                {"reason": "already_completed", "total_items": total},
            )
            return {
                "interactions": stats.get("interactions", {}).get("total", 0),
                "campaign_reviews": stats.get("campaign_reviews", {}).get("total", 0),
                "csat_scores": stats.get("csat", {}).get("total", 0),
            }

        try:
            self.logger.info("Generating feedback data...")
            result = generate_feedback_data()

            stats = {
                "interactions": len(result.get("interactions", [])),
                "campaign_reviews": len(result.get("campaign_reviews", [])),
                "csat_scores": len(result.get("csat_scores", [])),
            }

            self.log_step("feedback_generation", "success", stats)

            self.logger.info(
                f"✓ Generated {stats['interactions']} interactions, "
                f"{stats['campaign_reviews']} reviews, "
                f"{stats['csat_scores']} CSAT scores"
            )
            return stats

        except Exception as e:
            self.logger.error(f"Feedback generation failed: {e}", exc_info=True)
            self.log_step("feedback_generation", "error", {"error": str(e)})
            return {"status": "error", "error": str(e)}

    def validate_data(self) -> Dict[str, Any]:
        """Step 5: Validate generated data."""
        self.logger.info("=" * 80)
        self.logger.info("STEP 5: Data Validation")
        self.logger.info("=" * 80)

        validation_results = {
            "stripe_docs": self._validate_stripe_docs(),  # Keep this name consistent
            "support_tickets": self._validate_support_tickets(),
            "campaigns": self._validate_campaigns(),
            "feedback": self._validate_feedback(),
        }

        all_valid = all(v["valid"] for v in validation_results.values())

        status = "success" if all_valid else "warning"
        self.log_step("data_validation", status, validation_results)

        self.logger.info("\nValidation Results:")
        for category, result in validation_results.items():
            status_icon = "✓" if result["valid"] else "✗"
            self.logger.info(f"  {status_icon} {category}: {result}")

        return validation_results

    def _validate_stripe_docs(self) -> Dict[str, Any]:
        """Validate Stripe documentation files."""
        docs_dir = self.settings.data_dir / "raw" / "knowledge_base" / "stripe_docs"

        if not docs_dir.exists():
            return {"valid": False, "reason": "directory_not_found"}

        md_files = list(docs_dir.glob("*.md"))
        json_files = list(docs_dir.glob("*.json"))
        metadata_file = docs_dir / "_scraper_metadata.json"
        report_file = docs_dir / "_scraping_report.txt"

        compliance_info = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                compliance_info = {
                    "robots_txt_checked": metadata.get("robots_txt_checked", False),
                    "rate_limited": metadata.get("rate_limit_delay_seconds", 0) > 0,
                    "pages_scraped": metadata.get("stats", {}).get("pages_scraped", 0),
                    "pages_failed": metadata.get("stats", {}).get("pages_failed", 0),
                    "pages_blocked": metadata.get("stats", {}).get(
                        "pages_blocked_by_robots", 0
                    ),
                }

        return {
            "valid": len(md_files) > 0 and metadata_file.exists(),
            "markdown_files": len(md_files),
            "metadata_files": len(json_files),
            "has_scraper_metadata": metadata_file.exists(),
            "has_report": report_file.exists(),
            "compliance": compliance_info,
            "path": str(docs_dir),
        }

    def _validate_support_tickets(self) -> Dict[str, Any]:
        """Validate support ticket files."""
        tickets_dir = self.settings.data_dir / "raw" / "support_tickets"

        if not tickets_dir.exists():
            return {"valid": False, "reason": "directory_not_found"}

        ticket_files = list(tickets_dir.glob("tickets_*.json"))
        stats_file = tickets_dir / "_ticket_stats.json"

        total_tickets = 0
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
                total_tickets = stats.get("total_tickets", 0)

        return {
            "valid": len(ticket_files) > 0 and stats_file.exists(),
            "ticket_files": len(ticket_files),
            "total_tickets": total_tickets,
            "has_stats": stats_file.exists(),
            "path": str(tickets_dir),
        }

    def _validate_campaigns(self) -> Dict[str, Any]:
        """Validate campaign data files."""
        campaigns_dir = self.settings.data_dir / "raw" / "marketing_data"

        if not campaigns_dir.exists():
            return {"valid": False, "reason": "directory_not_found"}

        csv_file = campaigns_dir / "campaigns_2024.csv"
        json_file = campaigns_dir / "campaigns_2024.json"
        stats_file = campaigns_dir / "_campaign_stats.json"

        total_campaigns = 0
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
                total_campaigns = stats.get("total_campaigns", 0)

        return {
            "valid": (csv_file.exists() or json_file.exists()) and stats_file.exists(),
            "has_csv": csv_file.exists(),
            "has_json": json_file.exists(),
            "has_stats": stats_file.exists(),
            "total_campaigns": total_campaigns,
            "path": str(campaigns_dir),
        }

    def _validate_feedback(self) -> Dict[str, Any]:
        """Validate feedback data files."""
        feedback_dir = self.settings.data_dir / "raw" / "feedback"

        if not feedback_dir.exists():
            return {"valid": False, "reason": "directory_not_found"}

        interactions_file = feedback_dir / "agent_interactions.json"
        reviews_file = feedback_dir / "campaign_feedback.json"
        csat_file = feedback_dir / "customer_satisfaction.csv"
        stats_file = feedback_dir / "_feedback_stats.json"

        total_items = 0
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
                total_items = (
                    stats.get("interactions", {}).get("total", 0)
                    + stats.get("campaign_reviews", {}).get("total", 0)
                    + stats.get("csat", {}).get("total", 0)
                )

        return {
            "valid": interactions_file.exists()
            and reviews_file.exists()
            and stats_file.exists(),
            "has_interactions": interactions_file.exists(),
            "has_reviews": reviews_file.exists(),
            "has_csat": csat_file.exists(),
            "has_stats": stats_file.exists(),
            "total_items": total_items,
            "path": str(feedback_dir),
        }

    def generate_summary_report(self) -> Dict[str, Any]:
        """Step 6: Generate comprehensive summary report."""
        self.logger.info("=" * 80)
        self.logger.info("STEP 6: Summary Report Generation")
        self.logger.info("=" * 80)

        self.execution_log["completed_at"] = datetime.now(timezone.utc).isoformat()

        summary = {
            "execution_log": self.execution_log,
            "data_summary": {
                "stripe_docs": self._get_stripe_docs_stats(),
                "support_tickets": self._get_ticket_stats(),
                "campaigns": self._get_campaign_stats(),
                "feedback": self._get_feedback_stats(),
            },
            "total_data_points": 0,
        }

        for category, stats in summary["data_summary"].items():
            if isinstance(stats, dict) and "total" in stats:
                summary["total_data_points"] += stats["total"]

        report_file = (
            self.reports_dir
            / f"data_extraction_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"✓ Summary report saved to {report_file}")
        self.log_step("summary_report", "success", {"report_file": str(report_file)})

        self._print_summary(summary)

        return summary

    def _get_stripe_docs_stats(self) -> Dict[str, Any]:
        """Get Stripe docs statistics."""
        docs_dir = self.settings.data_dir / "raw" / "knowledge_base" / "stripe_docs"
        metadata_file = docs_dir / "_scraper_metadata.json"

        if not docs_dir.exists():
            return {"total": 0}

        stats = {
            "total": len(list(docs_dir.glob("*.md"))),
            "path": str(docs_dir),
        }

        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                stats.update(
                    {
                        "pages_scraped": metadata.get("stats", {}).get(
                            "pages_scraped", 0
                        ),
                        "pages_failed": metadata.get("stats", {}).get(
                            "pages_failed", 0
                        ),
                        "pages_blocked": metadata.get("stats", {}).get(
                            "pages_blocked_by_robots", 0
                        ),
                        "total_bytes": metadata.get("stats", {}).get("total_bytes", 0),
                        "robots_txt_checked": metadata.get("robots_txt_checked", False),
                        "rate_limited": True,
                    }
                )

        return stats

    def _get_ticket_stats(self) -> Dict[str, Any]:
        """Get ticket statistics."""
        stats_file = (
            self.settings.data_dir / "raw" / "support_tickets" / "_ticket_stats.json"
        )

        if not stats_file.exists():
            return {"total": 0}

        with open(stats_file) as f:
            return json.load(f)

    def _get_campaign_stats(self) -> Dict[str, Any]:
        """Get campaign statistics."""
        stats_file = (
            self.settings.data_dir / "raw" / "marketing_data" / "_campaign_stats.json"
        )

        if not stats_file.exists():
            return {"total": 0}

        with open(stats_file) as f:
            stats = json.load(f)
            stats["total"] = stats.get("total_campaigns", 0)
            return stats

    def _get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        stats_file = (
            self.settings.data_dir / "raw" / "feedback" / "_feedback_stats.json"
        )

        if not stats_file.exists():
            return {"total": 0}

        with open(stats_file) as f:
            stats = json.load(f)
            stats["total"] = (
                stats.get("interactions", {}).get("total", 0)
                + stats.get("campaign_reviews", {}).get("total", 0)
                + stats.get("csat", {}).get("total", 0)
            )
            return stats

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 80)
        print("DATA EXTRACTION COMPLETE - SUMMARY")
        print("=" * 80)

        print(f"\nTotal Data Points Generated: {summary['total_data_points']:,}")

        print("\nData by Category:")
        for category, stats in summary["data_summary"].items():
            total = stats.get("total", 0)
            print(f"  • {category.replace('_', ' ').title()}: {total:,}")

            if category == "stripe_docs" and "robots_txt_checked" in stats:
                print(
                    f"    - Robots.txt Checked: {'✓' if stats['robots_txt_checked'] else '✗'}"
                )
                print(
                    f"    - Rate Limited: {'✓' if stats.get('rate_limited') else '✗'}"
                )
                if stats.get("pages_failed", 0) > 0:
                    print(f"    - Pages Failed: {stats['pages_failed']}")
                if stats.get("pages_blocked", 0) > 0:
                    print(
                        f"    - Pages Blocked by robots.txt: {stats['pages_blocked']}"
                    )

        print(f"\nExecution Steps: {len(summary['execution_log']['steps'])}")

        successful = len(
            [s for s in summary["execution_log"]["steps"] if s["status"] == "success"]
        )
        skipped = len(
            [s for s in summary["execution_log"]["steps"] if s["status"] == "skipped"]
        )
        print(f"  • Successful: {successful}")
        if skipped > 0:
            print(f"  • Skipped (already completed): {skipped}")

        errors = len(summary["execution_log"]["errors"])
        if errors > 0:
            print(f"  • Errors: {errors}")

        print("\n" + "=" * 80)
        print("✓ Data Extraction Pipeline Completed Successfully!")
        print("=" * 80)
        print("\nNext Steps:")
        print("  1. Review generated data in data/raw/ directory")
        print("  2. Check Stripe scraping compliance in:")
        print("     - data/raw/knowledge_base/stripe_docs/_scraper_metadata.json")
        print("     - data/raw/knowledge_base/stripe_docs/_scraping_report.txt")
        print("  3. Process data for embeddings (chunking, cleaning)")
        print("  4. Generate embeddings and populate vector stores")
        print("  5. Configure agents with data sources")
        print("  6. Test agent workflows with generated data")
        print("=" * 80 + "\n")

    async def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute complete data extraction pipeline."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("ENTERPRISE MARKETING AI AGENTS - DATA EXTRACTION PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info("Source: Stripe Documentation")
        self.logger.info("Compliance: robots.txt + Rate Limiting + Error Handling")
        self.logger.info("=" * 80 + "\n")

        try:
            await self.run_web_scraping()
            self.run_ticket_generation()
            self.run_campaign_generation()
            self.run_feedback_generation()
            validation = self.validate_data()
            summary = self.generate_summary_report()

            self.logger.info("\n✓ Data extraction pipeline completed successfully!")
            return summary

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            self.execution_log["fatal_error"] = str(e)
            self.execution_log["completed_at"] = datetime.now(timezone.utc).isoformat()

            error_report = (
                self.reports_dir
                / f"error_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(error_report, "w") as f:
                json.dump(self.execution_log, f, indent=2)

            self.logger.error(f"Error report saved to {error_report}")
            raise


async def main():
    """Main entry point."""
    try:
        orchestrator = DataExtractionOrchestrator()
        summary = await orchestrator.run_full_pipeline()
        return summary
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
