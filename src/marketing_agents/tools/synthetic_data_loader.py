"""
Synthetic Data Loader - Load and transform synthetic data into execution history format.

Converts CSV/JSON synthetic data files into the format expected by the analytics agent.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

from src.marketing_agents.utils import get_logger
from config.settings import get_settings


class SyntheticDataLoader:
    """Load synthetic data and transform it into execution history format."""

    def __init__(self):
        """Initialize data loader."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()
        self.data_dir = self.settings.data_dir / "raw"

    def load_campaign_data(self) -> List[Dict[str, Any]]:
        """
        Load campaign data from CSV and transform to execution history format.

        Returns:
            List of execution records with campaign metrics
        """
        campaign_file = self.data_dir / "marketing_data" / "campaigns_2024.csv"

        if not campaign_file.exists():
            self.logger.warning(f"Campaign data file not found: {campaign_file}")
            return []

        execution_records = []

        try:
            with open(campaign_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Transform campaign data to execution record format
                    record = self._transform_campaign_to_execution(row)
                    execution_records.append(record)

            self.logger.info(f"Loaded {len(execution_records)} campaign records")
            return execution_records

        except Exception as e:
            self.logger.error(f"Error loading campaign data: {e}")
            return []

    def load_agent_interaction_data(self) -> List[Dict[str, Any]]:
        """
        Load agent interaction data from JSON and transform to execution history format.

        Returns:
            List of execution records with agent interaction metrics
        """
        interactions_file = self.data_dir / "feedback" / "agent_interactions.json"

        if not interactions_file.exists():
            self.logger.warning(f"Interactions file not found: {interactions_file}")
            return []

        execution_records = []

        try:
            with open(interactions_file, "r") as f:
                interactions = json.load(f)

            for interaction in interactions:
                record = self._transform_interaction_to_execution(interaction)
                execution_records.append(record)

            self.logger.info(f"Loaded {len(execution_records)} interaction records")
            return execution_records

        except Exception as e:
            self.logger.error(f"Error loading interaction data: {e}")
            return []

    def load_all_execution_data(
        self, time_range: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all execution data from both campaigns and interactions.

        Args:
            time_range: Optional time range filter (e.g., '24h', '7d', '30d')
            limit: Optional maximum number of records to return

        Returns:
            Combined list of execution records
        """
        # Load both data sources
        campaign_records = self.load_campaign_data()
        interaction_records = self.load_agent_interaction_data()

        # Combine all records
        all_records = campaign_records + interaction_records

        # Sort by timestamp (most recent first)
        all_records.sort(key=lambda x: x.get("started_at", ""), reverse=True)

        # Apply time range filter if specified
        if time_range:
            cutoff_time = self._parse_time_range(time_range)
            filtered_records = [
                r for r in all_records if self._is_within_time_range(r, cutoff_time)
            ]

            # If no records found in time range, use all available data
            # This ensures analytics always has data to show
            if len(filtered_records) == 0 and len(all_records) > 0:
                self.logger.warning(
                    f"No data found in {time_range} range. Using all {len(all_records)} available records instead."
                )
                filtered_records = all_records

            all_records = filtered_records

        # Apply limit if specified
        if limit:
            all_records = all_records[:limit]

        self.logger.info(f"Loaded {len(all_records)} total execution records")
        return all_records

    def _transform_campaign_to_execution(
        self, campaign: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform campaign CSV row to execution record format.

        Args:
            campaign: Campaign data from CSV

        Returns:
            Execution record with metrics
        """
        # Parse dates (convert to timezone-aware UTC)
        from datetime import timezone

        start_date = datetime.strptime(campaign["start_date"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        end_date = datetime.strptime(campaign["end_date"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

        # Calculate duration
        duration = (end_date - start_date).total_seconds()

        # Determine status based on campaign status
        status_map = {
            "completed": "completed",
            "active": "completed",
            "scheduled": "pending",
            "paused": "failed",
        }
        status = status_map.get(campaign.get("status", "completed"), "completed")

        # Build execution record
        record = {
            "execution_id": f"exec_{campaign['campaign_id']}",
            "agent_id": "marketing_strategy",
            "workflow_id": f"wf_{campaign['campaign_id']}",
            "started_at": start_date.isoformat(),
            "completed_at": end_date.isoformat(),
            "status": status,
            "duration_seconds": duration,
            "result": {
                "success": status == "completed",
                "campaign_id": campaign["campaign_id"],
                "campaign_name": campaign["campaign_name"],
                "campaign_type": campaign["campaign_type"],
                "channel": campaign["channel"],
                "metrics": {
                    "impressions": int(float(campaign.get("impressions", 0))),
                    "clicks": int(float(campaign.get("clicks", 0))),
                    "conversions": int(float(campaign.get("conversions", 0))),
                    "cost": float(campaign.get("budget", 0)),
                    "revenue": float(campaign.get("revenue", 0)),
                    "ctr": float(campaign.get("ctr", 0)),
                    "conversion_rate": float(campaign.get("conversion_rate", 0)),
                    "roi": float(campaign.get("roi", 0)),
                    "likes": random.randint(10, 1000),  # Synthetic engagement
                    "shares": random.randint(5, 500),
                    "comments": random.randint(2, 200),
                },
            },
            "metadata": {
                "audience_segment": campaign.get("audience_segment"),
                "geography": campaign.get("geography"),
                "device_type": campaign.get("device_type"),
                "age_group": campaign.get("age_group"),
            },
        }

        return record

    def _transform_interaction_to_execution(
        self, interaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transform agent interaction to execution record format.

        Args:
            interaction: Interaction data from JSON

        Returns:
            Execution record with agent metrics
        """
        # Parse timestamp
        timestamp = datetime.fromisoformat(
            interaction["timestamp"].replace("Z", "+00:00")
        )

        # Calculate completion time
        response_time = interaction.get("response_time_seconds", 10)
        completed_at = timestamp + timedelta(seconds=response_time)

        # Determine status from user feedback
        feedback = interaction.get("user_feedback", {})
        helpful = feedback.get("helpful", True)
        status = "completed" if helpful else "failed"

        # Check for handoff
        handoff_occurred = interaction.get("handoff_occurred", False)
        target_agent = interaction.get("handoff_to")

        # Build execution record
        record = {
            "execution_id": f"exec_{interaction['interaction_id']}",
            "agent_id": interaction.get("agent", "orchestrator"),
            "workflow_id": f"wf_{interaction['interaction_id']}",
            "started_at": timestamp.isoformat(),
            "completed_at": completed_at.isoformat(),
            "status": status,
            "duration_seconds": response_time,
            "result": {
                "success": helpful,
                "interaction_id": interaction["interaction_id"],
                "query": interaction.get("query"),
                "resolution_quality": interaction.get("resolution_quality"),
                "handoff_required": handoff_occurred,
                "target_agent": target_agent,
                "metrics": {
                    "tokens_used": interaction.get("tokens_used", 0),
                    "rating": feedback.get("rating", 0),
                },
            },
            "metadata": {
                "comment": feedback.get("comment"),
            },
        }

        return record

    def _parse_time_range(self, time_range: str) -> datetime:
        """
        Parse time range string to cutoff datetime.

        Args:
            time_range: Time range (e.g., '1h', '24h', '7d', '30d')

        Returns:
            Cutoff datetime
        """
        from datetime import timezone

        unit = time_range[-1]
        value = int(time_range[:-1])

        if unit == "h":
            delta = timedelta(hours=value)
        elif unit == "d":
            delta = timedelta(days=value)
        elif unit == "m":
            delta = timedelta(minutes=value)
        else:
            delta = timedelta(days=1)  # Default to 1 day

        # CRITICAL: Use timezone-aware datetime for comparison
        return datetime.now(timezone.utc) - delta

    def _is_within_time_range(
        self, record: Dict[str, Any], cutoff_time: datetime
    ) -> bool:
        """
        Check if record is within time range.

        Args:
            record: Execution record
            cutoff_time: Cutoff datetime

        Returns:
            True if record is within range
        """
        try:
            from datetime import timezone

            started_at = record.get("started_at", "")
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))

            # Ensure both datetimes are timezone-aware for comparison
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            if cutoff_time.tzinfo is None:
                cutoff_time = cutoff_time.replace(tzinfo=timezone.utc)

            return started_at >= cutoff_time
        except (ValueError, TypeError):
            return False


# Singleton instance
_loader_instance = None


def get_synthetic_data_loader() -> SyntheticDataLoader:
    """Get or create singleton data loader instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = SyntheticDataLoader()
    return _loader_instance


def load_execution_data(
    time_range: Optional[str] = None, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to load execution data.

    Args:
        time_range: Optional time range filter (e.g., '24h', '7d', '30d')
        limit: Optional maximum number of records

    Returns:
        List of execution records
    """
    loader = get_synthetic_data_loader()
    return loader.load_all_execution_data(time_range, limit)
