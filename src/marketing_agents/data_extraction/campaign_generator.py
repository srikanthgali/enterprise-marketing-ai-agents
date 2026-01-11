"""
Campaign data generator for creating realistic marketing performance data.

Generates comprehensive campaign metrics including impressions, clicks,
conversions, ROI, and other KPIs across multiple channels and time periods.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from src.marketing_agents.utils import get_logger
from config.settings import get_settings


class CampaignDataGenerator:
    """
    Generate realistic marketing campaign performance data.

    Creates campaigns across 6 types and 6 channels with realistic
    metrics, seasonality, and trends over 12 months.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize campaign generator."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        self.output_dir = output_dir or (
            self.settings.data_dir / "raw" / "marketing_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_campaign_templates()

    def _init_campaign_templates(self) -> None:
        """Initialize campaign templates and configurations."""

        self.campaign_types = {
            "product_launch": {
                "count": 20,
                "budget_range": (20000, 100000),
                "ctr_range": (2.5, 4.5),
                "conv_rate_range": (3, 8),
                "roi_range": (150, 350),
            },
            "feature_announcement": {
                "count": 30,
                "budget_range": (5000, 30000),
                "ctr_range": (2.0, 4.0),
                "conv_rate_range": (2, 6),
                "roi_range": (100, 250),
            },
            "webinar_promotion": {
                "count": 25,
                "budget_range": (3000, 15000),
                "ctr_range": (3.0, 5.0),
                "conv_rate_range": (4, 10),
                "roi_range": (120, 280),
            },
            "case_study": {
                "count": 20,
                "budget_range": (2000, 10000),
                "ctr_range": (1.8, 3.5),
                "conv_rate_range": (2, 5),
                "roi_range": (80, 180),
            },
            "retargeting": {
                "count": 25,
                "budget_range": (8000, 40000),
                "ctr_range": (3.5, 6.0),
                "conv_rate_range": (5, 12),
                "roi_range": (200, 400),
            },
            "brand_awareness": {
                "count": 30,
                "budget_range": (10000, 50000),
                "ctr_range": (1.5, 3.0),
                "conv_rate_range": (1, 3),
                "roi_range": (-20, 100),
            },
        }

        self.channels = {
            "google_ads": {
                "avg_cpc": 2.5,
                "avg_ctr": 3.2,
                "avg_conv_rate": 4.5,
            },
            "linkedin_ads": {
                "avg_cpc": 5.5,
                "avg_ctr": 2.8,
                "avg_conv_rate": 6.2,
            },
            "facebook_ads": {
                "avg_cpc": 1.8,
                "avg_ctr": 3.5,
                "avg_conv_rate": 3.8,
            },
            "email_marketing": {
                "avg_cpc": 0.1,
                "avg_ctr": 4.2,
                "avg_conv_rate": 5.5,
            },
            "content_marketing": {
                "avg_cpc": 0.5,
                "avg_ctr": 2.5,
                "avg_conv_rate": 3.2,
            },
            "partner_comarketing": {
                "avg_cpc": 1.2,
                "avg_ctr": 3.8,
                "avg_conv_rate": 5.8,
            },
        }

        self.audience_segments = [
            "enterprise_payment_managers",
            "fintech_founders",
            "ecommerce_businesses",
            "saas_billing_teams",
            "marketplace_platforms",
            "subscription_businesses",
            "developers_technical",
            "cfo_finance_teams",
        ]

        self.geographies = [
            "north_america",
            "europe",
            "asia_pacific",
            "latin_america",
            "global",
        ]

        self.device_types = ["desktop", "mobile", "tablet"]

        self.age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]

    def generate_campaigns(
        self, start_date: Optional[datetime] = None, num_months: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Generate campaign performance data.

        Args:
            start_date: Start date for campaigns
            num_months: Number of months to generate

        Returns:
            List of campaign dictionaries
        """
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=365)

        self.logger.info(f"Generating campaigns for {num_months} months...")

        campaigns = []
        campaign_counter = 1

        for campaign_type, config in self.campaign_types.items():
            for i in range(config["count"]):
                campaign = self._generate_campaign(
                    campaign_type, config, start_date, num_months, campaign_counter
                )
                campaigns.append(campaign)
                campaign_counter += 1

        self.logger.info(f"Generated {len(campaigns)} campaigns")
        return campaigns

    def _generate_campaign(
        self,
        campaign_type: str,
        config: Dict[str, Any],
        start_date: datetime,
        num_months: int,
        campaign_id: int,
    ) -> Dict[str, Any]:
        """Generate a single campaign."""

        # Random start date within period
        days_offset = random.randint(0, num_months * 30)
        campaign_start = start_date + timedelta(days=days_offset)

        # Campaign duration (7-60 days)
        duration_days = random.randint(7, 60)
        campaign_end = campaign_start + timedelta(days=duration_days)

        # Select channel
        channel = random.choice(list(self.channels.keys()))
        channel_config = self.channels[channel]

        # Budget
        budget = random.uniform(*config["budget_range"])

        # Calculate impressions based on channel CPC and budget
        avg_cpc = channel_config["avg_cpc"]
        clicks = int(budget / avg_cpc * random.uniform(0.8, 1.2))

        # CTR (influenced by channel and campaign type)
        base_ctr = random.uniform(*config["ctr_range"])
        ctr = base_ctr * random.uniform(0.9, 1.1) / 100  # Convert to decimal

        # Calculate impressions from clicks and CTR
        impressions = int(clicks / ctr) if ctr > 0 else 0

        # Conversion rate
        base_conv_rate = random.uniform(*config["conv_rate_range"])
        conversion_rate = base_conv_rate * random.uniform(0.85, 1.15) / 100

        # Conversions
        conversions = int(clicks * conversion_rate)

        # Revenue (assuming average order value)
        avg_order_value = random.uniform(500, 2000)
        revenue = conversions * avg_order_value

        # ROI
        if budget > 0:
            roi = ((revenue - budget) / budget) * 100
        else:
            roi = 0

        # CAC (Customer Acquisition Cost)
        cac = budget / conversions if conversions > 0 else budget

        # LTV (Lifetime Value) - typically 3-5x CAC for good campaigns
        ltv_multiplier = random.uniform(2.5, 5.5)
        ltv = cac * ltv_multiplier

        # Status
        if campaign_end < datetime.utcnow():
            status = "completed"
        elif campaign_start > datetime.utcnow():
            status = "scheduled"
        else:
            status = "active"

        # Apply seasonality adjustments
        campaign = {
            "campaign_id": f"CMP-2024-{campaign_id:04d}",
            "campaign_name": self._generate_campaign_name(campaign_type, channel),
            "campaign_type": campaign_type,
            "start_date": campaign_start.strftime("%Y-%m-%d"),
            "end_date": campaign_end.strftime("%Y-%m-%d"),
            "channel": channel,
            "audience_segment": random.choice(self.audience_segments),
            "budget": round(budget, 2),
            "impressions": impressions,
            "clicks": clicks,
            "ctr": round(ctr * 100, 2),  # As percentage
            "conversions": conversions,
            "conversion_rate": round(conversion_rate * 100, 2),  # As percentage
            "revenue": round(revenue, 2),
            "roi": round(roi, 2),
            "cac": round(cac, 2),
            "ltv": round(ltv, 2),
            "geography": random.choice(self.geographies),
            "device_type": random.choice(self.device_types),
            "age_group": random.choice(self.age_groups),
            "status": status,
        }

        return campaign

    def _generate_campaign_name(self, campaign_type: str, channel: str) -> str:
        """Generate a descriptive campaign name for Stripe products."""
        templates = {
            "product_launch": [
                "Payment Links 2.0 Launch",
                "Checkout with Custom Branding",
                "Financial Connections Launch",
                "Terminal SDK Release",
                "Stripe Tax Automation Launch",
            ],
            "feature_announcement": [
                "3D Secure 2.0 Update",
                "New Payment Methods - Buy Now Pay Later",
                "Subscription Billing Enhancements",
                "Connect Express Dashboard Update",
                "Radar Rules Engine v2",
            ],
            "webinar_promotion": [
                "Payment Optimization Webinar",
                "Subscription Best Practices",
                "Fraud Prevention Workshop",
                "International Expansion Guide",
                "PCI Compliance Training",
            ],
            "case_study": [
                "Enterprise Payment Success Story",
                "Subscription Growth Case Study",
                "Fraud Reduction Achievement",
                "Global Expansion Success",
                "Revenue Recovery Case Study",
            ],
            "retargeting": [
                "Free Trial Users Retargeting",
                "Checkout Abandonment Recovery",
                "Subscription Upgrade Campaign",
                "Feature Adoption - Payment Links",
                "Reactivation Campaign",
            ],
            "brand_awareness": [
                "Payment Innovation Leadership",
                "State of Payments Report",
                "Developer Experience Campaign",
                "Platform Economy Thought Leadership",
                "Financial Services Innovation",
            ],
        }

        base_name = random.choice(templates.get(campaign_type, ["Campaign"]))
        channel_suffix = channel.replace("_", " ").title()

        return f"{base_name} - {channel_suffix}"

    def save_campaigns(
        self, campaigns: List[Dict[str, Any]], format: str = "csv"
    ) -> None:
        """
        Save campaigns to file.

        Args:
            campaigns: List of campaign dictionaries
            format: Output format (csv or json)
        """
        if format == "csv":
            filepath = self.output_dir / "campaigns_2024.csv"

            if campaigns:
                fieldnames = campaigns[0].keys()

                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(campaigns)

                self.logger.info(f"Saved {len(campaigns)} campaigns to {filepath}")

        elif format == "json":
            filepath = self.output_dir / "campaigns_2024.json"

            with open(filepath, "w") as f:
                json.dump(campaigns, f, indent=2)

            self.logger.info(f"Saved {len(campaigns)} campaigns to {filepath}")

        # Save statistics
        self._save_campaign_stats(campaigns)

    def _save_campaign_stats(self, campaigns: List[Dict[str, Any]]) -> None:
        """Save campaign statistics."""
        stats = {
            "total_campaigns": len(campaigns),
            "by_type": {},
            "by_channel": {},
            "by_status": {},
            "total_budget": 0,
            "total_revenue": 0,
            "avg_roi": 0,
            "avg_ctr": 0,
            "avg_conversion_rate": 0,
        }

        budgets = []
        revenues = []
        rois = []
        ctrs = []
        conv_rates = []

        for campaign in campaigns:
            # By type
            ctype = campaign["campaign_type"]
            stats["by_type"][ctype] = stats["by_type"].get(ctype, 0) + 1

            # By channel
            channel = campaign["channel"]
            stats["by_channel"][channel] = stats["by_channel"].get(channel, 0) + 1

            # By status
            status = campaign["status"]
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Metrics
            budgets.append(campaign["budget"])
            revenues.append(campaign["revenue"])
            rois.append(campaign["roi"])
            ctrs.append(campaign["ctr"])
            conv_rates.append(campaign["conversion_rate"])

        stats["total_budget"] = round(sum(budgets), 2)
        stats["total_revenue"] = round(sum(revenues), 2)
        stats["avg_roi"] = round(sum(rois) / len(rois), 2) if rois else 0
        stats["avg_ctr"] = round(sum(ctrs) / len(ctrs), 2) if ctrs else 0
        stats["avg_conversion_rate"] = (
            round(sum(conv_rates) / len(conv_rates), 2) if conv_rates else 0
        )

        # Save stats
        stats_file = self.output_dir / "_campaign_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Campaign statistics saved to {stats_file}")

    def _calculate_stats(self, campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from generated campaigns."""
        stats = {
            "total_campaigns": len(campaigns),
            "by_type": {},
            "by_channel": {},
            "by_status": {},
            "total_budget": 0,
            "total_revenue": 0,
            "avg_roi": 0,
            "avg_ctr": 0,
            "avg_conversion_rate": 0,
            "date_range": {"start": None, "end": None},
        }

        budgets = []
        revenues = []
        rois = []
        ctrs = []
        conv_rates = []
        dates = []

        for campaign in campaigns:
            # By type
            ctype = campaign["campaign_type"]
            stats["by_type"][ctype] = stats["by_type"].get(ctype, 0) + 1

            # By channel
            channel = campaign["channel"]
            stats["by_channel"][channel] = stats["by_channel"].get(channel, 0) + 1

            # By status
            status = campaign["status"]
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Metrics
            budgets.append(campaign["budget"])
            revenues.append(campaign["revenue"])
            rois.append(campaign["roi"])
            ctrs.append(campaign["ctr"])
            conv_rates.append(campaign["conversion_rate"])
            dates.append(campaign["start_date"])

        stats["total_budget"] = round(sum(budgets), 2) if budgets else 0
        stats["total_revenue"] = round(sum(revenues), 2) if revenues else 0
        stats["avg_roi"] = round(sum(rois) / len(rois), 2) if rois else 0
        stats["avg_ctr"] = round(sum(ctrs) / len(ctrs), 2) if ctrs else 0
        stats["avg_conversion_rate"] = (
            round(sum(conv_rates) / len(conv_rates), 2) if conv_rates else 0
        )

        if dates:
            stats["date_range"]["start"] = min(dates)
            stats["date_range"]["end"] = max(dates)

        return stats

    def generate_audience_segments(self) -> None:
        """Generate audience segment definitions."""
        segments = []

        for segment in self.audience_segments:
            seg_def = {
                "segment_id": f"SEG-{segment.upper().replace('_', '-')}",
                "segment_name": segment.replace("_", " ").title(),
                "description": f"Targeting {segment.replace('_', ' ')}",
                "size_estimate": random.randint(10000, 500000),
                "avg_ltv": random.uniform(1000, 5000),
                "avg_engagement_score": random.uniform(3.0, 5.0),
                "preferred_channels": random.sample(
                    list(self.channels.keys()), random.randint(2, 4)
                ),
            }
            segments.append(seg_def)

        filepath = self.output_dir / "audience_segments.json"
        with open(filepath, "w") as f:
            json.dump(segments, f, indent=2)

        self.logger.info(f"Saved {len(segments)} audience segments to {filepath}")

    def generate_channel_benchmarks(self) -> None:
        """Generate industry benchmark data for channels."""
        benchmarks = {}

        for channel, config in self.channels.items():
            benchmarks[channel] = {
                "channel_name": channel.replace("_", " ").title(),
                "industry_avg_ctr": config["avg_ctr"],
                "industry_avg_conversion_rate": config["avg_conv_rate"],
                "industry_avg_cpc": config["avg_cpc"],
                "recommended_budget_range": {
                    "min": 5000,
                    "max": 50000,
                },
                "best_practices": [
                    f"Optimize ad copy for {channel}",
                    f"A/B test landing pages",
                    f"Monitor {channel} metrics daily",
                ],
            }

        filepath = self.output_dir / "channel_benchmarks.json"
        with open(filepath, "w") as f:
            json.dump(benchmarks, f, indent=2)

        self.logger.info(f"Saved channel benchmarks to {filepath}")


def generate_campaign_data(num_campaigns: int = 150) -> Dict[str, Any]:
    """
    Generate synthetic marketing campaign data.

    Args:
        num_campaigns: Number of campaigns to generate (not used, generator uses campaign_types config)

    Returns:
        Generation results with statistics as dict (not list)
    """
    generator = CampaignDataGenerator()

    # Pass datetime instead of int - generate_campaigns uses campaign_types config for count
    campaigns = generator.generate_campaigns()

    # Save campaigns
    generator.save_campaigns(campaigns)

    # Calculate and return statistics as dict
    stats = generator._calculate_stats(campaigns)

    return stats


if __name__ == "__main__":
    generate_campaign_data()
