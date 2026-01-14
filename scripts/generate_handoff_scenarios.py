#!/usr/bin/env python3
"""
Generate synthetic campaign data for handoff scenario testing.

Creates specific scenarios that match the Analytics Agent Test Guide:
1. Performance decline scenario → handoff to marketing strategy
2. Pattern discovery scenario → handoff to feedback learning
3. Customer satisfaction drop → handoff to customer support
4. Prediction accuracy scenario → handoff to feedback learning
5. Normal high-performing campaigns → no handoff
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import random


def generate_performance_decline_scenario():
    """Generate campaigns showing declining performance for Test Query #24."""
    campaigns = []
    base_date = datetime(2025, 10, 1)  # Q4 2025

    # Declining paid social campaigns (3 months of decline)
    for month in range(3):
        start_date = base_date + timedelta(days=month * 30)
        end_date = start_date + timedelta(days=29)

        # Performance gets worse each month
        roi = 320 - (month * 80)  # 320% → 240% → 160% (declining)
        ctr = 2.5 - (month * 0.5)  # 2.5% → 2.0% → 1.5%
        conversion_rate = 3.2 - (month * 0.6)  # 3.2% → 2.6% → 2.0%

        campaigns.append(
            {
                "campaign_id": f"paid_social_decline_{month+1}",
                "campaign_name": f"Paid Social Q4 - Month {month+1}",
                "campaign_type": "awareness",
                "channel": "paid_social",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "status": "completed",
                "budget": 25000,
                "impressions": 500000,
                "clicks": int(500000 * ctr / 100),
                "conversions": int(500000 * ctr / 100 * conversion_rate / 100),
                "revenue": int(500000 * ctr / 100 * conversion_rate / 100 * 150),
                "ctr": ctr,
                "conversion_rate": conversion_rate,
                "roi": roi,
                "audience_segment": "B2B SaaS",
                "geography": "North America",
                "device_type": "mixed",
                "age_group": "25-54",
            }
        )

    return campaigns


def generate_video_outperformance_pattern():
    """Generate campaigns showing video consistently outperforms static (Test Query #25)."""
    campaigns = []
    base_date = datetime(2025, 7, 1)

    # Create 6 campaigns over 3 months - alternating video and static
    for i in range(6):
        start_date = base_date + timedelta(days=i * 15)
        end_date = start_date + timedelta(days=14)

        is_video = i % 2 == 0
        content_type = "video" if is_video else "static_image"

        # Video performs 40% better consistently
        base_ctr = 3.0
        base_conv = 2.7
        base_roi = 280

        if is_video:
            ctr = base_ctr * 1.40  # 4.2%
            conversion_rate = base_conv * 1.41  # 3.8%
            roi = base_roi * 1.43  # 400%
        else:
            ctr = base_ctr
            conversion_rate = base_conv
            roi = base_roi

        impressions = 400000
        clicks = int(impressions * ctr / 100)
        conversions = int(clicks * conversion_rate / 100)

        campaigns.append(
            {
                "campaign_id": f"{content_type}_campaign_{i+1}",
                "campaign_name": f"Product Launch - {content_type.replace('_', ' ').title()}",
                "campaign_type": "conversion",
                "channel": "paid_social",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "status": "completed",
                "budget": 15000,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": conversions * 120,
                "ctr": round(ctr, 2),
                "conversion_rate": round(conversion_rate, 2),
                "roi": round(roi, 1),
                "audience_segment": "E-commerce",
                "geography": "Global",
                "device_type": "mobile",
                "age_group": "18-44",
                "content_type": content_type,
            }
        )

    return campaigns


def generate_customer_satisfaction_drop():
    """Generate data showing customer satisfaction drop (Test Query #27)."""
    # This will be interaction data showing declining satisfaction
    interactions = []
    base_date = datetime(2025, 11, 1)

    # Generate 60 interactions over 2 months
    for i in range(60):
        timestamp = base_date + timedelta(days=i)

        # Satisfaction drops in second month (after day 30)
        if i < 30:
            helpful = random.choice([True] * 9 + [False])  # 90% helpful
            rating = random.randint(4, 5)
        else:
            helpful = random.choice([True] * 6 + [False] * 4)  # 60% helpful (drop)
            rating = random.randint(2, 4)

        interactions.append(
            {
                "interaction_id": f"int_satisfaction_{i+1:03d}",
                "timestamp": timestamp.isoformat() + "Z",
                "agent": "customer_support",
                "query": "Help with checkout process" if i >= 30 else "General inquiry",
                "resolution_quality": "resolved" if helpful else "unresolved",
                "response_time_seconds": random.randint(15, 120),
                "tokens_used": random.randint(500, 1500),
                "handoff_occurred": False,
                "handoff_to": None,
                "user_feedback": {
                    "helpful": helpful,
                    "rating": rating,
                    "comment": (
                        "Confused by new checkout flow"
                        if i >= 30 and not helpful
                        else "Good service"
                    ),
                },
            }
        )

    return interactions


def generate_high_performing_campaigns():
    """Generate normal high-performing campaigns (no handoff needed)."""
    campaigns = []
    base_date = datetime(2025, 11, 1)

    # 5 well-performing campaigns
    channels = ["email", "content", "organic_social", "paid_search", "display"]

    for i, channel in enumerate(channels):
        start_date = base_date + timedelta(days=i * 7)
        end_date = start_date + timedelta(days=6)

        # All performing well
        ctr = random.uniform(2.5, 4.0)
        conversion_rate = random.uniform(2.8, 4.2)
        roi = random.uniform(250, 400)

        impressions = random.randint(300000, 600000)
        clicks = int(impressions * ctr / 100)
        conversions = int(clicks * conversion_rate / 100)

        campaigns.append(
            {
                "campaign_id": f"high_perf_{channel}_{i+1}",
                "campaign_name": f"Q4 {channel.replace('_', ' ').title()} Campaign",
                "campaign_type": "conversion",
                "channel": channel,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "status": "completed",
                "budget": random.randint(10000, 30000),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "revenue": conversions * random.randint(100, 180),
                "ctr": round(ctr, 2),
                "conversion_rate": round(conversion_rate, 2),
                "roi": round(roi, 1),
                "audience_segment": random.choice(
                    ["B2B SaaS", "E-commerce", "Enterprise"]
                ),
                "geography": random.choice(["North America", "Europe", "Global"]),
                "device_type": random.choice(["desktop", "mobile", "mixed"]),
                "age_group": "25-54",
            }
        )

    return campaigns


def generate_prediction_accuracy_scenario():
    """Generate campaigns for prediction/forecasting scenarios."""
    campaigns = []
    base_date = datetime(2025, 8, 1)

    # Historical campaigns with actual vs predicted discrepancies
    for i in range(4):
        start_date = base_date + timedelta(days=i * 30)
        end_date = start_date + timedelta(days=29)

        # Actual performance varies from predictions
        predicted_conversions = 800
        actual_conversions = random.randint(600, 1000)
        accuracy_variance = (
            abs(actual_conversions - predicted_conversions) / predicted_conversions
        )

        campaigns.append(
            {
                "campaign_id": f"forecast_test_{i+1}",
                "campaign_name": f"Forecast Validation Campaign {i+1}",
                "campaign_type": "conversion",
                "channel": "paid_search",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "status": "completed",
                "budget": 20000,
                "impressions": 500000,
                "clicks": 15000,
                "conversions": actual_conversions,
                "revenue": actual_conversions * 140,
                "ctr": 3.0,
                "conversion_rate": round(actual_conversions / 15000 * 100, 2),
                "roi": round(actual_conversions * 140 / 20000 * 100, 1),
                "predicted_conversions": predicted_conversions,
                "prediction_accuracy": round((1 - accuracy_variance) * 100, 1),
                "audience_segment": "B2B SaaS",
                "geography": "North America",
                "device_type": "desktop",
                "age_group": "35-54",
            }
        )

    return campaigns


def main():
    """Generate all scenario data and save to files."""
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "marketing_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    feedback_dir = Path(__file__).parent.parent / "data" / "raw" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Combine all campaign scenarios
    all_campaigns = []
    all_campaigns.extend(generate_performance_decline_scenario())
    all_campaigns.extend(generate_video_outperformance_pattern())
    all_campaigns.extend(generate_high_performing_campaigns())
    all_campaigns.extend(generate_prediction_accuracy_scenario())

    # Sort by date
    all_campaigns.sort(key=lambda x: x["start_date"])

    # Collect all possible fieldnames
    all_fieldnames = set()
    for campaign in all_campaigns:
        all_fieldnames.update(campaign.keys())

    # Sort fieldnames for consistency
    fieldnames = sorted(all_fieldnames)

    # Write campaigns to CSV
    campaigns_file = output_dir / "campaigns_2024.csv"

    if all_campaigns:
        with open(campaigns_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_campaigns)

        print(f"✓ Generated {len(all_campaigns)} campaign scenarios")
        print(f"  Saved to: {campaigns_file}")

    # Generate customer satisfaction interactions
    interactions = generate_customer_satisfaction_drop()

    interactions_file = feedback_dir / "agent_interactions.json"
    with open(interactions_file, "w") as f:
        json.dump(interactions, f, indent=2)

    print(f"✓ Generated {len(interactions)} interaction scenarios")
    print(f"  Saved to: {interactions_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Generated Handoff Test Scenarios:")
    print("=" * 70)
    print("\n1. Performance Decline (3 campaigns)")
    print("   → Should trigger handoff to marketing_strategy")
    print("   → Query: 'Our paid social campaigns have declining ROI'")

    print("\n2. Video Outperformance Pattern (6 campaigns)")
    print("   → Should trigger handoff to feedback_learning")
    print("   → Query: 'Video content consistently outperforms static images'")

    print("\n3. Customer Satisfaction Drop (60 interactions)")
    print("   → Should trigger handoff to customer_support")
    print("   → Query: 'Customer satisfaction scores dropped 15%'")

    print("\n4. Prediction Accuracy (4 campaigns)")
    print("   → Should trigger handoff to feedback_learning")
    print("   → Query: 'How can we improve conversion predictions?'")

    print("\n5. High Performing Campaigns (5 campaigns)")
    print("   → Should NOT trigger handoff")
    print("   → Query: 'Show me campaign metrics for last month'")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
