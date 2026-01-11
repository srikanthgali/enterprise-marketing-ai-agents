"""
Feedback data generator for creating agent interaction logs and learning data.

Generates synthetic feedback from agent interactions, campaign reviews,
and customer satisfaction scores for training the Feedback & Learning Agent.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid

from src.marketing_agents.utils import get_logger
from config.settings import get_settings


class FeedbackDataGenerator:
    """
    Generate realistic feedback and learning data.

    Creates:
    1. Agent interaction logs (1000 interactions)
    2. Campaign feedback reviews (200 reviews)
    3. Customer satisfaction scores (500 CSAT scores)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize feedback generator."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        self.output_dir = output_dir or (self.settings.data_dir / "raw" / "feedback")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_templates()

    def _init_templates(self) -> None:
        """Initialize feedback templates."""

        self.agents = [
            "orchestrator",
            "marketing_strategy",
            "customer_support",
            "analytics_evaluation",
            "feedback_learning",
        ]

        # Query templates by agent
        self.query_templates = {
            "marketing_strategy": [
                "Help me plan a payment product launch campaign",
                "What channels work best for fintech marketing?",
                "How should I allocate budget for developer-focused campaigns?",
                "Create content calendar for Stripe Tax launch",
                "Analyze competitor payment processor marketing",
                "Suggest audience segments for subscription billing campaign",
                "Best timing for payment optimization webinar?",
                "A/B test design for checkout flow improvements",
            ],
            "customer_support": [
                "How do I implement Payment Intents API?",
                "Payment failing with card_declined error",
                "How to set up subscription billing with trials?",
                "Explain webhook signature validation",
                "How do I handle 3D Secure authentication?",
                "API rate limit exceeded - what to do?",
                "Help with Stripe.js integration",
                "How to troubleshoot refund delays?",
            ],
            "analytics_evaluation": [
                "Generate payment acceptance rate report",
                "Compare revenue across payment methods",
                "Show me subscription retention funnel",
                "What are top performing checkout flows?",
                "Analyze fraud prevention effectiveness",
                "Calculate subscription MRR and churn",
                "Show payment method attribution",
                "Forecast next quarter revenue",
            ],
            "feedback_learning": [
                "How can we improve payment success rates?",
                "What patterns exist in failed payments?",
                "Suggest optimizations based on checkout data",
                "What did we learn from subscription churn?",
                "How to reduce card decline rates?",
                "Identify best practices from high-converting flows",
                "What payment methods should we stop supporting?",
                "Recommend fraud detection improvements",
            ],
        }

        # Response quality levels
        self.quality_levels = ["excellent", "good", "satisfactory", "needs_improvement"]

        # Feedback comments
        self.positive_comments = [
            "Very helpful and actionable advice",
            "Clear explanation with good examples",
            "Exactly what I needed",
            "Comprehensive and well-structured",
            "Saved me a lot of time",
            "Great insights backed by data",
        ]

        self.negative_comments = [
            "Too generic, needed more specific guidance",
            "Missing key information",
            "Response was unclear",
            "Didn't fully answer my question",
            "Could be more detailed",
            "Expected more actionable steps",
        ]

        # Lessons learned templates
        self.lessons_learned = [
            "Payment Links converted 45% better than custom checkout",
            "Stripe Checkout reduced cart abandonment by 35%",
            "Adding Buy Now Pay Later increased AOV by 28%",
            "Subscription trials with card capture improved conversion by 40%",
            "3D Secure 2.0 reduced friction without increasing fraud",
            "Express payouts improved seller retention by 25%",
            "Tax automation saved 20 hours/week in manual work",
            "Radar rules reduced false declines by 30%",
            "International payment methods increased global revenue 50%",
            "Subscription dunning recovered 15% failed payments",
        ]

        # Recommendation templates
        self.recommendations = [
            "Enable Payment Links for faster checkout",
            "Implement subscription billing with auto-retry",
            "Add Buy Now Pay Later options (Klarna, Affirm)",
            "Enable Radar fraud detection rules",
            "Expand to international payment methods",
            "Implement Stripe Tax for compliance",
            "Use Financial Connections for bank verification",
            "Optimize checkout flow with A/B testing",
            "Enable customer portal for self-service",
            "Implement smart retry logic for subscriptions",
        ]

    def generate_agent_interactions(
        self, num_interactions: int = 1000, date_range_days: int = 180
    ) -> List[Dict[str, Any]]:
        """
        Generate agent interaction logs.

        Args:
            num_interactions: Number of interactions to generate
            date_range_days: Date range for interactions

        Returns:
            List of interaction dictionaries
        """
        self.logger.info(f"Generating {num_interactions} agent interactions...")

        interactions = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=date_range_days)

        for i in range(num_interactions):
            interaction = self._generate_interaction(start_date, end_date, i + 1)
            interactions.append(interaction)

        self.logger.info(f"Generated {len(interactions)} interactions")
        return interactions

    def _generate_interaction(
        self, start_date: datetime, end_date: datetime, interaction_num: int
    ) -> Dict[str, Any]:
        """Generate a single agent interaction."""

        # Random timestamp
        timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # Select agent (weighted towards support and strategy)
        agent_weights = [
            0.10,
            0.35,
            0.35,
            0.15,
            0.05,
        ]  # orchestrator gets fewer direct queries
        agent = random.choices(self.agents, weights=agent_weights)[0]

        # Get query template
        if agent in self.query_templates:
            query = random.choice(self.query_templates[agent])
        else:
            query = "Route this request to appropriate agent"

        # Generate response (placeholder - in production would be actual AI response)
        response = self._generate_response(agent, query)

        # Response time (in seconds)
        response_time = random.uniform(1.5, 30.0)

        # User feedback
        helpful = random.random() > 0.15  # 85% helpful
        rating = random.randint(4, 5) if helpful else random.randint(1, 3)

        if rating >= 4:
            comment = random.choice(self.positive_comments)
        else:
            comment = random.choice(self.negative_comments)

        # Resolution quality
        if rating == 5:
            quality = "excellent"
        elif rating == 4:
            quality = "good"
        elif rating == 3:
            quality = "satisfactory"
        else:
            quality = "needs_improvement"

        # Handoff occurred (10% of interactions)
        handoff_occurred = random.random() < 0.10
        handoff_to = None
        if handoff_occurred and agent != "orchestrator":
            # Select different agent for handoff
            other_agents = [a for a in self.agents if a != agent]
            handoff_to = random.choice(other_agents)

        return {
            "interaction_id": f"INT-{interaction_num:06d}",
            "timestamp": timestamp.isoformat(),
            "agent": agent,
            "query": query,
            "response": response,
            "response_time_seconds": round(response_time, 2),
            "user_feedback": {
                "helpful": helpful,
                "rating": rating,
                "comment": comment,
            },
            "resolution_quality": quality,
            "handoff_occurred": handoff_occurred,
            "handoff_to": handoff_to,
            "tokens_used": random.randint(100, 1500),
        }

    def _generate_response(self, agent: str, query: str) -> str:
        """Generate a placeholder response."""
        responses = {
            "marketing_strategy": "Based on your goals and target audience, I recommend...",
            "customer_support": "I can help you with that. Here's how to...",
            "analytics_evaluation": "Here's the analysis you requested. The data shows...",
            "feedback_learning": "Looking at historical data, I've identified these patterns...",
            "orchestrator": "I'll route your request to the appropriate specialist agent...",
        }
        return responses.get(agent, "Let me help you with that request...")

    def generate_campaign_feedback(
        self, num_reviews: int = 200, campaign_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate campaign feedback reviews.

        Args:
            num_reviews: Number of reviews to generate
            campaign_ids: Optional list of campaign IDs to reference

        Returns:
            List of campaign feedback dictionaries
        """
        self.logger.info(f"Generating {num_reviews} campaign feedback reviews...")

        if campaign_ids is None:
            campaign_ids = [f"CMP-2024-{i:04d}" for i in range(1, 151)]

        reviews = []

        for i in range(num_reviews):
            review = self._generate_campaign_review(random.choice(campaign_ids), i + 1)
            reviews.append(review)

        self.logger.info(f"Generated {len(reviews)} campaign reviews")
        return reviews

    def _generate_campaign_review(
        self, campaign_id: str, review_num: int
    ) -> Dict[str, Any]:
        """Generate a single campaign review."""

        # Performance vs goal
        performance_options = ["exceeded", "met", "below", "significantly_below"]
        performance_weights = [0.30, 0.40, 0.20, 0.10]
        performance = random.choices(performance_options, weights=performance_weights)[
            0
        ]

        # Select lessons learned (2-4 per review)
        num_lessons = random.randint(2, 4)
        lessons = random.sample(self.lessons_learned, num_lessons)

        # Select recommendations (2-3 per review)
        num_recommendations = random.randint(2, 3)
        recommendations = random.sample(self.recommendations, num_recommendations)

        # Key metrics
        metrics = {
            "actual_roi": round(random.uniform(-10, 350), 2),
            "target_roi": 150.0,
            "actual_conversion_rate": round(random.uniform(1.5, 8.5), 2),
            "target_conversion_rate": 5.0,
            "budget_utilization": round(random.uniform(75, 105), 2),
        }

        # Success factors
        success_factors = []
        if performance in ["exceeded", "met"]:
            factors = [
                "Strong creative execution",
                "Excellent audience targeting",
                "Optimal timing and frequency",
                "Compelling call-to-action",
                "High-quality landing page",
                "Effective A/B testing",
            ]
            success_factors = random.sample(factors, random.randint(2, 4))

        # Challenges
        challenges = []
        if performance in ["below", "significantly_below"]:
            issues = [
                "Higher than expected CPC",
                "Lower than expected CTR",
                "Landing page conversion issues",
                "Audience targeting too broad",
                "Creative fatigue",
                "Seasonal competition",
            ]
            challenges = random.sample(issues, random.randint(1, 3))

        return {
            "review_id": f"REV-{review_num:06d}",
            "campaign_id": campaign_id,
            "reviewed_at": datetime.utcnow().isoformat(),
            "reviewer": random.choice(
                ["marketing_team", "strategy_lead", "campaign_manager"]
            ),
            "performance_vs_goal": performance,
            "metrics": metrics,
            "lessons_learned": lessons,
            "success_factors": success_factors,
            "challenges": challenges,
            "recommended_adjustments": recommendations,
            "overall_score": random.randint(1, 10),
            "would_repeat": performance in ["exceeded", "met"],
        }

    def generate_customer_satisfaction(
        self, num_scores: int = 500, date_range_days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Generate customer satisfaction scores.

        Args:
            num_scores: Number of CSAT scores to generate
            date_range_days: Date range for scores

        Returns:
            List of CSAT dictionaries
        """
        self.logger.info(f"Generating {num_scores} CSAT scores...")

        scores = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=date_range_days)

        for i in range(num_scores):
            score = self._generate_csat_score(start_date, end_date, i + 1)
            scores.append(score)

        self.logger.info(f"Generated {len(scores)} CSAT scores")
        return scores

    def _generate_csat_score(
        self, start_date: datetime, end_date: datetime, score_num: int
    ) -> Dict[str, Any]:
        """Generate a single CSAT score."""

        # Timestamp
        timestamp = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # CSAT score (1-5, weighted towards positive)
        score_weights = [0.05, 0.10, 0.20, 0.35, 0.30]  # More 4s and 5s
        score = random.choices([1, 2, 3, 4, 5], weights=score_weights)[0]

        # Touchpoint
        touchpoints = [
            "support_interaction",
            "product_usage",
            "onboarding",
            "feature_adoption",
            "billing_interaction",
            "documentation",
        ]
        touchpoint = random.choice(touchpoints)

        # Customer segment
        segments = ["enterprise", "business", "growth", "starter"]
        segment = random.choice(segments)

        # NPS score (0-10)
        if score >= 4:
            nps = random.randint(7, 10)  # Promoter
        elif score == 3:
            nps = random.randint(5, 6)  # Passive
        else:
            nps = random.randint(0, 4)  # Detractor

        return {
            "csat_id": f"CSAT-{score_num:06d}",
            "timestamp": timestamp.isoformat(),
            "customer_id": f"CUST-{random.randint(1000, 9999)}",
            "touchpoint": touchpoint,
            "customer_segment": segment,
            "csat_score": score,
            "nps_score": nps,
            "account_age_months": random.randint(1, 48),
        }

    def save_all_feedback(
        self,
        interactions: Optional[List[Dict]] = None,
        campaign_reviews: Optional[List[Dict]] = None,
        csat_scores: Optional[List[Dict]] = None,
    ) -> None:
        """Save all feedback data to files."""

        # Save agent interactions
        if interactions:
            filepath = self.output_dir / "agent_interactions.json"
            with open(filepath, "w") as f:
                json.dump(interactions, f, indent=2)
            self.logger.info(f"Saved {len(interactions)} interactions to {filepath}")

        # Save campaign feedback
        if campaign_reviews:
            filepath = self.output_dir / "campaign_feedback.json"
            with open(filepath, "w") as f:
                json.dump(campaign_reviews, f, indent=2)
            self.logger.info(
                f"Saved {len(campaign_reviews)} campaign reviews to {filepath}"
            )

        # Save CSAT scores
        if csat_scores:
            import csv

            filepath = self.output_dir / "customer_satisfaction.csv"

            if csat_scores:
                fieldnames = csat_scores[0].keys()
                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(csat_scores)

                self.logger.info(f"Saved {len(csat_scores)} CSAT scores to {filepath}")

        # Save summary statistics
        self._save_feedback_stats(interactions, campaign_reviews, csat_scores)

    def _save_feedback_stats(
        self,
        interactions: Optional[List[Dict]],
        campaign_reviews: Optional[List[Dict]],
        csat_scores: Optional[List[Dict]],
    ) -> None:
        """Save feedback statistics."""
        stats = {
            "generated_at": datetime.utcnow().isoformat(),
            "interactions": {},
            "campaign_reviews": {},
            "csat": {},
        }

        if interactions:
            ratings = [i["user_feedback"]["rating"] for i in interactions]
            stats["interactions"] = {
                "total": len(interactions),
                "avg_rating": round(sum(ratings) / len(ratings), 2),
                "helpful_percentage": round(
                    len([i for i in interactions if i["user_feedback"]["helpful"]])
                    / len(interactions)
                    * 100,
                    2,
                ),
                "handoff_percentage": round(
                    len([i for i in interactions if i["handoff_occurred"]])
                    / len(interactions)
                    * 100,
                    2,
                ),
            }

        if campaign_reviews:
            stats["campaign_reviews"] = {
                "total": len(campaign_reviews),
                "exceeded_goal": len(
                    [
                        r
                        for r in campaign_reviews
                        if r["performance_vs_goal"] == "exceeded"
                    ]
                ),
                "met_goal": len(
                    [r for r in campaign_reviews if r["performance_vs_goal"] == "met"]
                ),
                "below_goal": len(
                    [
                        r
                        for r in campaign_reviews
                        if r["performance_vs_goal"] in ["below", "significantly_below"]
                    ]
                ),
            }

        if csat_scores:
            scores = [s["csat_score"] for s in csat_scores]
            stats["csat"] = {
                "total": len(csat_scores),
                "avg_score": round(sum(scores) / len(scores), 2),
                "score_distribution": {
                    str(i): len([s for s in csat_scores if s["csat_score"] == i])
                    for i in range(1, 6)
                },
            }

        stats_file = self.output_dir / "_feedback_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Feedback statistics saved to {stats_file}")


def generate_feedback_data():
    """Convenience function to generate all feedback data."""
    generator = FeedbackDataGenerator()

    # Generate all types of feedback
    interactions = generator.generate_agent_interactions(num_interactions=1000)
    campaign_reviews = generator.generate_campaign_feedback(num_reviews=200)
    csat_scores = generator.generate_customer_satisfaction(num_scores=500)

    # Save all data
    generator.save_all_feedback(interactions, campaign_reviews, csat_scores)

    return {
        "interactions": interactions,
        "campaign_reviews": campaign_reviews,
        "csat_scores": csat_scores,
    }


if __name__ == "__main__":
    generate_feedback_data()
