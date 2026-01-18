#!/usr/bin/env python3
"""
Test script to verify the updated routing logic in Gradio UI.
Tests all the use cases mentioned by the user.
"""


def test_routing():
    """Simulate the routing logic from gradio_app.py"""

    test_cases = [
        # Customer Support Agent cases (may handoff to analytics)
        (
            "Customer satisfaction scores dropped 15% this month. What's causing this?",
            "customer_support",
        ),
        # Analytics Agent cases
        ("Generate a monthly performance report for December.", "analytics"),
        ("Show me the conversion funnel for our checkout process.", "analytics"),
        # Marketing Strategy queries → Route to Analytics (will handoff to Strategy)
        (
            "What positioning strategy should we use to stand out in the payment processing market?",
            "analytics",  # Changed: now routes to analytics first
        ),
        (
            "Our conversion rates are falling. Need help optimizing our approach.",
            "analytics",  # Changed: now routes to analytics first
        ),
        # Direct campaign creation → Marketing Strategy
        (
            "I need to create a marketing campaign to promote our new payment processing feature for small businesses.",
            "campaign_launch",
        ),
        # Improvement recommendations → Route to Analytics (will handoff to Feedback Learning)
        ("Recommend improvements for our campaign performance", "analytics"),  # Changed
        (
            "Are our conversion rate predictions accurate? How can we improve them?",
            "analytics",  # Changed
        ),
        # Direct Feedback Learning (no handoff needed)
        (
            "Rate the quality of the campaign strategy provided by the Marketing Agent: 2/5 stars. It was too generic.",
            "feedback_learning",
        ),
        (
            "Multiple agents are reporting that mobile checkout has issues. Can you investigate and recommend improvements?",
            "feedback_learning",
        ),
    ]

    print("Testing Updated Routing Logic with Handoff Support")
    print("=" * 80)

    passed = 0
    failed = 0

    for query, expected_workflow in test_cases:
        # Simulate routing logic
        message_lower = query.lower()

        # PRIORITY 1: Customer satisfaction/support issues
        if any(
            keyword in message_lower
            for keyword in [
                "customer satisfaction",
                "satisfaction score",
                "satisfaction dropped",
                "satisfaction fell",
                "satisfaction decreased",
                "nps score",
                "csat",
                "customer complaints",
                "what's causing this",
                "what is causing this",
            ]
        ):
            workflow_type = "customer_support"

        # PRIORITY 2: Analytics/Reporting
        elif any(
            keyword in message_lower
            for keyword in [
                "monthly report",
                "monthly performance",
                "quarterly report",
                "weekly report",
                "generate report",
                "generate a report",
                "show me the",
                "show me metrics",
                "conversion funnel",
                "checkout funnel",
                "funnel analysis",
                "analyze performance",
                "performance report",
                "create dashboard",
                "view statistics",
                "display data",
                "calculate roi",
                "measure conversion",
                "track funnel",
            ]
        ):
            workflow_type = "analytics"

        # PRIORITY 3: Marketing Strategy vs Analytics (for handoff scenarios)
        # Strategy questions WITH performance decline → Route to Analytics first (will handoff to Strategy)
        elif (
            any(
                keyword in message_lower
                for keyword in [
                    "positioning strategy",
                    "market positioning",
                    "stand out in",
                    "differentiate in",
                ]
            )
            or (
                any(
                    keyword in message_lower
                    for keyword in ["conversion rates", "conversion rate", "rates are"]
                )
                and any(
                    keyword in message_lower
                    for keyword in ["falling", "dropping", "declined", "decreased"]
                )
            )
            or (
                any(keyword in message_lower for keyword in ["need help", "help us"])
                and any(
                    keyword in message_lower for keyword in ["optimizing", "optimize"]
                )
            )
        ):
            # Route to Analytics - it will analyze and handoff to Marketing Strategy
            workflow_type = "analytics"

        # Pure campaign creation (no handoff needed) → Direct to Marketing Strategy
        elif any(
            keyword in message_lower
            for keyword in [
                "create a marketing campaign",
                "create marketing campaign",
                "launch a campaign",
                "launch campaign",
                "new campaign",
                "plan a campaign",
                "content calendar",
                "email sequence",
                "market research",
                "competitive analysis",
                "go-to-market",
                "gtm strategy",
            ]
        ):
            workflow_type = "campaign_launch"

        # PRIORITY 4: Feedback Learning - Route strategically for handoffs
        # Campaign improvement recommendations → Analytics first (will analyze and handoff to Feedback Learning)
        elif any(
            keyword in message_lower
            for keyword in [
                "recommend improvement",
                "recommend changes",
            ]
        ) and any(
            keyword in message_lower
            for keyword in ["campaign", "performance", "marketing"]
        ):
            # Route to Analytics - it will analyze performance and handoff to Feedback Learning
            workflow_type = "analytics"

        # Prediction accuracy questions → Analytics first (will handoff to Feedback Learning)
        elif any(
            keyword in message_lower
            for keyword in [
                "predictions accurate",
                "prediction accurate",
                "improve predictions",
                "improve accuracy",
                "improve forecast",
            ]
        ):
            # Route to Analytics - it will assess and handoff to Feedback Learning
            workflow_type = "analytics"

        # Direct feedback/ratings or system issues → Feedback Learning (no handoff needed)
        elif any(
            keyword in message_lower
            for keyword in [
                "rate the quality",
                "rating",
                "/5 stars",
                "/5",
                "too generic",
                "not specific enough",
                "multiple agents",
                "several agents",
                "agents are reporting",
                "agents report",
                "investigate and recommend",
                "recurring issue",
                "recurring problem",
                "keeps happening",
                "optimize system",
                "system improvement",
                "learn from",
                "insights from",
                "takeaways from",
                "lessons from",
                "how well is",
                "agent performing",
                "agent performance",
                "evaluate agent",
            ]
        ):
            workflow_type = "feedback_learning"

        else:
            workflow_type = "customer_support"

        # Check result
        status = "✅ PASS" if workflow_type == expected_workflow else "❌ FAIL"
        if workflow_type == expected_workflow:
            passed += 1
        else:
            failed += 1

        print(f"\n{status}")
        print(f"Query: {query[:80]}...")
        print(f"Expected: {expected_workflow}")
        print(f"Got: {workflow_type}")

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = test_routing()
    exit(0 if success else 1)
