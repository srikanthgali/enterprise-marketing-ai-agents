"""
Support ticket generator for creating realistic synthetic customer support data.

Generates diverse, realistic support tickets across multiple categories
for training and testing the Customer Support Agent.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.marketing_agents.utils import get_logger
from config.settings import get_settings


class SupportTicketGenerator:
    """
    Generate realistic synthetic support tickets.

    Creates tickets across 5 categories:
    1. Integration & Technical
    2. Billing & Usage
    3. Privacy & Compliance
    4. Performance & Debugging
    5. Account & Access
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize ticket generator."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        self.output_dir = output_dir or (
            self.settings.data_dir / "raw" / "support_tickets"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ticket templates
        self._init_templates()

    def _init_templates(self) -> None:
        """Initialize ticket templates for each category."""

        self.categories = {
            "payment_processing": {
                "weight": 0.40,
                "templates": [
                    {
                        "subject": "Payment failed with {error_code}",
                        "description": "Customer attempted payment of ${amount} using {payment_method} but transaction failed with error: {error_code}. Customer ID: {customer_id}. Occurred {time_ago}. No pending authorization visible. Customer is requesting immediate resolution as this is blocking their {use_case}.",
                        "priority": ["high", "urgent"],
                        "resolution_time_range": (1, 4),
                    },
                    {
                        "subject": "Refund not appearing in customer account",
                        "description": "Issued refund of ${amount} for charge {charge_id} on {date} but customer reports funds not received. Refund status shows 'succeeded' in dashboard. Bank: {bank}. Expected arrival: {expected_date}. Customer following up {count} times.",
                        "priority": ["high", "medium"],
                        "resolution_time_range": (4, 24),
                    },
                    {
                        "subject": "3D Secure authentication failing for {percentage}% of transactions",
                        "description": "Customers in {region} experiencing 3DS authentication failures. Flow breaks at {step}. Browser: {browser}. Payment method: {payment_method}. Started {time_ago}. Affecting approximately {volume} transactions/day. Revenue impact: ${revenue_impact}.",
                        "priority": ["urgent", "high"],
                        "resolution_time_range": (2, 6),
                    },
                    {
                        "subject": "International payment currency conversion issue",
                        "description": "Customer in {country} charged ${amount} {currency} but statement shows ${converted_amount} {local_currency}. Exchange rate applied: {rate}. Expected rate: {expected_rate}. Discrepancy: {discrepancy}%. Customer disputing charge.",
                        "priority": ["medium", "high"],
                        "resolution_time_range": (3, 12),
                    },
                    {
                        "subject": "Subscription payment retry not working",
                        "description": "Subscription {subscription_id} for customer {customer_id} failed payment {time_ago}. Auto-retry configured for {retry_count} attempts but only {actual_attempts} executed. Customer should be on retry schedule but shows as 'past_due'. Risk of involuntary churn.",
                        "priority": ["high"],
                        "resolution_time_range": (2, 8),
                    },
                ],
            },
            "integration_technical": {
                "weight": 0.30,
                "templates": [
                    {
                        "subject": "Webhook signature verification failing",
                        "description": "Implementing webhook endpoint for {event_types} events. Signature validation failing {percentage}% of time using HMAC-SHA256. Secret key verified correct. Code: {code_snippet}. Stripe-Signature header present. Timestamps within tolerance window.",
                        "priority": ["medium", "high"],
                        "resolution_time_range": (2, 8),
                    },
                    {
                        "subject": "API rate limit exceeded on {endpoint}",
                        "description": "Hitting rate limits on {endpoint} endpoint. Current volume: {volume} requests/second. Plan: {plan}. Error code: rate_limit_exceeded. Started {time_ago}. Affecting {percentage}% of API calls. Need rate limit increase or optimization guidance.",
                        "priority": ["high"],
                        "resolution_time_range": (1, 4),
                    },
                    {
                        "subject": "Payment Intent not transitioning to succeeded state",
                        "description": "Created PaymentIntent {pi_id} with amount ${amount}. Payment method attached successfully. Customer completed payment {time_ago} but PI stuck in 'processing' state. No error in events. Customer charged but order not fulfilled.",
                        "priority": ["urgent", "high"],
                        "resolution_time_range": (1, 3),
                    },
                    {
                        "subject": "{sdk} SDK integration issues on {platform}",
                        "description": "Integrating Stripe {sdk} SDK version {version} on {platform}. Error: {error_message}. Environment: {environment}. Following docs at {doc_link}. Already tried: {attempted_solutions}. Need implementation guidance.",
                        "priority": ["medium"],
                        "resolution_time_range": (2, 6),
                    },
                    {
                        "subject": "Checkout Session timing out before completion",
                        "description": "Checkout Sessions created with {timeout} minute expiry. {percentage}% of customers reporting session expires before payment completion. Payment method collection takes {duration}. Need to extend timeout or optimize flow.",
                        "priority": ["high", "medium"],
                        "resolution_time_range": (3, 12),
                    },
                ],
            },
            "billing_subscriptions": {
                "weight": 0.15,
                "templates": [
                    {
                        "subject": "Unexpected charges on subscription invoice",
                        "description": "Invoice {invoice_id} for subscription {subscription_id} shows ${amount} but expected ${expected_amount}. Line items: {line_items}. Proration charges not explained. Customer on {plan} plan. Upgraded from {old_plan} on {upgrade_date}.",
                        "priority": ["high", "medium"],
                        "resolution_time_range": (4, 24),
                    },
                    {
                        "subject": "How to change subscription billing cycle from {current} to {target}?",
                        "description": "Customer wants to switch billing from {current} to {target}. Current subscription: {subscription_id}. MRR: ${mrr}. Concerns: 1) Proration handling, 2) Invoice timing, 3) Payment date alignment. Need guidance on best approach.",
                        "priority": ["medium", "low"],
                        "resolution_time_range": (2, 8),
                    },
                    {
                        "subject": "Subscription upgrade proration calculation incorrect",
                        "description": "Customer upgraded from ${old_price} to ${new_price} plan mid-cycle on {date}. Expected proration: ${expected}. Actual charge: ${actual}. Difference: ${difference}. Billing period: {period}. Need clarification on proration logic.",
                        "priority": ["medium"],
                        "resolution_time_range": (3, 12),
                    },
                    {
                        "subject": "Invoice payment failed - customer payment method valid",
                        "description": "Invoice {invoice_id} for ${amount} failed payment. Customer payment method {payment_method_id} successfully charged ${test_amount} in test mode. Card valid, not expired. Failure code: {failure_code}. Customer frustrated after {count} attempts.",
                        "priority": ["high"],
                        "resolution_time_range": (2, 8),
                    },
                    {
                        "subject": "Tax calculation incorrect for {region}",
                        "description": "Automatic tax calculation for {region} showing ${tax_amount} ({tax_rate}% rate). Expected {expected_rate}% based on {tax_type}. Customer address: {address}. Product type: {product_type}. Need tax configuration review.",
                        "priority": ["medium"],
                        "resolution_time_range": (4, 24),
                    },
                ],
            },
            "compliance_security": {
                "weight": 0.10,
                "templates": [
                    {
                        "subject": "PCI compliance attestation of compliance (AOC) needed",
                        "description": "Enterprise customer requiring PCI AOC for vendor risk assessment. Our integration: {integration_type}. SAQ level needed: {saq_level}. Deadline: {deadline}. Document location unclear in dashboard. Blocking customer contract renewal.",
                        "priority": ["high", "urgent"],
                        "resolution_time_range": (8, 48),
                    },
                    {
                        "subject": "GDPR data deletion request for customer {customer_id}",
                        "description": "Received GDPR Article 17 deletion request from customer {customer_id}. Need to delete: payment methods, charges, invoices, personal data. Customer since {date}. Total transactions: {count}. Confirm 30-day deletion timeline and data retention policies.",
                        "priority": ["high", "urgent"],
                        "resolution_time_range": (4, 72),
                    },
                    {
                        "subject": "Fraud alert - suspected card testing on account",
                        "description": "Radar flagged {count} transactions from IP {ip_address} in {timeframe}. Pattern: small amounts ${min_amount}-${max_amount}, rapid succession, different cards. Risk score: {risk_score}. No successful payments yet. Need to block and review.",
                        "priority": ["urgent"],
                        "resolution_time_range": (1, 2),
                    },
                    {
                        "subject": "Card data storage compliance question",
                        "description": "Implementing {use_case} feature. Question: Can we store last 4 digits and brand for display? Current approach: {current_approach}. Concerned about PCI DSS scope. Need clarity on what's permissible vs what requires SAQ.",
                        "priority": ["medium"],
                        "resolution_time_range": (4, 24),
                    },
                    {
                        "subject": "Security audit - need architecture documentation",
                        "description": "Enterprise customer security audit requesting: 1) Data flow diagrams, 2) Encryption methods, 3) Key management, 4) Access controls. Timeline: {timeline}. Docs location: {location}. Missing: {missing_items}. Blocking ${{deal_size}} deal.",
                        "priority": ["high", "urgent"],
                        "resolution_time_range": (8, 48),
                    },
                ],
            },
            "account_platform": {
                "weight": 0.05,
                "templates": [
                    {
                        "subject": "Connect account onboarding stuck at {step}",
                        "description": "Setting up Connect {account_type} account for platform marketplace. Stuck at {step}. Requirements: {requirements}. Submitted: {submitted}. Error: {error}. Account ID: {account_id}. {count} sellers waiting to go live.",
                        "priority": ["high", "medium"],
                        "resolution_time_range": (4, 24),
                    },
                    {
                        "subject": "Multi-user dashboard access not working",
                        "description": "Added team member {email} with {role} permissions {time_ago}. They confirmed email but can't access dashboard. Error: {error}. Other team members accessing fine. Account type: {account_type}. Plan: {plan}.",
                        "priority": ["medium"],
                        "resolution_time_range": (1, 4),
                    },
                    {
                        "subject": "Payout schedule change request",
                        "description": "Current payout schedule: {current_schedule}. Want to change to: {target_schedule}. Business reason: {reason}. Account: {account_id}. MRR: ${mrr}. Any holds or restrictions? Process and timeline?",
                        "priority": ["low", "medium"],
                        "resolution_time_range": (2, 12),
                    },
                    {
                        "subject": "Express Dashboard customization needed",
                        "description": "Using Connect Express accounts. Want to customize Express Dashboard with: {customizations}. Branding: {branding}. Currently showing: {current}. Target: {target}. Documentation unclear on limitations.",
                        "priority": ["low"],
                        "resolution_time_range": (4, 24),
                    },
                ],
            },
        }

        # Stripe-specific payment values
        self.payment_methods = [
            "card",
            "ach_debit",
            "sepa_debit",
            "ideal",
            "klarna",
            "affirm",
            "us_bank_account",
        ]
        self.error_codes = [
            "card_declined",
            "insufficient_funds",
            "invalid_cvc",
            "expired_card",
            "processing_error",
        ]
        self.plans = ["Integrated", "Standard", "Plus", "Enterprise"]
        self.regions = ["US", "EU", "APAC", "UK", "Canada", "Australia"]
        self.banks = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "HSBC"]
        self.currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]

        # Legacy attributes for template compatibility (can be removed after full migration)
        self.destinations = ["Dashboard", "API", "Webhook", "Report"]
        self.sources = ["Web", "Mobile", "Server", "POS"]
        self.platforms = ["Node.js", "Python", "Ruby", "Java", "Go", "PHP"]
        self.customer_tiers = ["starter", "growth", "business", "enterprise"]

    def generate_tickets(
        self, num_tickets: int = 500, date_range_days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic support tickets.

        Args:
            num_tickets: Number of tickets to generate
            date_range_days: Date range for ticket creation

        Returns:
            List of ticket dictionaries
        """
        self.logger.info(f"Generating {num_tickets} support tickets...")

        tickets = []
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=date_range_days)

        for i in range(num_tickets):
            # Select category based on weights
            category = self._select_category()
            template = random.choice(self.categories[category]["templates"])

            # Generate ticket
            ticket = self._generate_ticket(
                category, template, start_date, end_date, i + 1
            )

            tickets.append(ticket)

        self.logger.info(f"Generated {len(tickets)} tickets")
        return tickets

    def _select_category(self) -> str:
        """Select a category based on weights."""
        categories = list(self.categories.keys())
        weights = [self.categories[cat]["weight"] for cat in categories]
        return random.choices(categories, weights=weights)[0]

    def _generate_ticket(
        self,
        category: str,
        template: Dict,
        start_date: datetime,
        end_date: datetime,
        ticket_num: int,
    ) -> Dict[str, Any]:
        """Generate a single ticket from template."""

        # Generate timestamp
        created_at = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        # Fill template variables
        subject = self._fill_template(template["subject"])
        description = self._fill_template(template["description"])

        # Select priority
        priority = random.choice(template["priority"])

        # Calculate resolution time
        min_hours, max_hours = template["resolution_time_range"]
        resolution_hours = random.uniform(min_hours, max_hours)
        resolved_at = created_at + timedelta(hours=resolution_hours)

        # Generate customer context
        customer_tier = random.choice(self.customer_tiers)
        customer_plan = random.choice(self.plans)

        # Satisfaction score (higher for faster resolution)
        if resolution_hours < 2:
            satisfaction_score = random.randint(4, 5)
        elif resolution_hours < 8:
            satisfaction_score = random.randint(3, 5)
        elif resolution_hours < 24:
            satisfaction_score = random.randint(2, 4)
        else:
            satisfaction_score = random.randint(1, 3)

        return {
            "ticket_id": f"TKT-2024-{ticket_num:05d}",
            "created_at": created_at.isoformat(),
            "resolved_at": resolved_at.isoformat(),
            "priority": priority,
            "category": category,
            "subcategory": self._get_subcategory(category),
            "customer_tier": customer_tier,
            "subject": subject,
            "description": description,
            "customer_context": {
                "plan": customer_plan,
                "monthly_events": random.choice(
                    [50000, 100000, 500000, 1000000, 5000000]
                ),
                "sources": random.sample(self.sources, random.randint(1, 3)),
                "destinations": random.sample(self.destinations, random.randint(1, 4)),
                "account_age_months": random.randint(1, 36),
            },
            "resolution_time_hours": round(resolution_hours, 2),
            "satisfaction_score": satisfaction_score,
            "tags": self._generate_tags(category, subject, description),
            "agent_handoffs": random.randint(0, 2),
            "status": "resolved",
        }

    def _fill_template(self, template: str) -> str:
        """Fill template with random values."""
        replacements = {
            # Stripe-specific payment values
            "{error_code}": random.choice(self.error_codes),
            "{payment_method}": random.choice(self.payment_methods),
            "{customer_id}": f"cus_{random.randint(100000, 999999)}",
            "{amount}": str(random.randint(10, 10000)),
            "{use_case}": random.choice(
                ["subscription renewal", "checkout", "payment"]
            ),
            "{charge_id}": f"ch_{random.randint(100000, 999999)}",
            "{bank}": random.choice(self.banks),
            "{expected_date}": (
                datetime.utcnow() + timedelta(days=random.randint(3, 7))
            ).strftime("%Y-%m-%d"),
            "{region}": random.choice(self.regions),
            "{step}": random.choice(["challenge", "redirect", "fingerprint"]),
            "{browser}": random.choice(["Chrome", "Safari", "Firefox", "Edge"]),
            "{revenue_impact}": str(random.randint(1000, 50000)),
            "{country}": random.choice(["France", "Germany", "Japan", "UK"]),
            "{currency}": random.choice(self.currencies),
            "{converted_amount}": str(random.randint(10, 10000)),
            "{local_currency}": random.choice(["EUR", "GBP", "JPY"]),
            "{rate}": f"{random.uniform(0.8, 1.2):.4f}",
            "{expected_rate}": f"{random.uniform(0.8, 1.2):.4f}",
            "{discrepancy}": str(random.randint(1, 10)),
            "{subscription_id}": f"sub_{random.randint(100000, 999999)}",
            "{retry_count}": str(random.randint(3, 5)),
            "{actual_attempts}": str(random.randint(0, 2)),
            "{event_types}": random.choice(["payment_intent", "charge", "invoice"]),
            "{code_snippet}": "hmac.new(secret.encode(), msg.encode(), hashlib.sha256)",
            "{endpoint}": random.choice(
                ["/v1/payment_intents", "/v1/charges", "/v1/customers"]
            ),
            "{pi_id}": f"pi_{random.randint(100000, 999999)}",
            "{sdk}": random.choice(["JavaScript", "iOS", "Android", "React Native"]),
            "{version}": f"{random.randint(1, 5)}.{random.randint(0, 20)}.0",
            "{platform}": random.choice(self.platforms),
            "{environment}": random.choice(["Node v18", "Python 3.11", "Ruby 3.2"]),
            "{doc_link}": "https://stripe.com/docs",
            "{attempted_solutions}": random.choice(
                ["cleared cache", "updated SDK", "checked API keys"]
            ),
            "{timeout}": str(random.choice([30, 60, 90])),
            "{duration}": str(random.randint(2, 10)),
            "{invoice_id}": f"in_{random.randint(100000, 999999)}",
            "{expected_amount}": str(random.randint(50, 500)),
            "{line_items}": "base plan + overages",
            "{plan}": random.choice(self.plans),
            "{old_plan}": "Standard",
            "{upgrade_date}": (
                datetime.utcnow() - timedelta(days=random.randint(1, 30))
            ).strftime("%Y-%m-%d"),
            "{current}": random.choice(["monthly", "annual"]),
            "{target}": random.choice(["annual", "monthly"]),
            "{mrr}": str(random.randint(100, 10000)),
            "{old_price}": str(random.randint(50, 200)),
            "{new_price}": str(random.randint(100, 500)),
            "{date}": (
                datetime.utcnow() - timedelta(days=random.randint(1, 15))
            ).strftime("%Y-%m-%d"),
            "{expected}": str(random.randint(50, 200)),
            "{actual}": str(random.randint(60, 250)),
            "{difference}": str(random.randint(10, 50)),
            "{period}": "monthly",
            "{payment_method_id}": f"pm_{random.randint(100000, 999999)}",
            "{test_amount}": "1.00",
            "{failure_code}": random.choice(["card_declined", "insufficient_funds"]),
            "{tax_amount}": str(random.randint(10, 100)),
            "{tax_rate}": str(random.uniform(5, 10)),
            "{expected_rate}": str(random.uniform(5, 10)),
            "{tax_type}": "VAT",
            "{address}": "123 Main St, City, Country",
            "{product_type}": "SaaS subscription",
            "{integration_type}": random.choice(
                ["Stripe.js", "PaymentIntents", "Checkout"]
            ),
            "{saq_level}": random.choice(["SAQ A", "SAQ A-EP", "SAQ D"]),
            "{deadline}": (
                datetime.utcnow() + timedelta(days=random.randint(7, 30))
            ).strftime("%Y-%m-%d"),
            "{ip_address}": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "{timeframe}": random.choice(["5 minutes", "10 minutes", "30 minutes"]),
            "{min_amount}": str(random.randint(1, 10)),
            "{max_amount}": str(random.randint(20, 100)),
            "{risk_score}": str(random.randint(60, 95)),
            "{current_approach}": "storing full card data",
            "{timeline}": random.choice(["1 week", "2 weeks", "1 month"]),
            "{location}": "Stripe Dashboard > Security",
            "{missing_items}": "encryption details",
            "{deal_size}": str(random.randint(50000, 500000)),
            "{account_type}": random.choice(["Express", "Custom", "Standard"]),
            "{requirements}": "business verification",
            "{submitted}": "tax ID, business details",
            "{error}": random.choice(["verification_failed", "document_invalid"]),
            "{account_id}": f"acct_{random.randint(100000, 999999)}",
            "{email}": f"user{random.randint(100, 999)}@example.com",
            "{role}": random.choice(["Admin", "Developer", "Viewer"]),
            "{current_schedule}": "daily",
            "{target_schedule}": "weekly",
            "{reason}": "cash flow management",
            "{customizations}": "logo, colors, terms",
            "{branding}": "company logo",
            # Generic values for compatibility
            "{destination}": random.choice(self.destinations),
            "{source}": random.choice(self.sources),
            "{time_ago}": random.choice(
                ["2 hours ago", "yesterday", "3 days ago", "last week"]
            ),
            "{volume}": str(random.choice([100, 500, 1000, 5000, 10000])),
            "{percentage}": str(random.randint(10, 95)),
            "{count}": str(random.randint(2, 10)),
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def _get_subcategory(self, category: str) -> str:
        """Get subcategory for a category."""
        subcategories = {
            "integration": [
                "data_delivery",
                "api_usage",
                "sdk_implementation",
                "transformation",
            ],
            "billing": ["overages", "plan_changes", "invoicing", "usage_analysis"],
            "privacy": ["gdpr", "ccpa", "data_deletion", "consent_management"],
            "performance": [
                "latency",
                "data_quality",
                "destination_sync",
                "deduplication",
            ],
            "account": ["access_control", "sso", "workspace_management", "permissions"],
        }
        return random.choice(subcategories.get(category, ["general"]))

    def _generate_tags(
        self, category: str, subject: str, description: str
    ) -> List[str]:
        """Generate relevant tags for ticket."""
        tags = [category]

        # Add destination/source tags
        for dest in self.destinations:
            if dest.lower() in subject.lower() or dest.lower() in description.lower():
                tags.append(dest.lower().replace(" ", "_"))

        # Add priority keywords
        keywords = [
            "urgent",
            "high_priority",
            "debugging",
            "compliance",
            "billing_issue",
        ]
        for keyword in keywords:
            if keyword.replace("_", " ") in description.lower():
                tags.append(keyword)

        return list(set(tags))[:5]  # Max 5 tags

    def save_tickets(
        self, tickets: List[Dict[str, Any]], split_by_quarter: bool = True
    ) -> None:
        """
        Save tickets to JSON files.

        Args:
            tickets: List of ticket dictionaries
            split_by_quarter: Whether to split by quarter
        """
        if split_by_quarter:
            # Group by quarter
            quarters = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}

            for ticket in tickets:
                created = datetime.fromisoformat(ticket["created_at"])
                quarter = f"Q{(created.month - 1) // 3 + 1}"
                quarters[quarter].append(ticket)

            # Save each quarter
            for quarter, quarter_tickets in quarters.items():
                if quarter_tickets:
                    filename = f"tickets_2024_{quarter.lower()}.json"
                    filepath = self.output_dir / filename

                    with open(filepath, "w") as f:
                        json.dump(quarter_tickets, f, indent=2)

                    self.logger.info(
                        f"Saved {len(quarter_tickets)} tickets to {filename}"
                    )
        else:
            # Save all together
            filepath = self.output_dir / "tickets_2024_all.json"
            with open(filepath, "w") as f:
                json.dump(tickets, f, indent=2)

            self.logger.info(f"Saved {len(tickets)} tickets to {filepath}")

        # Save summary statistics
        self._save_ticket_stats(tickets)

    def _save_ticket_stats(self, tickets: List[Dict[str, Any]]) -> None:
        """Save ticket statistics."""
        stats = {
            "total_tickets": len(tickets),
            "by_category": {},
            "by_priority": {},
            "by_customer_tier": {},
            "avg_resolution_hours": 0,
            "avg_satisfaction_score": 0,
        }

        resolution_times = []
        satisfaction_scores = []

        for ticket in tickets:
            # Category
            category = ticket["category"]
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            # Priority
            priority = ticket["priority"]
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

            # Customer tier
            tier = ticket["customer_tier"]
            stats["by_customer_tier"][tier] = stats["by_customer_tier"].get(tier, 0) + 1

            # Metrics
            resolution_times.append(ticket["resolution_time_hours"])
            satisfaction_scores.append(ticket["satisfaction_score"])

        stats["avg_resolution_hours"] = round(
            sum(resolution_times) / len(resolution_times), 2
        )
        stats["avg_satisfaction_score"] = round(
            sum(satisfaction_scores) / len(satisfaction_scores), 2
        )

        # Save stats
        stats_file = self.output_dir / "_ticket_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        self.logger.info(f"Ticket statistics saved to {stats_file}")

    def _calculate_stats(self, tickets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from generated tickets."""
        stats = {
            "total_tickets": len(tickets),
            "by_category": {},
            "by_priority": {},
            "by_customer_tier": {},
            "avg_resolution_hours": 0,
            "avg_satisfaction_score": 0,
        }

        resolution_times = []
        satisfaction_scores = []

        for ticket in tickets:
            # Category
            category = ticket["category"]
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

            # Priority
            priority = ticket["priority"]
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

            # Customer tier
            tier = ticket["customer_tier"]
            stats["by_customer_tier"][tier] = stats["by_customer_tier"].get(tier, 0) + 1

            # Metrics
            resolution_times.append(ticket["resolution_time_hours"])
            satisfaction_scores.append(ticket["satisfaction_score"])

        if resolution_times:
            stats["avg_resolution_hours"] = round(
                sum(resolution_times) / len(resolution_times), 2
            )
        if satisfaction_scores:
            stats["avg_satisfaction_score"] = round(
                sum(satisfaction_scores) / len(satisfaction_scores), 2
            )

        return stats


def generate_support_tickets(num_tickets: int = 500) -> Dict[str, Any]:
    """
    Generate synthetic support tickets.

    Args:
        num_tickets: Number of tickets to generate

    Returns:
        Generation results with statistics as dict (not list)
    """
    generator = SupportTicketGenerator()
    tickets = generator.generate_tickets(num_tickets)

    # Save tickets
    generator.save_tickets(tickets)

    # Calculate and return statistics as dict
    stats = generator._calculate_stats(tickets)

    # Ensure we return the stats dict, not the tickets list
    return stats


if __name__ == "__main__":
    generate_support_tickets()
