"""
Data extraction and generation package.

Provides utilities for extracting marketing data from various sources
and generating synthetic training data.
"""

from .web_scraper import StripeDocsScraper
from .ticket_generator import SupportTicketGenerator, generate_support_tickets
from .campaign_generator import CampaignDataGenerator, generate_campaign_data
from .feedback_generator import FeedbackDataGenerator, generate_feedback_data

__all__ = [
    "StripeDocsScraper",
    "SupportTicketGenerator",
    "generate_support_tickets",
    "CampaignDataGenerator",
    "generate_campaign_data",
    "FeedbackDataGenerator",
    "generate_feedback_data",
]
