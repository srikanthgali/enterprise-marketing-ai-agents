"""
Web scraper for Stripe Documentation.

Extracts public documentation while respecting robots.txt and rate limits.
Converts content to clean markdown format for knowledge base.

Strategy: Multi-source approach targeting:
1. Main documentation guides
2. Complete API reference (all resources)
3. Integration guides
4. Use case documentation
"""

import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import json
from datetime import datetime, timezone
import time
from urllib.parse import urljoin, urlparse
import re
from urllib.robotparser import RobotFileParser

from src.marketing_agents.utils import get_logger, log_execution
from src.marketing_agents.utils.exceptions import AgentError
from config.settings import get_settings


class StripeDocsScraper:
    """
    Scraper for Stripe Documentation.

    Features:
    - Multi-source scraping strategy (guides + API reference + integrations)
    - Respects robots.txt (checks before scraping)
    - Rate limiting (2 seconds between requests)
    - Retry logic with exponential backoff
    - Clean markdown conversion
    - Comprehensive metadata extraction
    - Error handling and logging
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize scraper with compliance checks."""
        self.logger = get_logger(__name__)
        self.settings = get_settings()

        # Stripe Documentation base URL
        self.base_url = "https://stripe.com/docs/"
        self.robots_url = "https://stripe.com/robots.txt"

        # Output directory
        self.output_dir = output_dir or (
            self.settings.data_dir / "raw" / "knowledge_base" / "stripe_docs"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting (conservative: 2 seconds between requests)
        self.rate_limit_delay = 2.0
        self.last_request_time = 0
        self.max_concurrent_requests = 1  # Sequential scraping only

        # Robots.txt parser
        self.robot_parser = RobotFileParser()
        self.robot_parser.set_url(self.robots_url)
        self.robots_checked = False

        # Track visited URLs and failures
        self.visited_urls: Set[str] = set()
        self.failed_urls: List[Dict] = []
        self.robots_blocked_urls: List[str] = []

        # Track pending URLs to scrape (queue for BFS traversal)
        self.pending_urls: List[str] = []

        # Statistics
        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "pages_blocked_by_robots": 0,
            "total_bytes": 0,
            "start_time": None,
            "end_time": None,
            "rate_limit_delays": 0,
            "retry_attempts": 0,
            "api_pages": 0,
            "guide_pages": 0,
            "integration_pages": 0,
        }

        # User agent for polite scraping
        self.user_agent = (
            "EnterpriseMarketingBot/1.0 (+https://github.com/yourorg/marketing-ai-agents; "
            "research@example.com) - Educational/Research Purpose"
        )

    async def check_robots_txt(self) -> bool:
        """
        Check and parse robots.txt before scraping.

        Returns:
            True if robots.txt loaded successfully
        """
        if self.robots_checked:
            return True

        try:
            self.logger.info(f"Checking robots.txt: {self.robots_url}")

            headers = {"User-Agent": self.user_agent}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.robots_url, timeout=10) as response:
                    if response.status == 200:
                        robots_content = await response.text()
                        self.robot_parser.parse(robots_content.splitlines())
                        self.robots_checked = True
                        self.logger.info("✓ robots.txt parsed successfully")

                        self._log_robots_rules()
                        return True
                    else:
                        self.logger.warning(
                            f"Could not fetch robots.txt (HTTP {response.status}). "
                            "Proceeding with caution..."
                        )
                        return False

        except Exception as e:
            self.logger.error(f"Error checking robots.txt: {e}")
            self.logger.warning("Proceeding without robots.txt validation")
            return False

    def _log_robots_rules(self) -> None:
        """Log relevant robots.txt rules for our user agent."""
        self.logger.info("Robots.txt Rules:")
        self.logger.info(f"  User-Agent: {self.user_agent}")

        test_paths = [
            "docs/",
            "docs/api",
            "docs/payments",
            "docs/billing",
            "docs/connect",
        ]

        for path in test_paths:
            can_fetch = self.robot_parser.can_fetch(
                self.user_agent, f"https://stripe.com/{path}"
            )
            status = "✓ Allowed" if can_fetch else "✗ Blocked"
            self.logger.info(f"  /{path}: {status}")

    def can_fetch_url(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.

        Args:
            url: URL to check

        Returns:
            True if allowed, False if blocked
        """
        if not self.robots_checked:
            return True

        can_fetch = self.robot_parser.can_fetch(self.user_agent, url)

        if not can_fetch:
            self.stats["pages_blocked_by_robots"] += 1
            self.robots_blocked_urls.append(url)
            self.logger.debug(f"Blocked by robots.txt: {url}")

        return can_fetch

    def _generate_api_reference_urls(self) -> List[str]:
        """
        Generate URLs for Stripe API reference documentation.

        Stripe API has predictable URL patterns for resources.
        Returns comprehensive list of API endpoint documentation.
        """
        # Core API resources (verified from Stripe docs structure)
        api_resources = [
            # Core resources
            "balance",
            "balance_transactions",
            "charges",
            "customers",
            "disputes",
            "events",
            "files",
            "file_links",
            "mandates",
            "payouts",
            "refunds",
            "tokens",
            "payment_intents",
            "setup_intents",
            "setup_attempts",
            "payment_methods",
            # Payment methods
            "payment_method_configurations",
            "sources",
            "cards",
            "bank_accounts",
            "cash_balance",
            "funding_instructions",
            # Products and prices
            "products",
            "prices",
            "coupons",
            "promotion_codes",
            "discounts",
            "tax_codes",
            "tax_rates",
            "shipping_rates",
            # Checkout
            "checkout/sessions",
            "payment_links",
            # Billing
            "subscriptions",
            "subscription_items",
            "subscription_schedules",
            "usage_records",
            "invoices",
            "invoice_items",
            "credit_notes",
            "customer_balance_transactions",
            "customer_tax_ids",
            "quotes",
            "quote_line_items",
            # Connect
            "accounts",
            "application_fees",
            "application_fee_refunds",
            "capabilities",
            "country_specs",
            "external_accounts",
            "persons",
            "topups",
            "transfers",
            "transfer_reversals",
            "account_links",
            "account_sessions",
            # Issuing
            "issuing/authorizations",
            "issuing/cardholders",
            "issuing/cards",
            "issuing/disputes",
            "issuing/transactions",
            # Terminal
            "terminal/configurations",
            "terminal/connection_tokens",
            "terminal/locations",
            "terminal/readers",
            # Financial Connections
            "financial_connections/accounts",
            "financial_connections/sessions",
            "financial_connections/transactions",
            # Identity
            "identity/verification_sessions",
            "identity/verification_reports",
            # Radar
            "radar/early_fraud_warnings",
            "radar/value_lists",
            "radar/value_list_items",
            "reviews",
            # Webhooks
            "webhook_endpoints",
            # Reporting
            "reporting/report_runs",
            "reporting/report_types",
            # Sigma
            "sigma/scheduled_query_runs",
            # Climate
            "climate/orders",
            "climate/products",
            "climate/suppliers",
        ]

        api_urls = []
        for resource in api_resources:
            api_urls.append(f"https://stripe.com/docs/api/{resource}")

        self.logger.info(f"Generated {len(api_urls)} API reference URLs")
        return api_urls

    def _generate_guide_urls(self) -> List[str]:
        """
        Generate URLs for Stripe guide documentation.

        Returns comprehensive list of guide pages.
        """
        guide_sections = [
            # Getting started
            "development",
            "development/quickstart",
            "development/dashboard",
            "development/dashboard/webhooks",
            # Payments
            "payments",
            "payments/accept-a-payment",
            "payments/payment-intents",
            "payments/save-and-reuse",
            "payments/payment-methods",
            "payments/payment-methods/overview",
            "payments/cards",
            "payments/ach-debit",
            "payments/sepa-debit",
            "payments/ideal",
            "payments/giropay",
            "payments/sofort",
            "payments/alipay",
            "payments/wechat-pay",
            "payments/bancontact",
            "payments/eps",
            "payments/p24",
            "payments/fpx",
            "payments/grabpay",
            "payments/boleto",
            "payments/oxxo",
            "payments/afterpay-clearpay",
            "payments/klarna",
            "payments/affirm",
            "payments/us-bank-account",
            "payments/link",
            "payments/cashapp",
            "payments/customer-balance",
            # Checkout
            "checkout",
            "checkout/quickstart",
            "checkout/how-checkout-works",
            "checkout/embedded/quickstart",
            "payment-links",
            "payment-links/overview",
            # Billing
            "billing",
            "billing/subscriptions/overview",
            "billing/subscriptions/creating",
            "billing/subscriptions/upgrade-downgrade",
            "billing/subscriptions/usage-based",
            "billing/subscriptions/metered-billing",
            "billing/subscriptions/trials",
            "billing/subscriptions/discounts",
            "billing/subscriptions/webhooks",
            "billing/invoices",
            "billing/invoices/overview",
            "billing/invoices/workflow",
            "billing/customer-portal",
            "billing/prices-guide",
            "billing/taxes",
            "billing/taxes/tax-rates",
            "billing/taxes/tax-ids",
            # Connect
            "connect",
            "connect/enable-payment-acceptance-guide",
            "connect/collect-then-transfer-guide",
            "connect/charges",
            "connect/payouts",
            "connect/account-capabilities",
            "connect/account-tokens",
            "connect/identity-verification",
            "connect/express-accounts",
            "connect/custom-accounts",
            "connect/standard-accounts",
            # Security
            "security",
            "security/guide",
            "security/stripe",
            "security/encryption",
            "security/pci-compliance",
            "security/radar",
            "security/sca",
            "security/3d-secure",
            # Webhooks
            "webhooks",
            "webhooks/quickstart",
            "webhooks/integration-builder",
            "webhooks/best-practices",
            "webhooks/signatures",
            # Testing
            "testing",
            # Reporting
            "reporting",
            "reports",
            "reports/reporting-categories",
            # Customer management
            "customer-management",
            "customer-management/overview",
            # Fraud prevention
            "fraud",
            "fraud/radar-for-fraud-teams",
            "radar",
            "radar/rules",
            "radar/risk-evaluation",
            # Financial Connections
            "financial-connections",
            # Identity
            "identity",
            "identity/how-sessions-work",
            # Issuing
            "issuing",
            "issuing/quickstart",
            # Terminal
            "terminal",
            "terminal/payments",
            # Climate
            "climate",
            # Treasury
            "treasury",
            # Tax
            "tax",
            "tax/checkout",
            "tax/invoicing",
        ]

        guide_urls = []
        for section in guide_sections:
            guide_urls.append(f"https://stripe.com/docs/{section}")

        self.logger.info(f"Generated {len(guide_urls)} guide URLs")
        return guide_urls

    @log_execution()
    async def scrape_documentation(
        self, sections: Optional[List[str]] = None, max_pages: int = 300
    ) -> Dict[str, Any]:
        """
        Scrape Stripe Documentation using multi-source strategy.

        Args:
            sections: Specific sections to scrape (None = comprehensive scrape)
            max_pages: Maximum number of pages to scrape

        Returns:
            Scraping statistics
        """
        self.stats["start_time"] = datetime.now(timezone.utc).isoformat()

        await self.check_robots_txt()

        # Generate comprehensive URL list from multiple sources
        all_urls = []

        # 1. API Reference (100+ pages)
        api_urls = self._generate_api_reference_urls()
        all_urls.extend(api_urls)

        # 2. Guide documentation (100+ pages)
        guide_urls = self._generate_guide_urls()
        all_urls.extend(guide_urls)

        # 3. Custom sections if provided
        if sections:
            for section in sections:
                section_url = urljoin(self.base_url, section)
                if section_url not in all_urls:
                    all_urls.append(section_url)

        self.logger.info(
            f"Starting Stripe documentation scrape: {len(all_urls)} URLs queued"
        )
        self.logger.info(f"  - API reference URLs: {len(api_urls)}")
        self.logger.info(f"  - Guide URLs: {len(guide_urls)}")
        self.logger.info(f"Max pages limit: {max_pages}")
        self.logger.info(
            f"Strategy: Comprehensive multi-source scraping with link discovery"
        )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # Initialize pending URLs queue
        self.pending_urls = all_urls.copy()

        async with aiohttp.ClientSession(
            headers=headers,
            connector=aiohttp.TCPConnector(limit=self.max_concurrent_requests),
        ) as session:
            # Process URLs in queue (breadth-first traversal)
            while self.pending_urls and self.stats["pages_scraped"] < max_pages:
                url = self.pending_urls.pop(0)

                if url in self.visited_urls:
                    continue

                if not self.can_fetch_url(url):
                    continue

                await self._scrape_page(session, url)

                # Log progress every 20 pages
                if (
                    self.stats["pages_scraped"] > 0
                    and self.stats["pages_scraped"] % 20 == 0
                ):
                    self.logger.info(
                        f"Progress: {self.stats['pages_scraped']}/{max_pages} pages scraped, "
                        f"{len(self.pending_urls)} URLs in queue"
                    )

        self.stats["end_time"] = datetime.now(timezone.utc).isoformat()

        self._save_metadata()

        self.logger.info(
            f"Scraping complete: {self.stats['pages_scraped']} pages scraped, "
            f"{self.stats['pages_failed']} failed, "
            f"{self.stats['pages_blocked_by_robots']} blocked by robots.txt"
        )
        self.logger.info(
            f"  - API pages: {self.stats['api_pages']}, "
            f"Guide pages: {self.stats['guide_pages']}, "
            f"Integration pages: {self.stats['integration_pages']}"
        )

        return self.stats

    async def _scrape_page(self, session: aiohttp.ClientSession, url: str) -> None:
        """
        Scrape a single documentation page and extract links.

        Args:
            session: aiohttp session
            url: Page URL to scrape
        """
        if url in self.visited_urls:
            return

        try:
            await self._rate_limit()

            html_content = await self._fetch_page(session, url)
            if not html_content:
                self.visited_urls.add(url)
                return

            soup = BeautifulSoup(html_content, "html.parser")

            # Extract and save content
            content, metadata = self._extract_content(soup, url)
            if content:
                filename = self._generate_filename(url)
                filepath = self.output_dir / filename

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                metadata_file = filepath.with_suffix(".json")
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                self.stats["pages_scraped"] += 1
                self.stats["total_bytes"] += len(content.encode("utf-8"))

                # Categorize page type
                if "/api/" in url:
                    self.stats["api_pages"] += 1
                elif any(
                    x in url for x in ["/checkout", "/payments", "/billing", "/connect"]
                ):
                    self.stats["guide_pages"] += 1
                else:
                    self.stats["integration_pages"] += 1

                self.logger.debug(f"✓ Scraped: {url} -> {filename}")

            # Extract and queue new links (still useful for discovering related pages)
            new_links = self._extract_related_links(soup, url)
            for link in new_links[:10]:  # Limit to 10 most relevant per page
                if link not in self.visited_urls and link not in self.pending_urls:
                    self.pending_urls.append(link)

            self.visited_urls.add(url)

        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}", exc_info=True)
            self.failed_urls.append(
                {
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self.stats["pages_failed"] += 1
            self.visited_urls.add(url)  # Mark as visited to avoid retry loops

    async def _fetch_page(
        self, session: aiohttp.ClientSession, url: str, retries: int = 3
    ) -> Optional[str]:
        """Fetch page content with retry logic and error handling."""
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 403:
                        self.logger.warning(
                            f"HTTP 403 Forbidden for {url} - "
                            "Site may be blocking automated access"
                        )
                        return None
                    elif response.status == 404:
                        self.logger.warning(f"HTTP 404 Not Found: {url}")
                        return None
                    elif response.status == 429:
                        wait_time = 2 ** (attempt + 1)
                        self.logger.warning(
                            f"Rate limited (HTTP 429), waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        self.stats["retry_attempts"] += 1
                    elif response.status >= 500:
                        wait_time = 2**attempt
                        self.logger.warning(
                            f"Server error {response.status}, retrying in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        self.stats["retry_attempts"] += 1
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout fetching {url}, attempt {attempt + 1}/{retries}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                self.stats["retry_attempts"] += 1
            except aiohttp.ClientError as e:
                self.logger.error(f"Client error fetching {url}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
                self.stats["retry_attempts"] += 1
            except Exception as e:
                self.logger.error(
                    f"Unexpected error fetching {url}: {e}", exc_info=True
                )
                return None

        self.logger.error(f"Failed to fetch {url} after {retries} attempts")
        return None

    def _extract_content(
        self, soup: BeautifulSoup, url: str
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Extract and clean main content from Stripe documentation page.

        Args:
            soup: BeautifulSoup parsed HTML
            url: Source URL

        Returns:
            Tuple of (markdown_content, metadata_dict)
        """
        # Stripe-specific content selectors (more comprehensive)
        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_="DocSearch-content")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_="container")
            or soup.find("div", id="content")
        )

        if not article:
            # Try broader selection
            article = soup.find("body")
            if not article:
                return None, {}

        # Remove unwanted elements
        for element in article.find_all(
            ["nav", "script", "style", "aside", "footer", "iframe", "header", "form"]
        ):
            element.decompose()

        # Also remove common navigation/UI elements
        for selector in [
            {"class": "nav"},
            {"class": "navigation"},
            {"class": "sidebar"},
            {"class": "header"},
            {"class": "footer"},
            {"id": "header"},
            {"id": "footer"},
        ]:
            for element in article.find_all(**selector):
                element.decompose()

        # Extract metadata
        title_elem = soup.find("h1") or soup.find("title")
        title = title_elem.get_text().strip() if title_elem else "Untitled"

        # Extract breadcrumb/categories
        breadcrumb = soup.find("nav", attrs={"aria-label": "Breadcrumb"}) or soup.find(
            "ol", class_="breadcrumb"
        )
        categories = []
        if breadcrumb:
            for link in breadcrumb.find_all("a"):
                cat_text = link.get_text().strip()
                if cat_text:
                    categories.append(cat_text)

        # Extract description meta tag
        description = ""
        meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
            "meta", attrs={"property": "og:description"}
        )
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"]

        # Build metadata
        metadata = {
            "title": title,
            "url": url,
            "source": "Stripe Documentation",
            "author": "Stripe",
            "date_extracted": datetime.now(timezone.utc).isoformat(),
            "categories": categories,
            "description": description,
            "content_type": "documentation",
        }

        # Convert to markdown
        content_lines = [
            f"# {title}",
            f"\n**Source:** {url}",
            f"**Categories:** {' > '.join(categories) if categories else 'Stripe Documentation'}",
            f"**Extracted:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        ]

        if description:
            content_lines.append(f"\n**Description:** {description}")

        content_lines.append("\n---\n")

        # Extract content with structure preservation
        content_elements = article.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "pre",
                "code",
                "blockquote",
                "table",
                "div",
            ]
        )

        for element in content_elements:
            if element.name.startswith("h"):
                level = int(element.name[1])
                text = element.get_text().strip()
                if text:
                    content_lines.append(f"\n{'#' * level} {text}\n")

            elif element.name == "p":
                text = element.get_text().strip()
                if text and len(text) > 10:  # Filter out very short paragraphs
                    content_lines.append(f"{text}\n")

            elif element.name in ["ul", "ol"]:
                items = element.find_all("li", recursive=False)
                if items:
                    for li in items:
                        text = li.get_text().strip()
                        if text:
                            content_lines.append(f"- {text}")
                    content_lines.append("")

            elif element.name in ["pre", "code"]:
                code_text = element.get_text().strip()
                if code_text and len(code_text) > 5:
                    content_lines.append(f"```\n{code_text}\n```\n")

            elif element.name == "blockquote":
                quote_text = element.get_text().strip()
                if quote_text:
                    content_lines.append(f"> {quote_text}\n")

            elif element.name == "table":
                rows = element.find_all("tr")
                if rows:
                    content_lines.append("\n")
                    for row in rows:
                        cells = [
                            cell.get_text().strip()
                            for cell in row.find_all(["td", "th"])
                        ]
                        if cells:
                            content_lines.append(f"| {' | '.join(cells)} |")
                            if row == rows[0]:  # Add separator after header
                                content_lines.append(f"|{'---|' * len(cells)}")
                    content_lines.append("")

        markdown_content = "\n".join(content_lines)

        # Filter out pages with minimal content (but be less aggressive)
        if len(markdown_content) < 150:
            return None, {}

        return markdown_content, metadata

    def _extract_related_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract related documentation links from page (more aggressive).

        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for relative links

        Returns:
            List of related URLs
        """
        links = []
        base_domain = urlparse(base_url).netloc

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchor links, mailto, javascript, etc.
            if href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue

            full_url = urljoin(base_url, href)

            # Only include Stripe docs URLs (be more inclusive)
            if (
                full_url.startswith("https://stripe.com/docs/")
                and urlparse(full_url).netloc == base_domain
            ):
                # Remove anchors and query params
                clean_url = full_url.split("#")[0].split("?")[0]

                # Avoid certain patterns that are not useful
                skip_patterns = [
                    "/docs/search",
                    "/docs/api/changelog",
                    "/docs/downloads",
                ]

                if not any(pattern in clean_url for pattern in skip_patterns):
                    if clean_url not in links and clean_url != base_url:
                        links.append(clean_url)

        return links

    def _generate_filename(self, url: str) -> str:
        """
        Generate safe filename from URL.

        Args:
            url: Source URL

        Returns:
            Safe filename
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")

        # Remove 'docs_' prefix if present
        if path.startswith("docs_"):
            path = path[5:]

        # Clean up filename
        filename = re.sub(r"[^\w\-_]", "_", path)
        filename = re.sub(r"_+", "_", filename)
        filename = filename.strip("_")

        if not filename or filename == "docs":
            # Generate from URL hash as fallback
            import hashlib

            filename = f"page_{hashlib.md5(url.encode()).hexdigest()[:8]}"

        return f"{filename}.md"

    async def _rate_limit(self) -> None:
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            delay = self.rate_limit_delay - time_since_last
            await asyncio.sleep(delay)
            self.stats["rate_limit_delays"] += 1

        self.last_request_time = time.time()

    def _save_metadata(self) -> None:
        """Save comprehensive scraping metadata."""
        metadata = {
            "scraper": "StripeDocsScraper",
            "version": "1.0.0",
            "base_url": self.base_url,
            "user_agent": self.user_agent,
            "robots_txt_url": self.robots_url,
            "robots_txt_checked": self.robots_checked,
            "rate_limit_delay_seconds": self.rate_limit_delay,
            "stats": self.stats,
            "visited_urls": list(self.visited_urls),
            "failed_urls": self.failed_urls,
            "robots_blocked_urls": self.robots_blocked_urls,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        metadata_file = self.output_dir / "_scraper_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"✓ Metadata saved to {metadata_file}")

        self._save_summary_report()

    def _save_summary_report(self) -> None:
        """Generate human-readable summary report."""
        report_lines = [
            "=" * 80,
            "STRIPE DOCUMENTATION SCRAPING REPORT",
            "=" * 80,
            f"\nScraper: StripeDocsScraper v1.0.0",
            f"Start Time: {self.stats.get('start_time', 'N/A')}",
            f"End Time: {self.stats.get('end_time', 'N/A')}",
            f"\n{'-' * 80}",
            f"\nRESULTS:",
            f"  Pages Successfully Scraped: {self.stats['pages_scraped']}",
            f"  Pages Failed: {self.stats['pages_failed']}",
            f"  Pages Blocked by robots.txt: {self.stats['pages_blocked_by_robots']}",
            f"  Total Data Size: {self.stats['total_bytes'] / 1024:.2f} KB",
            f"\n{'-' * 80}",
            f"\nCOMPLIANCE:",
            f"  robots.txt Checked: {'✓ Yes' if self.robots_checked else '✗ No'}",
            f"  Rate Limit Delays: {self.stats['rate_limit_delays']}",
            f"  Retry Attempts: {self.stats['retry_attempts']}",
            f"  User Agent: {self.user_agent}",
            f"\n{'-' * 80}",
        ]

        if self.failed_urls:
            report_lines.append(f"\nFAILED URLS ({len(self.failed_urls)}):")
            for failure in self.failed_urls[:10]:
                report_lines.append(f"  - {failure['url']}")
                report_lines.append(f"    Error: {failure['error']}")

        if self.robots_blocked_urls:
            report_lines.append(
                f"\nBLOCKED BY ROBOTS.TXT ({len(self.robots_blocked_urls)}):"
            )
            for url in self.robots_blocked_urls[:10]:
                report_lines.append(f"  - {url}")

        report_lines.append("\n" + "=" * 80)

        report_file = self.output_dir / "_scraping_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        self.logger.info(f"✓ Summary report saved to {report_file}")

    async def scrape_docs(self) -> Dict[str, Any]:
        """
        Main entry point for scraping Stripe documentation.

        Returns:
            Scraping results with statistics
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Stripe Documentation scraping...")
        self.logger.info("=" * 80)

        try:
            stats = await self.scrape_documentation()

            return {
                "status": "success",
                "pages_scraped": stats.get("pages_scraped", 0),
                "pages_failed": stats.get("pages_failed", 0),
                "pages_blocked": stats.get("pages_blocked_by_robots", 0),
                "total_pages": stats.get("pages_scraped", 0),
                "output_dir": str(self.output_dir),
                "compliance": {
                    "robots_txt_checked": self.robots_checked,
                    "rate_limited": True,
                    "user_agent": self.user_agent,
                },
            }

        except Exception as e:
            self.logger.error(f"Scraping failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "pages_scraped": self.stats.get("pages_scraped", 0),
                "output_dir": str(self.output_dir),
            }


async def scrape_stripe_docs():
    """Convenience function to scrape Stripe documentation."""
    scraper = StripeDocsScraper()
    return await scraper.scrape_documentation()


if __name__ == "__main__":
    asyncio.run(scrape_stripe_docs())
