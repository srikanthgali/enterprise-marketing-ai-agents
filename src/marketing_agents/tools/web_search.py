"""
Web search tool using Serper API for market research and competitor analysis.

Provides real-time web search capabilities for marketing intelligence gathering.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool using Serper API.

    Capabilities:
    - General web search
    - News search
    - Market trend research
    - Competitor information gathering
    """

    def __init__(self):
        """Initialize web search tool with Serper API."""
        self.settings = get_settings()
        self.api_key = self.settings.api.serper_api_key

        if not self.api_key:
            logger.warning("Serper API key not configured. Web search will be limited.")

        self.base_url = "https://google.serper.dev"
        self.cache = {}  # Simple in-memory cache

    async def search(
        self,
        query: str,
        search_type: str = "search",
        num_results: int = 10,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform web search using Serper API.

        Args:
            query: Search query string
            search_type: Type of search ('search', 'news', 'images')
            num_results: Number of results to return (default: 10)
            location: Optional geographic location for search

        Returns:
            Dictionary containing search results with metadata
        """
        if not self.api_key:
            logger.error("Serper API key not configured")
            return {"success": False, "error": "API key not configured", "results": []}

        # Check cache
        cache_key = f"{query}_{search_type}_{num_results}_{location}"
        if cache_key in self.cache:
            logger.debug(f"Returning cached results for query: {query}")
            return self.cache[cache_key]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "X-API-KEY": self.api_key.get_secret_value(),
                    "Content-Type": "application/json",
                }

                payload = {"q": query, "num": num_results}

                if location:
                    payload["location"] = location

                response = await client.post(
                    f"{self.base_url}/{search_type}", headers=headers, json=payload
                )

                response.raise_for_status()
                result = response.json()

                # Parse and structure results
                structured_result = self._parse_results(result, search_type)

                # Cache results
                self.cache[cache_key] = structured_result

                logger.info(f"Web search successful for query: {query}")
                return structured_result

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during web search: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "results": [],
            }
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def _parse_results(self, raw_results: Dict, search_type: str) -> Dict[str, Any]:
        """
        Parse and structure raw Serper API results.

        Args:
            raw_results: Raw API response
            search_type: Type of search performed

        Returns:
            Structured results dictionary
        """
        structured = {
            "success": True,
            "search_type": search_type,
            "results": [],
            "metadata": {},
        }

        if search_type == "search":
            # Organic search results
            organic = raw_results.get("organic", [])
            structured["results"] = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                }
                for item in organic
            ]

            # Knowledge graph if available
            if "knowledgeGraph" in raw_results:
                kg = raw_results["knowledgeGraph"]
                structured["metadata"]["knowledge_graph"] = {
                    "title": kg.get("title", ""),
                    "type": kg.get("type", ""),
                    "description": kg.get("description", ""),
                    "attributes": kg.get("attributes", {}),
                }

            # Related searches
            if "relatedSearches" in raw_results:
                structured["metadata"]["related_searches"] = [
                    item.get("query", "") for item in raw_results["relatedSearches"]
                ]

        elif search_type == "news":
            # News results
            news = raw_results.get("news", [])
            structured["results"] = [
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                }
                for item in news
            ]

        return structured

    async def search_market_trends(
        self,
        industry: str,
        keywords: Optional[List[str]] = None,
        time_range: str = "past_year",
    ) -> Dict[str, Any]:
        """
        Search for market trends in a specific industry.

        Args:
            industry: Industry or market segment
            keywords: Optional list of specific keywords to include
            time_range: Time range for trends (default: 'past_year')

        Returns:
            Dictionary with market trend insights
        """
        # Build comprehensive query
        query_parts = [f"{industry} market trends", time_range]
        if keywords:
            query_parts.extend(keywords)

        query = " ".join(query_parts)

        # Search both general and news
        results = await asyncio.gather(
            self.search(query, search_type="search", num_results=10),
            self.search(f"{industry} news trends", search_type="news", num_results=10),
        )

        return {
            "industry": industry,
            "query": query,
            "web_results": results[0],
            "news_results": results[1],
            "timestamp": asyncio.get_event_loop().time(),
        }

    async def search_competitor(
        self, competitor_name: str, aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for competitor information.

        Args:
            competitor_name: Name of the competitor
            aspects: Optional specific aspects to research (e.g., 'pricing', 'features')

        Returns:
            Dictionary with competitor insights
        """
        if not aspects:
            aspects = ["overview", "products", "pricing", "strategy"]

        # Perform multiple searches for different aspects
        searches = []
        for aspect in aspects:
            query = f"{competitor_name} {aspect}"
            searches.append(self.search(query, num_results=5))

        results = await asyncio.gather(*searches)

        # Combine results by aspect
        competitor_data = {"competitor": competitor_name, "aspects": {}}

        for aspect, result in zip(aspects, results):
            competitor_data["aspects"][aspect] = result

        return competitor_data

    def clear_cache(self) -> None:
        """Clear the search results cache."""
        self.cache.clear()
        logger.info("Web search cache cleared")
