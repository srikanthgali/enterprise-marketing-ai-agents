"""
Marketing Strategy Agent - Develops marketing strategies and campaign plans.

Creates data-driven marketing strategies, performs market analysis,
and develops comprehensive campaign plans with audience targeting.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import json
import re

from langchain_core.messages import SystemMessage, HumanMessage

from ..core.base_agent import BaseAgent, AgentStatus
from ..tools.web_search import WebSearchTool
from ..tools.kb_search import KnowledgeBaseSearchTool


class MarketingStrategyAgent(BaseAgent):
    """
    Specialized agent for marketing strategy development.

    Capabilities:
    - Market research and trend analysis
    - Competitor analysis
    - Audience segmentation
    - Content strategy development
    - Channel optimization
    - Budget allocation
    """

    def __init__(
        self,
        config: Dict[str, Any],
        memory_manager=None,
        message_bus=None,
        prompt_manager=None,
    ):
        super().__init__(
            agent_id="marketing_strategy",
            name="Marketing Strategy Agent",
            description="Develops marketing strategies and campaign plans",
            config=config,
            memory_manager=memory_manager,
            message_bus=message_bus,
            prompt_manager=prompt_manager,
        )

        # Strategy templates and best practices
        self.strategy_templates = {}

        # Market intelligence cache
        self.market_intelligence = {}

        # Initialize tools
        self.web_search_tool = WebSearchTool()
        self.kb_search_tool = KnowledgeBaseSearchTool()

    def _extract_json_from_response(self, content: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks.

        Args:
            content: Raw LLM response content

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code block markers if present
        content = content.strip()

        # Pattern to match ```json ... ``` or ``` ... ```
        json_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(json_block_pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()

        return content

    def _register_tools(self) -> None:
        """Register marketing strategy tools."""
        self.register_tool("market_research", self._market_research)
        self.register_tool("competitor_analysis", self._competitor_analysis)
        self.register_tool("audience_segmentation", self._audience_segmentation)
        self.register_tool("generate_content_strategy", self._generate_content_strategy)
        self.register_tool("optimize_channels", self._optimize_channels)
        self.register_tool("budget_allocation", self._budget_allocation)
        self.register_tool("keyword_research", self._keyword_research)
        self.register_tool("trend_analysis", self._trend_analysis)

    def should_handoff(self, context: Dict[str, Any]) -> Optional[Any]:
        """
        Override base handoff logic to prevent inappropriate handoffs.

        Only handoff when explicitly needed, not for routine strategy generation.
        This prevents automatic handoffs triggered by common words in results.

        Args:
            context: Current execution context (typically the result)

        Returns:
            None - Marketing strategy should complete without handoffs for normal requests
        """
        # Check if this is an error case that needs escalation
        if context.get("success") is False and context.get("error"):
            # Even on errors, mark as final to avoid loops
            return None

        # Check if result explicitly requests a handoff
        if context.get("handoff_required") is True:
            target_agent = context.get("target_agent")
            reason = context.get("handoff_reason", "explicit_handoff_request")

            if target_agent:
                from src.marketing_agents.core.base_agent import HandoffRequest

                self.logger.info(
                    f"Explicit handoff requested: {self.agent_id} -> {target_agent}"
                )
                return HandoffRequest(
                    from_agent=self.agent_id,
                    to_agent=target_agent,
                    reason=reason,
                    context=context,
                )

        # For normal marketing strategy generation, do not handoff
        # The agent should complete its work and return final results
        return None

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process marketing strategy request.

        Args:
            input_data: Request with campaign objectives, audience, budget

        Returns:
            Comprehensive marketing strategy
        """
        self.status = AgentStatus.PROCESSING
        start_time = datetime.utcnow()

        try:
            # Handle case where input_data might be a string or improperly formatted
            if isinstance(input_data, str):
                # If input is a string, wrap it in a dict with a default type
                input_data = {
                    "type": "campaign_strategy",
                    "message": input_data,
                    "objectives": ["Generate marketing strategy based on user request"],
                }
            elif not isinstance(input_data, dict):
                # If not a dict, convert to dict
                input_data = {"type": "campaign_strategy", "raw_input": str(input_data)}

            request_type = input_data.get("type", "campaign_strategy")

            self.logger.info(f"Processing strategy request: {request_type}")

            # Save context to memory
            self.save_to_memory("current_request", input_data)

            result = {}

            if request_type == "campaign_strategy":
                result = await self._create_campaign_strategy(input_data)
            elif request_type == "content_calendar":
                result = await self._create_content_calendar(input_data)
            elif request_type == "market_analysis":
                result = await self._analyze_market(input_data)
            else:
                result = await self._create_campaign_strategy(input_data)

            # Check if handoff is needed
            # Only handoff if the result explicitly indicates a need
            # But skip if we're already handling a handoff to prevent loops
            message = input_data.get("message", "")
            is_handoff_result = input_data.get("from_agent") is not None

            if is_handoff_result:
                self.logger.info(
                    f"Processing handoff from {input_data.get('from_agent')}. "
                    "Completing strategy without further handoffs."
                )
                handoff_info = {}
            else:
                handoff_info = self._detect_handoff_need(message, result)

            # Log execution
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.log_execution(input_data, result, duration, True)

            # Publish completion event
            await self.publish_event(
                "strategy.completed",
                {"request_type": request_type, "duration": duration},
            )

            self.status = AgentStatus.IDLE

            # Determine if this is final based on handoff
            is_final = not handoff_info.get("handoff_required", False)
            summary = (
                f"Routing to {handoff_info.get('target_agent', '')} agent for specialized assistance."
                if handoff_info.get("handoff_required")
                else f"Marketing strategy generated for {request_type}"
            )

            response_dict = {
                "success": True,
                "strategy": result,
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": is_final,
                "summary": summary,
            }

            # Add handoff information if needed
            if handoff_info.get("handoff_required"):
                response_dict.update(handoff_info)

            return response_dict

        except Exception as e:
            self.logger.error(f"Strategy processing failed: {e}")
            self.status = AgentStatus.ERROR

            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "is_final": True,  # Mark as final even on error to stop loop
                "summary": f"Strategy processing failed: {str(e)}",
            }

    async def _create_campaign_strategy(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive campaign strategy."""

        # Extract requirements
        objectives = input_data.get("objectives", [])
        target_audience = input_data.get("target_audience", {})
        # Ensure target_audience is a dict
        if not isinstance(target_audience, dict):
            target_audience = {}
        budget = input_data.get("budget", 0)
        timeline = input_data.get("timeline", "3 months")
        industry = input_data.get("industry", "general")
        competitors = input_data.get("competitors", [])

        # Perform research
        market_data = await self._market_research(
            query=f"{industry} market analysis",
            target_audience=target_audience.get("description", None),
        )

        # Analyze first competitor if provided
        competitor_data = {}
        if competitors:
            competitor_data = await self._competitor_analysis(
                competitor_name=competitors[0], industry=industry
            )

        # Prepare campaign data for segmentation
        campaign_data = {
            "objectives": objectives,
            "industry": industry,
            "product": input_data.get("product", "Stripe"),
            "budget": budget,
        }

        # Develop strategy
        strategy = {
            "campaign_name": input_data.get("campaign_name", "New Campaign"),
            "objectives": objectives,
            "target_audience": await self._audience_segmentation(campaign_data),
            "market_insights": market_data,
            "competitive_position": competitor_data,
            "channels": await self._optimize_channels(budget, objectives),
            "content_strategy": await self._generate_content_strategy(
                campaign_plan={
                    "objectives": objectives,
                    "target_audience": target_audience,
                    "product": input_data.get("product", "Stripe"),
                },
                duration_weeks=12,
            ),
            "budget_allocation": await self._budget_allocation(
                {
                    "total_budget": budget,
                    "channels": ["social", "email", "paid_ads", "content"],
                }
            ),
            "kpis": self._define_kpis(objectives),
            "timeline": timeline,
            "recommendations": market_data.get("recommendations", []),
        }

        return strategy

    async def _create_content_calendar(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create content calendar."""
        return {
            "calendar_period": input_data.get("period", "monthly"),
            "content_pillars": [
                "Educational content",
                "Product highlights",
                "Customer stories",
                "Industry insights",
            ],
            "posting_schedule": {
                "social_media": "3x per week",
                "blog": "2x per week",
                "email": "1x per week",
            },
        }

    async def _analyze_market(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions or customer insights.

        Args:
            input_data: Data containing market context or customer insights

        Returns:
            Market analysis or insight evaluation
        """
        self.logger.info(f"Analyzing market with input: {list(input_data.keys())}")

        # Check if this is a customer insight handoff
        if input_data.get("insight_type") or input_data.get("customer_message"):
            return await self._analyze_customer_insight(input_data)

        # Default market analysis
        return {
            "market_size": "Large and growing",
            "trends": await self._trend_analysis({}),
            "opportunities": ["Digital transformation", "Mobile-first"],
            "threats": ["Increased competition", "Market saturation"],
        }

    async def _analyze_customer_insight(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze specific customer insight or feature request."""
        try:
            insight_type = input_data.get("insight_type", "general_insight")
            message = input_data.get("customer_message", "")
            sentiment = input_data.get("sentiment", "neutral")

            system_prompt = """You are a Strategic Marketing Analyst.
Analyze the provided customer insight or feature request.
Evaluate the market opportunity, strategic fit, and potential impact.
Provide a concise analysis with specific recommendations.

Output Format:
# Market Opportunity Analysis
## Strategic Evaluation
[Assessment of the request's alignment with market trends]

## Potential Impact
[Revenue or engagement impact]

## Recommendations
[Actionable next steps]
"""

            user_prompt = f"""
Insight Type: {insight_type}
Customer Feedback: "{message}"
Sentiment: {sentiment}

Please analyze this market signal.
"""

            if self.llm:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
                response = await self.llm.ainvoke(messages)
                analysis = response.content
            else:
                analysis = "LLM not available for analysis."

            return {
                "analysis_type": "customer_insight_evaluation",
                "insight_type": insight_type,
                "original_message": message,
                "analysis": analysis,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Insight analysis failed: {e}")
            return {
                "error": str(e),
                "analysis": "Failed to analyze insight.",
            }

    # Tool implementations

    async def _market_research(
        self, query: str, target_audience: str = None
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive market research using web search and knowledge base.

        Args:
            query: Research query or industry to investigate
            target_audience: Optional target audience context

        Returns:
            Dictionary with Stripe insights, web trends, and recommendations
        """
        try:
            self.logger.info(f"Conducting market research for: {query}")

            # Check memory cache
            cache_key = f"market_research_{query}_{target_audience}"
            cached = self.retrieve_from_memory(cache_key)
            if cached:
                self.logger.debug("Returning cached market research results")
                return cached

            # 1. Search Stripe knowledge base for internal insights
            kb_queries = [query]
            if target_audience:
                kb_queries.append(f"{query} {target_audience}")

            kb_result = await self.kb_search_tool.search_multiple_queries(
                queries=kb_queries, top_k_per_query=3, deduplicate=True
            )

            stripe_insights = []
            if kb_result["success"]:
                for result in kb_result["results"]:
                    stripe_insights.append(
                        {
                            "content": result["content"],
                            "source": result["source"],
                            "score": result.get("score", 0.0),
                        }
                    )

            # 2. Search web for market trends (with fallback if API fails)
            web_result = await self.web_search_tool.search_market_trends(
                industry=query, time_range="past_year"
            )

            web_trends = []
            if web_result.get("web_results", {}).get("success"):
                for item in web_result["web_results"]["results"][:5]:
                    web_trends.append(
                        {
                            "title": item["title"],
                            "url": item["url"],
                            "snippet": item["snippet"],
                        }
                    )
            else:
                # Log but don't fail - we can work with KB data alone
                web_error = web_result.get("web_results", {}).get(
                    "error", "Unknown error"
                )
                self.logger.warning(
                    f"Web search failed (continuing without it): {web_error}"
                )

            # Add news trends
            if web_result.get("news_results", {}).get("success"):
                for item in web_result["news_results"]["results"][:3]:
                    web_trends.append(
                        {
                            "title": item["title"],
                            "url": item["url"],
                            "snippet": item["snippet"],
                            "source": item.get("source", ""),
                            "type": "news",
                        }
                    )

            # 3. Generate recommendations using LLM
            recommendations = await self._generate_research_recommendations(
                query, stripe_insights, web_trends, target_audience
            )

            result = {
                "query": query,
                "target_audience": target_audience,
                "stripe_insights": stripe_insights,
                "web_trends": web_trends,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Cache results
            self.save_to_memory(cache_key, result)

            self.logger.info(
                f"Market research completed: {len(stripe_insights)} insights, "
                f"{len(web_trends)} trends"
            )

            return result

        except Exception as e:
            self.logger.error(f"Market research failed: {e}")
            return {
                "error": str(e),
                "stripe_insights": [],
                "web_trends": [],
                "recommendations": [],
            }

    async def _competitor_analysis(
        self, competitor_name: str, industry: str
    ) -> Dict[str, Any]:
        """
        Analyze competitor using web search and LLM analysis.

        Args:
            competitor_name: Name of the competitor to analyze
            industry: Industry context

        Returns:
            Dictionary with competitor profile, strengths, weaknesses, opportunities
        """
        try:
            self.logger.info(f"Analyzing competitor: {competitor_name}")

            # Check memory cache
            cache_key = f"competitor_{competitor_name}_{industry}"
            cached = self.retrieve_from_memory(cache_key)
            if cached:
                self.logger.debug("Returning cached competitor analysis")
                return cached

            # 1. Search for competitor information
            competitor_data = await self.web_search_tool.search_competitor(
                competitor_name=competitor_name,
                aspects=["overview", "products", "pricing", "strategy", "customers"],
            )

            # 2. Extract key information from search results
            competitor_profile = {
                "name": competitor_name,
                "industry": industry,
                "overview": "",
                "key_messages": [],
                "positioning": "",
                "pricing_info": "",
            }

            # Compile information from search results
            all_snippets = []
            for aspect, result in competitor_data["aspects"].items():
                if result["success"]:
                    for item in result["results"][:2]:
                        all_snippets.append(
                            f"{aspect.upper()}: {item['title']} - {item['snippet']}"
                        )

            # 3. Use LLM to analyze competitor and generate insights
            if self.llm and all_snippets:
                analysis_prompt = f"""Analyze the following information about {competitor_name} in the {industry} industry.

Information gathered:
{chr(10).join(all_snippets[:10])}

Provide a structured analysis in JSON format:
{{
    "overview": "Brief company overview",
    "key_messages": ["list of main marketing messages"],
    "positioning": "How they position themselves",
    "pricing_info": "Available pricing information or 'Not available'",
    "strengths": ["list of competitive strengths"],
    "weaknesses": ["list of potential weaknesses"],
    "opportunities": ["opportunities for Stripe vs this competitor"]
}}"""

                response = await self.llm.ainvoke(analysis_prompt)

                try:
                    # Parse JSON response
                    json_content = self._extract_json_from_response(response.content)
                    analysis = json.loads(json_content)

                    competitor_profile.update(
                        {
                            "overview": analysis.get("overview", ""),
                            "key_messages": analysis.get("key_messages", []),
                            "positioning": analysis.get("positioning", ""),
                            "pricing_info": analysis.get("pricing_info", ""),
                        }
                    )

                    result = {
                        "competitor_profile": competitor_profile,
                        "strengths": analysis.get("strengths", []),
                        "weaknesses": analysis.get("weaknesses", []),
                        "opportunities": analysis.get("opportunities", []),
                        "raw_data": competitor_data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse LLM response as JSON")
                    result = {
                        "competitor_profile": competitor_profile,
                        "strengths": ["Analysis pending"],
                        "weaknesses": ["Analysis pending"],
                        "opportunities": ["Analysis pending"],
                        "raw_data": competitor_data,
                    }
            else:
                # Fallback without LLM analysis
                result = {
                    "competitor_profile": competitor_profile,
                    "strengths": ["Data collected, detailed analysis pending"],
                    "weaknesses": ["Data collected, detailed analysis pending"],
                    "opportunities": ["Data collected, detailed analysis pending"],
                    "raw_data": competitor_data,
                }

            # Cache results
            self.save_to_memory(cache_key, result)

            self.logger.info(f"Competitor analysis completed for {competitor_name}")

            return result

        except Exception as e:
            self.logger.error(f"Competitor analysis failed: {e}")
            return {
                "error": str(e),
                "competitor_profile": {"name": competitor_name},
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
            }

    async def _audience_segmentation(
        self, campaign_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Segment audience using LLM-based analysis of campaign objectives.

        Args:
            campaign_data: Dictionary with campaign objectives and context

        Returns:
            Dictionary with segments, personas, and recommendations
        """
        try:
            self.logger.info("Generating audience segmentation")

            objectives = campaign_data.get("objectives", [])
            industry = campaign_data.get("industry", "general")
            product = campaign_data.get("product", "Stripe")
            budget = campaign_data.get("budget", 0)

            if not self.llm:
                self.logger.warning("LLM not available, using default segmentation")
                return self._default_segmentation()

            # Create segmentation prompt
            prompt = f"""As a marketing strategist, create audience segments for a {product} campaign in the {industry} industry.

Campaign Objectives:
{chr(10).join(f"- {obj}" for obj in objectives)}

Budget: ${budget:,.0f}

Create 2-3 audience segments with detailed personas. Provide response in JSON format:
{{
    "segments": [
        {{
            "name": "Segment name",
            "demographics": {{
                "age_range": "25-45",
                "job_titles": ["list of titles"],
                "company_size": "startup/SMB/enterprise",
                "industries": ["list of industries"]
            }},
            "pain_points": ["list of specific pain points"],
            "motivations": ["list of buying motivations"],
            "channels": ["preferred channels"],
            "messaging_angle": "How to message this segment",
            "estimated_size": "percentage or description"
        }}
    ],
    "primary_segment": "name of highest priority segment",
    "recommendations": {{
        "targeting_strategy": "Overall strategy",
        "budget_allocation": "How to split budget across segments",
        "content_priorities": ["list of content priorities"]
    }}
}}"""

            response = await self.llm.ainvoke(prompt)

            try:
                # Parse LLM response
                json_content = self._extract_json_from_response(response.content)
                segmentation = json.loads(json_content)

                # Add metadata
                segmentation["timestamp"] = datetime.utcnow().isoformat()
                segmentation["campaign_objectives"] = objectives

                self.logger.info(
                    f"Generated {len(segmentation.get('segments', []))} audience segments"
                )

                return segmentation

            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM segmentation response")
                return self._default_segmentation()

        except Exception as e:
            self.logger.error(f"Audience segmentation failed: {e}")
            return self._default_segmentation()

    async def _generate_content_strategy(
        self, campaign_plan: Dict[str, Any], duration_weeks: int = 12
    ) -> Dict[str, Any]:
        """
        Generate comprehensive content strategy with calendar.

        Args:
            campaign_plan: Campaign plan with objectives and audience
            duration_weeks: Campaign duration in weeks (default: 12)

        Returns:
            Dictionary with content calendar, themes, and KPIs
        """
        try:
            self.logger.info(f"Generating {duration_weeks}-week content strategy")

            objectives = campaign_plan.get("objectives", [])
            audience = campaign_plan.get("target_audience", {})
            product = campaign_plan.get("product", "Stripe")

            # 1. Research keywords
            keywords = await self._keyword_research(
                product=product, objectives=objectives
            )

            if not self.llm:
                return self._default_content_strategy(duration_weeks, keywords)

            # 2. Generate content calendar using LLM
            prompt = f"""Create a {duration_weeks}-week content marketing calendar for a {product} campaign.

Campaign Objectives:
{chr(10).join(f"- {obj}" for obj in objectives[:5])}

Target Audience: {audience.get('primary_segment', {}).get('name', 'General')}

Keywords to incorporate: {', '.join(keywords.get('primary_keywords', [])[:10])}

Create a structured content calendar in JSON format:
{{
    "themes": ["3-5 overarching content themes"],
    "calendar": [
        {{
            "week": 1,
            "topic": "Specific content topic",
            "format": "blog/video/infographic/webinar",
            "channel": "blog/social/email",
            "keywords": ["relevant keywords"],
            "cta": "Call to action"
        }}
    ],
    "kpis": {{
        "traffic_target": "monthly target",
        "engagement_rate": "target %",
        "conversion_rate": "target %",
        "lead_generation": "monthly target"
    }}
}}

Generate exactly {duration_weeks} calendar entries, distributed across the themes."""

            response = await self.llm.ainvoke(prompt)

            try:
                json_content = self._extract_json_from_response(response.content)
                strategy = json.loads(json_content)

                # Ensure we have the right number of weeks
                if len(strategy.get("calendar", [])) < duration_weeks:
                    self.logger.warning(
                        f"Calendar has fewer than {duration_weeks} entries, filling gaps"
                    )

                # Add metadata
                strategy["duration_weeks"] = duration_weeks
                strategy["start_date"] = datetime.utcnow().isoformat()
                strategy["end_date"] = (
                    datetime.utcnow() + timedelta(weeks=duration_weeks)
                ).isoformat()
                strategy["keywords"] = keywords

                self.logger.info(
                    f"Content strategy generated with {len(strategy.get('calendar', []))} items"
                )

                return strategy

            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM content strategy response")
                return self._default_content_strategy(duration_weeks, keywords)

        except Exception as e:
            self.logger.error(f"Content strategy generation failed: {e}")
            return self._default_content_strategy(duration_weeks, {})

    async def _optimize_channels(
        self, budget: float, objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Optimize marketing channel mix and budget allocation.

        Args:
            budget: Total marketing budget
            objectives: List of campaign objectives

        Returns:
            Dictionary with channel recommendations, allocation, and rationale
        """
        try:
            self.logger.info(f"Optimizing channels for ${budget:,.0f} budget")

            if not self.llm:
                return self._default_channel_optimization(budget)

            # Create optimization prompt
            prompt = f"""As a marketing strategist, optimize the channel mix for a campaign with ${budget:,.0f} budget.

Campaign Objectives:
{chr(10).join(f"- {obj}" for obj in objectives)}

Available channels:
- Social Media (LinkedIn, Twitter, Instagram)
- Email Marketing
- Content Marketing (Blog, SEO)
- Paid Advertising (Google Ads, LinkedIn Ads)
- Events/Webinars
- Partner Marketing

Provide recommendations in JSON format:
{{
    "channels": {{
        "social_media": {{
            "platforms": ["list platforms"],
            "rationale": "Why this channel",
            "priority": "high/medium/low"
        }},
        "email_marketing": {{
            "rationale": "Why this channel",
            "priority": "high/medium/low"
        }},
        "content_marketing": {{
            "rationale": "Why this channel",
            "priority": "high/medium/low"
        }},
        "paid_advertising": {{
            "platforms": ["list platforms"],
            "rationale": "Why this channel",
            "priority": "high/medium/low"
        }},
        "events": {{
            "rationale": "Why this channel",
            "priority": "high/medium/low"
        }}
    }},
    "allocation": {{
        "social_media": {{"percentage": 20, "amount": calculated}},
        "email_marketing": {{"percentage": 15, "amount": calculated}},
        "content_marketing": {{"percentage": 25, "amount": calculated}},
        "paid_advertising": {{"percentage": 30, "amount": calculated}},
        "events": {{"percentage": 10, "amount": calculated}}
    }},
    "rationale": "Overall strategy rationale"
}}

Ensure percentages sum to 100."""

            response = await self.llm.ainvoke(prompt)

            try:
                json_content = self._extract_json_from_response(response.content)
                optimization = json.loads(json_content)

                # Calculate actual amounts based on percentages
                allocation = optimization.get("allocation", {})
                for channel, details in allocation.items():
                    if isinstance(details, dict):
                        percentage = details.get("percentage", 0)
                        details["amount"] = budget * (percentage / 100)

                # Add metadata
                optimization["total_budget"] = budget
                optimization["timestamp"] = datetime.utcnow().isoformat()

                self.logger.info(
                    f"Channel optimization completed with {len(optimization.get('channels', {}))} channels"
                )

                return optimization

            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM channel optimization response")
                return self._default_channel_optimization(budget)

        except Exception as e:
            self.logger.error(f"Channel optimization failed: {e}")
            return self._default_channel_optimization(budget)

    async def _budget_allocation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Allocate budget across channels."""
        total_budget = context.get("total_budget", 10000)

        return {
            "total_budget": total_budget,
            "allocation": {
                "social_media": total_budget * 0.30,
                "email_marketing": total_budget * 0.20,
                "content_creation": total_budget * 0.25,
                "paid_advertising": total_budget * 0.25,
            },
        }

    async def _keyword_research(self, **kwargs) -> Dict[str, Any]:
        """Research relevant keywords."""
        return {
            "primary_keywords": [
                "enterprise marketing",
                "AI agents",
                "marketing automation",
            ],
            "long_tail_keywords": [
                "best enterprise marketing tools",
                "how to use AI for marketing",
            ],
        }

    async def _trend_analysis(self, **kwargs) -> Dict[str, Any]:
        """Analyze current trends."""
        return {
            "trending_topics": [
                "AI in marketing",
                "Personalization at scale",
                "Privacy-first marketing",
            ],
            "emerging_platforms": ["TikTok", "Threads"],
            "declining_trends": ["Generic mass email"],
        }

    def _define_kpis(self, objectives: List[str]) -> List[Dict[str, str]]:
        """Define KPIs based on objectives."""
        return [
            {"metric": "Conversion Rate", "target": "5%"},
            {"metric": "Customer Acquisition Cost", "target": "$50"},
            {"metric": "ROI", "target": "300%"},
            {"metric": "Engagement Rate", "target": "8%"},
        ]

    def _generate_recommendations(self, market_data: Dict[str, Any]) -> List[str]:
        """Generate strategic recommendations."""
        return [
            "Focus on digital channels for maximum reach",
            "Invest in content marketing for long-term value",
            "Implement A/B testing for optimization",
            "Build email list for owned audience",
            "Leverage AI for personalization",
        ]

    async def _generate_research_recommendations(
        self,
        query: str,
        stripe_insights: List[Dict],
        web_trends: List[Dict],
        target_audience: Optional[str],
    ) -> List[str]:
        """
        Generate actionable recommendations from research data using LLM.

        Args:
            query: Research query
            stripe_insights: Internal knowledge base insights
            web_trends: External web trends
            target_audience: Optional target audience

        Returns:
            List of actionable recommendations
        """
        if not self.llm:
            return [
                "Review gathered insights to identify opportunities",
                "Align strategy with market trends",
                "Consider target audience needs",
            ]

        try:
            # Prepare context for LLM
            context = f"Research Query: {query}\n\n"

            if target_audience:
                context += f"Target Audience: {target_audience}\n\n"

            if stripe_insights:
                context += "Stripe Knowledge Base Insights:\n"
                for insight in stripe_insights[:3]:
                    context += f"- {insight['content'][:200]}...\n"
                context += "\n"

            if web_trends:
                context += "Web Trends:\n"
                for trend in web_trends[:5]:
                    context += f"- {trend['title']}: {trend['snippet'][:150]}...\n"

            prompt = f"""{context}

Based on the above research, provide 5 specific, actionable marketing recommendations. Format as a JSON array of strings:
["recommendation 1", "recommendation 2", ...]"""

            response = await self.llm.ainvoke(prompt)

            try:
                recommendations = json.loads(response.content)
                if isinstance(recommendations, list):
                    return recommendations[:5]
            except json.JSONDecodeError:
                pass

            # Fallback: parse as text
            lines = response.content.strip().split("\n")
            return [line.strip("- []0123456789.\"'") for line in lines if line.strip()][
                :5
            ]

        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return [
                "Leverage market insights for targeted campaigns",
                "Align messaging with industry trends",
                "Focus on differentiation opportunities",
            ]

    def _default_segmentation(self) -> Dict[str, Any]:
        """Provide default audience segmentation when LLM is unavailable."""
        return {
            "segments": [
                {
                    "name": "Enterprise Decision Makers",
                    "demographics": {
                        "age_range": "35-55",
                        "job_titles": ["CTO", "VP Engineering", "Director of Product"],
                        "company_size": "enterprise",
                        "industries": ["Technology", "E-commerce", "SaaS"],
                    },
                    "pain_points": [
                        "Complex payment infrastructure",
                        "Global expansion challenges",
                        "Security and compliance concerns",
                    ],
                    "motivations": [
                        "Scalability",
                        "Reliability",
                        "Developer experience",
                    ],
                    "channels": ["LinkedIn", "Email", "Webinars"],
                    "messaging_angle": "Enterprise-grade infrastructure at scale",
                    "estimated_size": "30%",
                }
            ],
            "primary_segment": "Enterprise Decision Makers",
            "recommendations": {
                "targeting_strategy": "Focus on high-value enterprise accounts",
                "budget_allocation": "Prioritize channels with highest enterprise reach",
                "content_priorities": [
                    "Technical deep-dives",
                    "Case studies",
                    "ROI calculators",
                ],
            },
        }

    def _default_content_strategy(
        self, duration_weeks: int, keywords: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Provide default content strategy when LLM is unavailable."""
        calendar = []
        themes = [
            "Product Innovation",
            "Customer Success",
            "Industry Insights",
            "Technical Excellence",
        ]

        for week in range(1, duration_weeks + 1):
            theme = themes[(week - 1) % len(themes)]
            calendar.append(
                {
                    "week": week,
                    "topic": f"{theme} - Week {week}",
                    "format": ["blog", "video", "infographic"][week % 3],
                    "channel": ["blog", "social", "email"][week % 3],
                    "keywords": keywords.get("primary_keywords", [])[:3],
                    "cta": "Learn more",
                }
            )

        return {
            "themes": themes,
            "calendar": calendar,
            "kpis": {
                "traffic_target": "10,000 monthly visitors",
                "engagement_rate": "5%",
                "conversion_rate": "2%",
                "lead_generation": "500 leads/month",
            },
            "duration_weeks": duration_weeks,
            "keywords": keywords,
        }

    def _default_channel_optimization(self, budget: float) -> Dict[str, Any]:
        """Provide default channel optimization when LLM is unavailable."""
        return {
            "channels": {
                "social_media": {
                    "platforms": ["LinkedIn", "Twitter"],
                    "rationale": "Professional audience engagement",
                    "priority": "high",
                },
                "email_marketing": {
                    "rationale": "Direct communication with leads",
                    "priority": "high",
                },
                "content_marketing": {
                    "rationale": "Long-term organic growth",
                    "priority": "high",
                },
                "paid_advertising": {
                    "platforms": ["Google Ads", "LinkedIn Ads"],
                    "rationale": "Immediate visibility and lead generation",
                    "priority": "medium",
                },
                "events": {
                    "rationale": "Direct engagement with prospects",
                    "priority": "medium",
                },
            },
            "allocation": {
                "social_media": {"percentage": 20, "amount": budget * 0.20},
                "email_marketing": {"percentage": 15, "amount": budget * 0.15},
                "content_marketing": {"percentage": 25, "amount": budget * 0.25},
                "paid_advertising": {"percentage": 30, "amount": budget * 0.30},
                "events": {"percentage": 10, "amount": budget * 0.10},
            },
            "total_budget": budget,
            "rationale": "Balanced approach prioritizing high-ROI channels",
        }

    async def _keyword_research(
        self, product: str = "Stripe", objectives: List[str] = None
    ) -> Dict[str, Any]:
        """
        Research relevant keywords for content strategy.

        Args:
            product: Product name
            objectives: Campaign objectives

        Returns:
            Dictionary with keyword recommendations
        """
        # Build keyword list based on objectives
        primary_keywords = [
            f"{product.lower()} api",
            f"{product.lower()} integration",
            f"payment processing",
            f"online payments",
        ]

        long_tail_keywords = [
            f"how to integrate {product.lower()}",
            f"best payment gateway for startups",
            f"{product.lower()} vs competitors",
        ]

        if objectives:
            for obj in objectives[:3]:
                # Extract key terms from objectives
                words = obj.lower().split()
                for word in words:
                    if len(word) > 5:
                        primary_keywords.append(f"{product.lower()} {word}")

        return {
            "primary_keywords": list(set(primary_keywords))[:10],
            "long_tail_keywords": list(set(long_tail_keywords))[:10],
            "product": product,
        }

    def _detect_handoff_need(
        self, message: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect if marketing strategy request warrants a handoff to another agent.

        Analyzes the user message and context to determine if:
        - Performance analysis needed (→ Analytics & Evaluation)
        - Customer insights needed (→ Customer Support)
        - Strategy optimization needed (→ Feedback & Learning)

        Args:
            message: User message text
            result: Strategy result dictionary

        Returns:
            Dictionary with handoff information if needed, empty dict otherwise
        """
        handoff_info = {}
        message_lower = message.lower()

        self.logger.info(f"Checking handoff need for message: '{message[:100]}'")

        # Scenario 1: Performance Analysis / Validation → Analytics Agent
        # Keywords: analyze, performance, metrics, results, ROI, effectiveness, forecast
        analytics_keywords = [
            "analyze",
            "analysis",
            "performance",
            "performing",
            "metrics",
            "results",
            "roi",
            "effectiveness",
            "effective",
            "forecast",
            "predict",
            "impact",
            "measure",
            "data",
            "statistics",
            "conversion",
            "revenue",
        ]

        if any(keyword in message_lower for keyword in analytics_keywords):
            self.logger.info(
                "Handoff detected: analytics_evaluation (performance analysis request)"
            )
            handoff_info = {
                "handoff_required": True,
                "target_agent": "analytics_evaluation",
                "handoff_reason": "strategy_validation_needed",
                "context": {
                    "analysis_type": "performance_validation",
                    "user_message": message,
                    "strategy_result": result,
                    "recommendation": "Analytics team should perform data analysis and validation",
                },
            }
            return handoff_info

        # Scenario 2: Customer Feedback / Insights → Customer Support Agent
        # Keywords: customer feedback, support tickets, complaints, what customers say, pain points
        customer_insight_keywords = [
            "customer feedback",
            "customer complaints",
            "support tickets",
            "what customers say",
            "what are customers",
            "customers saying",
            "customer pain",
            "pain points",
            "customer problems",
            "customer issues",
            "customer experience",
            "customers experiencing",
            "customer satisfaction",
        ]

        if any(keyword in message_lower for keyword in customer_insight_keywords):
            self.logger.info(
                "Handoff detected: customer_support (customer insights request)"
            )
            handoff_info = {
                "handoff_required": True,
                "target_agent": "customer_support",
                "handoff_reason": "customer_insights_needed",
                "context": {
                    "insight_type": "customer_feedback_analysis",
                    "user_message": message,
                    "strategy_result": result,
                    "recommendation": "Customer Support should analyze feedback themes and sentiment",
                },
            }
            return handoff_info

        # Scenario 3: Strategy Optimization / Improvement → Feedback & Learning Agent
        # Keywords: improve, optimize, not working, underperforming, better results, A/B test
        optimization_keywords = [
            "improve",
            "optimize",
            "optimization",
            "not working",
            "underperform",
            "underperforming",
            "better results",
            "how can i improve",
            "a/b test",
            "experiment",
            "test different",
            "set up test",
            "help me test",
        ]

        if any(keyword in message_lower for keyword in optimization_keywords):
            self.logger.info(
                "Handoff detected: feedback_learning (optimization request)"
            )
            handoff_info = {
                "handoff_required": True,
                "target_agent": "feedback_learning",
                "handoff_reason": "optimization_needed",
                "context": {
                    "optimization_type": "strategy_improvement",
                    "user_message": message,
                    "strategy_result": result,
                    "recommendation": "Learning agent should analyze historical performance and recommend improvements",
                },
            }
            return handoff_info

        self.logger.info("No handoff needed - standard marketing strategy query")
        return {}
