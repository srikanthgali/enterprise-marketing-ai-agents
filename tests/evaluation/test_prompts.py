"""
Prompt Evaluation Framework

Evaluates agent prompts against test cases to ensure:
- Expected behaviors are present
- Quality criteria are met
- Performance is consistent across versions
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from src.marketing_agents.core.prompt_manager import PromptManager
from src.marketing_agents.agents.marketing_strategy import MarketingStrategyAgent
from src.marketing_agents.agents.customer_support import CustomerSupportAgent
from src.marketing_agents.core.orchestrator import OrchestratorAgent


class PromptEvaluator:
    """
    Evaluates agent prompts against test cases.

    Features:
    - Behavior checking (citations, KPIs, structure)
    - Quality scoring (clarity, completeness, actionability)
    - Version comparison
    - Detailed reporting
    """

    def __init__(self, test_cases_path: str = "data/evaluation/prompt_test_cases.json"):
        """
        Initialize the evaluator.

        Args:
            test_cases_path: Path to JSON file with test cases
        """
        self.test_cases_path = Path(test_cases_path)
        self.test_cases = self._load_test_cases()
        self.prompt_manager = PromptManager()

        # Agent registry for instantiation
        self.agent_classes = {
            "marketing_strategy": MarketingStrategyAgent,
            "customer_support": CustomerSupportAgent,
            "orchestrator": OrchestratorAgent,
        }

    def _load_test_cases(self) -> Dict[str, List[Dict]]:
        """Load test cases from JSON file."""
        if not self.test_cases_path.exists():
            return {}

        with open(self.test_cases_path, "r") as f:
            return json.load(f)

    async def evaluate_prompt(
        self,
        agent_id: str,
        test_cases: Optional[List[Dict]] = None,
        prompt_version: str = "latest",
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt using test cases.

        Args:
            agent_id: Agent identifier (e.g., 'marketing_strategy')
            test_cases: List of test cases (uses defaults if None)
            prompt_version: Prompt version to evaluate

        Returns:
            Evaluation results with scores and details
        """
        # Use default test cases if not provided
        if test_cases is None:
            test_cases = self.test_cases.get(agent_id, [])

        if not test_cases:
            return {
                "error": f"No test cases found for agent: {agent_id}",
                "agent_id": agent_id,
                "prompt_version": prompt_version,
            }

        # Initialize agent with specific prompt version
        agent = await self._create_agent(agent_id, prompt_version)

        if not agent:
            return {
                "error": f"Could not create agent: {agent_id}",
                "agent_id": agent_id,
                "prompt_version": prompt_version,
            }

        # Run test cases
        results = {
            "agent_id": agent_id,
            "prompt_version": prompt_version,
            "timestamp": datetime.utcnow().isoformat(),
            "test_cases": [],
            "summary": {},
        }

        total_score = 0.0
        passed_count = 0

        for test_case in test_cases:
            # Execute agent with test input
            try:
                output = await agent.process(test_case["input"])
            except Exception as e:
                output = {"error": str(e), "success": False}

            # Score the output
            score_result = self._score_output(
                output,
                test_case.get("expected_behaviors", []),
                test_case.get("quality_criteria", {}),
            )

            # Add test case details
            test_result = {
                "id": test_case["id"],
                "description": test_case.get("description", ""),
                "score": score_result["total_score"],
                "passed": score_result["total_score"] >= 0.8,
                "behavior_scores": score_result["behavior_scores"],
                "quality_scores": score_result["quality_scores"],
                "output": output,
            }

            results["test_cases"].append(test_result)
            total_score += score_result["total_score"]
            if test_result["passed"]:
                passed_count += 1

        # Calculate summary statistics
        num_tests = len(test_cases)
        results["summary"] = {
            "avg_score": total_score / num_tests if num_tests > 0 else 0.0,
            "pass_rate": passed_count / num_tests if num_tests > 0 else 0.0,
            "total_tests": num_tests,
            "passed_tests": passed_count,
            "failed_tests": num_tests - passed_count,
        }

        return results

    async def _create_agent(self, agent_id: str, prompt_version: str):
        """Create an agent instance with specific prompt version."""
        if agent_id not in self.agent_classes:
            return None

        agent_class = self.agent_classes[agent_id]

        config = {"model": {"name": "gpt-4o-mini", "temperature": 0.7}}

        # Create agent
        agent = agent_class(config=config, prompt_manager=self.prompt_manager)

        # Load specific version if not latest
        if prompt_version != "latest":
            agent.reload_prompt(version=prompt_version)

        return agent

    def _score_output(
        self,
        output: Dict[str, Any],
        expected_behaviors: List[str],
        quality_criteria: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Score agent output against expected behaviors and quality criteria.

        Args:
            output: Agent output to evaluate
            expected_behaviors: List of expected behaviors
            quality_criteria: Dict of quality criteria with thresholds

        Returns:
            Scoring results with behavior and quality scores
        """
        behavior_scores = {}
        quality_scores = {}

        # Check behaviors
        for behavior in expected_behaviors:
            behavior_scores[behavior] = self._check_behavior(output, behavior)

        # Check quality criteria
        for criterion, threshold in quality_criteria.items():
            quality_scores[criterion] = self._check_quality(
                output, criterion, threshold
            )

        # Calculate total score
        all_scores = list(behavior_scores.values()) + list(quality_scores.values())
        total_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        return {
            "behavior_scores": behavior_scores,
            "quality_scores": quality_scores,
            "total_score": total_score,
        }

    def _check_behavior(self, output: Dict[str, Any], behavior: str) -> float:
        """
        Check if specific behavior is present in output.

        Args:
            output: Agent output
            behavior: Behavior to check

        Returns:
            Score: 1.0 if present, 0.0 if absent
        """
        output_str = json.dumps(output, default=str).lower()

        behavior_checks = {
            "cites_kb": lambda: self._check_citations(output_str),
            "provides_kpis": lambda: self._check_kpis(output_str),
            "includes_budget_breakdown": lambda: self._check_budget_breakdown(
                output_str
            ),
            "defines_target_audience": lambda: self._check_target_audience(output_str),
            "specifies_channels": lambda: self._check_channels(output_str),
            "includes_timeline": lambda: self._check_timeline(output_str),
            "risk_assessment": lambda: self._check_risk_assessment(output_str),
            "addresses_constraints": lambda: self._check_constraints(output_str),
            "provides_troubleshooting_steps": lambda: self._check_troubleshooting(
                output_str
            ),
            "empathetic_tone": lambda: self._check_empathy(output_str),
            "offers_escalation": lambda: self._check_escalation(output_str),
            "includes_examples": lambda: self._check_examples(output_str),
            "explains_clearly": lambda: self._check_clarity_markers(output_str),
            "offers_resolution": lambda: self._check_resolution(output_str),
            "identifies_correct_agent": lambda: self._check_agent_identification(
                output_str
            ),
            "provides_routing_rationale": lambda: self._check_routing_rationale(
                output_str
            ),
            "checks_prerequisites": lambda: self._check_prerequisites(output_str),
        }

        check_func = behavior_checks.get(behavior)
        if check_func:
            return 1.0 if check_func() else 0.0

        return 0.0

    def _check_citations(self, output_str: str) -> bool:
        """Check for knowledge base citations."""
        patterns = [
            r"\[source:",
            r"\(source:",
            r"according to",
            r"referenced from",
            r"documentation shows",
        ]
        return any(
            re.search(pattern, output_str, re.IGNORECASE) for pattern in patterns
        )

    def _check_kpis(self, output_str: str) -> bool:
        """Check for KPI definitions."""
        kpi_indicators = [
            "kpi",
            "metric",
            "measure",
            "conversion rate",
            "roi",
            "ctr",
            "click-through",
            "engagement rate",
            "revenue",
            "target:",
            "goal:",
            "objective:",
        ]
        return sum(indicator in output_str for indicator in kpi_indicators) >= 2

    def _check_budget_breakdown(self, output_str: str) -> bool:
        """Check for budget breakdown with amounts/percentages."""
        has_budget = (
            "budget" in output_str or "cost" in output_str or "spend" in output_str
        )
        has_numbers = bool(re.search(r"\$[\d,]+|[\d]+%|[\d]+\s*percent", output_str))
        has_breakdown = bool(
            re.search(r"allocation|distribution|breakdown|split", output_str)
        )
        return has_budget and has_numbers and has_breakdown

    def _check_target_audience(self, output_str: str) -> bool:
        """Check for target audience definition."""
        audience_indicators = [
            "target audience",
            "persona",
            "demographic",
            "customer segment",
            "ideal customer",
            "buyer profile",
            "audience profile",
        ]
        return any(indicator in output_str for indicator in audience_indicators)

    def _check_channels(self, output_str: str) -> bool:
        """Check for marketing channels specification."""
        channels = [
            "linkedin",
            "twitter",
            "facebook",
            "email",
            "content marketing",
            "seo",
            "sem",
            "ppc",
            "social media",
            "blog",
            "webinar",
            "channel",
            "platform",
        ]
        return sum(channel in output_str for channel in channels) >= 2

    def _check_timeline(self, output_str: str) -> bool:
        """Check for timeline/schedule."""
        timeline_indicators = [
            "timeline",
            "schedule",
            "phase",
            "week",
            "month",
            "quarter",
            "milestone",
            "roadmap",
            "sprint",
        ]
        return any(indicator in output_str for indicator in timeline_indicators)

    def _check_risk_assessment(self, output_str: str) -> bool:
        """Check for risk assessment."""
        risk_indicators = [
            "risk",
            "challenge",
            "concern",
            "mitigation",
            "contingency",
            "potential issue",
            "limitation",
        ]
        return any(indicator in output_str for indicator in risk_indicators)

    def _check_constraints(self, output_str: str) -> bool:
        """Check for addressing constraints."""
        constraint_indicators = [
            "constraint",
            "limitation",
            "budget",
            "resource",
            "priority",
            "given the",
            "considering",
            "within",
        ]
        return sum(indicator in output_str for indicator in constraint_indicators) >= 2

    def _check_troubleshooting(self, output_str: str) -> bool:
        """Check for troubleshooting steps."""
        troubleshooting_indicators = [
            "step 1",
            "first,",
            "second,",
            "then",
            "next",
            "check",
            "verify",
            "ensure",
            "confirm",
        ]
        return (
            sum(indicator in output_str for indicator in troubleshooting_indicators)
            >= 3
        )

    def _check_empathy(self, output_str: str) -> bool:
        """Check for empathetic tone."""
        empathy_indicators = [
            "understand",
            "appreciate",
            "apologize",
            "sorry",
            "frustrated",
            "help you",
            "assist you",
            "here for you",
        ]
        return any(indicator in output_str for indicator in empathy_indicators)

    def _check_escalation(self, output_str: str) -> bool:
        """Check for escalation offer."""
        escalation_indicators = [
            "escalate",
            "senior",
            "specialist",
            "manager",
            "team member",
            "additional support",
        ]
        return any(indicator in output_str for indicator in escalation_indicators)

    def _check_examples(self, output_str: str) -> bool:
        """Check for examples."""
        example_indicators = [
            "example",
            "for instance",
            "such as",
            "like",
            "e.g.",
            "sample",
            "demo",
        ]
        return any(indicator in output_str for indicator in example_indicators)

    def _check_clarity_markers(self, output_str: str) -> bool:
        """Check for clear explanation markers."""
        clarity_indicators = [
            "specifically",
            "in other words",
            "this means",
            "to clarify",
            "simply put",
            "essentially",
        ]
        return any(indicator in output_str for indicator in clarity_indicators)

    def _check_resolution(self, output_str: str) -> bool:
        """Check for resolution offer."""
        resolution_indicators = [
            "resolve",
            "fix",
            "solution",
            "correct",
            "address",
            "will",
            "can help",
        ]
        return sum(indicator in output_str for indicator in resolution_indicators) >= 2

    def _check_agent_identification(self, output_str: str) -> bool:
        """Check for correct agent identification."""
        agent_indicators = [
            "marketing_strategy",
            "customer_support",
            "analytics",
            "feedback",
            "agent",
            "route to",
            "forward to",
        ]
        return any(indicator in output_str for indicator in agent_indicators)

    def _check_routing_rationale(self, output_str: str) -> bool:
        """Check for routing rationale."""
        rationale_indicators = [
            "because",
            "reason",
            "requires",
            "needs",
            "best suited",
            "specialized",
            "expertise",
        ]
        return any(indicator in output_str for indicator in rationale_indicators)

    def _check_prerequisites(self, output_str: str) -> bool:
        """Check for prerequisite checks."""
        prerequisite_indicators = [
            "prerequisite",
            "required",
            "need",
            "must have",
            "ensure",
            "before",
            "first",
        ]
        return any(indicator in output_str for indicator in prerequisite_indicators)

    def _check_quality(
        self, output: Dict[str, Any], criterion: str, threshold: float
    ) -> float:
        """
        Check quality criterion and compare against threshold.

        Args:
            output: Agent output
            criterion: Quality criterion to check
            threshold: Minimum acceptable score

        Returns:
            Score: 1.0 if meets threshold, proportional score otherwise
        """
        output_str = json.dumps(output, default=str)

        quality_checks = {
            "clarity": lambda: self._score_clarity(output_str),
            "completeness": lambda: self._score_completeness(output, output_str),
            "actionability": lambda: self._score_actionability(output_str),
            "accuracy": lambda: self._score_accuracy(output),
            "helpfulness": lambda: self._score_helpfulness(output_str),
            "professionalism": lambda: self._score_professionalism(output_str),
            "efficiency": lambda: self._score_efficiency(output),
        }

        check_func = quality_checks.get(criterion)
        if check_func:
            score = check_func()
            return 1.0 if score >= threshold else score / threshold

        return 0.0

    def _score_clarity(self, output_str: str) -> float:
        """Score clarity of output (0.0-1.0)."""
        # Check for clear structure
        has_sections = bool(re.search(r"\n\n|##|###|\*\*", output_str))

        # Check for clear language (not too complex)
        word_count = len(output_str.split())
        avg_word_length = sum(len(word) for word in output_str.split()) / max(
            word_count, 1
        )
        clarity_from_simplicity = 1.0 - min((avg_word_length - 5) / 10, 0.5)

        # Check for bullets/lists
        has_lists = bool(re.search(r"[•\-*]\s|^\d+\.", output_str, re.MULTILINE))

        score = (
            (0.4 if has_sections else 0.0)
            + (0.3 * clarity_from_simplicity)
            + (0.3 if has_lists else 0.0)
        )

        return min(score, 1.0)

    def _score_completeness(self, output: Dict[str, Any], output_str: str) -> float:
        """Score completeness of output (0.0-1.0)."""
        # Check for success indicator
        has_success = output.get("success", True)

        # Check for substantial content
        has_content = len(output_str) > 200

        # Check for multiple components
        component_count = sum(
            [
                bool(re.search(r"overview|summary", output_str, re.IGNORECASE)),
                bool(
                    re.search(r"recommendation|suggestion", output_str, re.IGNORECASE)
                ),
                bool(re.search(r"detail|specific", output_str, re.IGNORECASE)),
                bool(re.search(r"next step|action", output_str, re.IGNORECASE)),
            ]
        )

        score = (
            (0.3 if has_success else 0.0)
            + (0.3 if has_content else 0.0)
            + (0.4 * (component_count / 4))
        )

        return min(score, 1.0)

    def _score_actionability(self, output_str: str) -> float:
        """Score actionability of output (0.0-1.0)."""
        # Check for specific actions
        action_words = [
            "should",
            "must",
            "will",
            "implement",
            "execute",
            "create",
            "develop",
        ]
        action_count = sum(word in output_str.lower() for word in action_words)

        # Check for specifics (numbers, dates, names)
        has_specifics = bool(
            re.search(r"\d+|[\$€£]|[A-Z][a-z]+\s[A-Z][a-z]+", output_str)
        )

        # Check for step-by-step
        has_steps = bool(
            re.search(
                r"step \d+|first|second|third|then|next", output_str, re.IGNORECASE
            )
        )

        score = (
            (0.4 * min(action_count / 5, 1.0))
            + (0.3 if has_specifics else 0.0)
            + (0.3 if has_steps else 0.0)
        )

        return min(score, 1.0)

    def _score_accuracy(self, output: Dict[str, Any]) -> float:
        """Score accuracy (0.0-1.0) - checks for errors/hallucinations."""
        # Check for error indicators
        has_error = output.get("error") is not None
        has_success = output.get("success", True)

        # Simple heuristic: if no obvious errors, assume accurate
        # In practice, this would need more sophisticated checking
        if has_error or not has_success:
            return 0.0

        return 0.9  # High confidence if no errors

    def _score_helpfulness(self, output_str: str) -> float:
        """Score helpfulness (0.0-1.0)."""
        helpful_indicators = [
            "help",
            "assist",
            "guide",
            "support",
            "solution",
            "resolve",
            "answer",
            "clarify",
            "explain",
        ]

        indicator_count = sum(
            indicator in output_str.lower() for indicator in helpful_indicators
        )
        has_contact_info = bool(
            re.search(r"email|phone|support|help@", output_str, re.IGNORECASE)
        )
        has_resources = bool(
            re.search(r"documentation|link|resource|guide", output_str, re.IGNORECASE)
        )

        score = (
            (0.5 * min(indicator_count / 3, 1.0))
            + (0.25 if has_contact_info else 0.0)
            + (0.25 if has_resources else 0.0)
        )

        return min(score, 1.0)

    def _score_professionalism(self, output_str: str) -> float:
        """Score professionalism (0.0-1.0)."""
        # Check for professional language
        unprofessional_words = ["stupid", "dumb", "idiotic", "terrible", "awful"]
        has_unprofessional = any(
            word in output_str.lower() for word in unprofessional_words
        )

        # Check for professional tone markers
        professional_markers = ["please", "thank", "appreciate", "understand", "assist"]
        professional_count = sum(
            marker in output_str.lower() for marker in professional_markers
        )

        # Check for proper structure
        has_greeting = bool(
            re.search(r"^(hello|hi|dear|greetings)", output_str, re.IGNORECASE)
        )
        has_closing = bool(
            re.search(r"(regards|sincerely|best|thank)", output_str, re.IGNORECASE)
        )

        if has_unprofessional:
            return 0.0

        score = (
            (0.4 * min(professional_count / 2, 1.0))
            + (0.3 if has_greeting else 0.0)
            + (0.3 if has_closing else 0.0)
        )

        return min(score, 1.0)

    def _score_efficiency(self, output: Dict[str, Any]) -> float:
        """Score efficiency (0.0-1.0) - response time and conciseness."""
        # Check for success
        has_success = output.get("success", True)

        # Check response length (not too verbose)
        output_str = json.dumps(output, default=str)
        length = len(output_str)

        # Optimal range: 500-2000 characters
        if 500 <= length <= 2000:
            length_score = 1.0
        elif length < 500:
            length_score = length / 500
        else:
            length_score = max(0.5, 2000 / length)

        score = (0.5 if has_success else 0.0) + (0.5 * length_score)

        return min(score, 1.0)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a formatted evaluation report.

        Args:
            results: Evaluation results from evaluate_prompt()

        Returns:
            Formatted report string
        """
        if "error" in results:
            return f"ERROR: {results['error']}"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"  Prompt Evaluation Report")
        report_lines.append("=" * 70)
        report_lines.append(f"\nAgent: {results['agent_id']}")
        report_lines.append(f"Prompt Version: {results['prompt_version']}")
        report_lines.append(f"Timestamp: {results['timestamp']}")

        # Summary
        summary = results["summary"]
        report_lines.append(f"\n{'─' * 70}")
        report_lines.append("SUMMARY")
        report_lines.append(f"{'─' * 70}")
        report_lines.append(f"Average Score:    {summary['avg_score']:.2f}")
        report_lines.append(f"Pass Rate:        {summary['pass_rate']:.1%}")
        report_lines.append(f"Total Tests:      {summary['total_tests']}")
        report_lines.append(f"Passed:           {summary['passed_tests']}")
        report_lines.append(f"Failed:           {summary['failed_tests']}")

        # Test case details
        report_lines.append(f"\n{'─' * 70}")
        report_lines.append("TEST CASE DETAILS")
        report_lines.append(f"{'─' * 70}")

        for test_case in results["test_cases"]:
            status = "✅ PASS" if test_case["passed"] else "❌ FAIL"
            report_lines.append(
                f"\n{status} {test_case['id']} - {test_case['description']}"
            )
            report_lines.append(f"  Score: {test_case['score']:.2f}")

            # Behavior scores
            if test_case["behavior_scores"]:
                report_lines.append(f"  Behaviors:")
                for behavior, score in test_case["behavior_scores"].items():
                    status_icon = "✓" if score >= 0.5 else "✗"
                    report_lines.append(f"    {status_icon} {behavior}: {score:.2f}")

            # Quality scores
            if test_case["quality_scores"]:
                report_lines.append(f"  Quality:")
                for criterion, score in test_case["quality_scores"].items():
                    status_icon = "✓" if score >= 0.8 else "✗"
                    report_lines.append(f"    {status_icon} {criterion}: {score:.2f}")

        # Failed tests section
        failed_tests = [tc for tc in results["test_cases"] if not tc["passed"]]
        if failed_tests:
            report_lines.append(f"\n{'─' * 70}")
            report_lines.append("FAILED TESTS ANALYSIS")
            report_lines.append(f"{'─' * 70}")

            for test_case in failed_tests:
                report_lines.append(f"\n{test_case['id']}:")

                # Find failing behaviors
                failing_behaviors = [
                    b for b, s in test_case["behavior_scores"].items() if s < 0.5
                ]
                if failing_behaviors:
                    report_lines.append(
                        f"  Missing behaviors: {', '.join(failing_behaviors)}"
                    )

                # Find failing quality criteria
                failing_quality = [
                    q for q, s in test_case["quality_scores"].items() if s < 0.8
                ]
                if failing_quality:
                    report_lines.append(
                        f"  Quality issues: {', '.join(failing_quality)}"
                    )

        report_lines.append(f"\n{'=' * 70}")

        return "\n".join(report_lines)

    async def compare_versions(
        self,
        agent_id: str,
        version1: str,
        version2: str,
        test_cases: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Compare two prompt versions.

        Args:
            agent_id: Agent identifier
            version1: First version to compare
            version2: Second version to compare
            test_cases: Test cases to use

        Returns:
            Comparison results with improvement metrics
        """
        # Evaluate both versions
        results_v1 = await self.evaluate_prompt(agent_id, test_cases, version1)
        results_v2 = await self.evaluate_prompt(agent_id, test_cases, version2)

        if "error" in results_v1 or "error" in results_v2:
            return {
                "error": "Failed to evaluate one or both versions",
                "v1_error": results_v1.get("error"),
                "v2_error": results_v2.get("error"),
            }

        # Calculate improvements
        v1_score = results_v1["summary"]["avg_score"]
        v2_score = results_v2["summary"]["avg_score"]
        improvement = v2_score - v1_score
        improvement_pct = (improvement / v1_score * 100) if v1_score > 0 else 0

        return {
            "agent_id": agent_id,
            "version1": version1,
            "version2": version2,
            "v1_results": results_v1,
            "v2_results": results_v2,
            "comparison": {
                "score_improvement": improvement,
                "score_improvement_pct": improvement_pct,
                "pass_rate_change": results_v2["summary"]["pass_rate"]
                - results_v1["summary"]["pass_rate"],
                "better_version": version2 if improvement > 0 else version1,
            },
        }


# Example usage and tests
async def example_usage():
    """Example of how to use the PromptEvaluator."""
    evaluator = PromptEvaluator()

    # Evaluate current prompt
    print("\n" + "=" * 70)
    print("Evaluating Marketing Strategy Agent Prompt")
    print("=" * 70)

    results = await evaluator.evaluate_prompt("marketing_strategy")

    # Generate and print report
    report = evaluator.generate_report(results)
    print(report)

    print(f"\n\nOverall Score: {results['summary']['avg_score']:.2f}")
    print(f"Pass Rate: {results['summary']['pass_rate']:.1%}")


async def example_version_comparison():
    """Example of comparing prompt versions."""
    evaluator = PromptEvaluator()

    # Get available versions
    versions = evaluator.prompt_manager.list_versions("marketing_strategy")

    if len(versions) >= 2:
        v1 = versions[1]["version_id"]  # First historical
        v2 = "latest"

        print(f"\n\nComparing versions: {v1} vs {v2}")
        comparison = await evaluator.compare_versions("marketing_strategy", v1, v2)

        if "error" not in comparison:
            comp = comparison["comparison"]
            print(
                f"Score improvement: {comp['score_improvement']:+.2f} ({comp['score_improvement_pct']:+.1f}%)"
            )
            print(f"Pass rate change: {comp['pass_rate_change']:+.1%}")
            print(f"Better version: {comp['better_version']}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_usage())
    # asyncio.run(example_version_comparison())
