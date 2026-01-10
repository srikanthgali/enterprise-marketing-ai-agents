#!/usr/bin/env python3
"""
Enterprise Marketing AI Agents - Project Structure Generator

A Enterprise-ready multi-agent AI system for marketing platforms.
Demonstrates real-world architecture, orchestration, agent handoffs, memory,
evaluation, and learning workflows optimized for technical interviews.

Features:
    - Multi-agent orchestration system
    - Specialized marketing agents (Content Strategist, SEO, Copywriter, etc.)
    - Agent handoff and message bus architecture
    - Vector-based memory and context management
    - Comprehensive evaluation and learning framework
    - FastAPI REST API
    - Streamlit/Gradio UI interfaces
    - End-to-end marketing workflows
    - Configuration via YAML

Author: Srikanth Gali
"""

import logging
from pathlib import Path
from typing import Dict
from datetime import datetime


class MarketingAgentsProjectGenerator:
    """
    Generates an enterprise marketing AI agents project structure.

    This generator creates a professional, modular project layout demonstrating
    production-ready patterns for multi-agent systems. Designed to showcase
    advanced AI engineering skills in technical interviews and portfolio reviews.

    Attributes:
        project_name (str): Name of the project
        base_path (Path): Base directory path where project will be created
        created_count (int): Counter for tracking created files/directories

    Example:
        >>> generator = MarketingAgentsProjectGenerator()
        >>> generator.create_project_structure()
    """

    def __init__(self, project_name: str = "enterprise-marketing-ai-agents"):
        """
        Initialize the project generator.

        Args:
            project_name (str): Name of the project
        """
        self.project_name = project_name
        self.base_path = Path(".")
        self.created_count = 0

    def create_project_structure(self) -> None:
        """
        Create the complete project directory structure.

        Creates an enterprise-grade structure focusing on:
        - Multi-agent orchestration
        - Clean, modular architecture
        - Production-ready patterns
        - Comprehensive testing and evaluation
        - Documentation and notebooks
        """
        logging.info(f"Creating project structure for {self.project_name}")
        logging.info(f"Base path: {self.base_path.absolute()}")

        self._create_directories()
        self._create_files()
        self._create_gitkeep_files()

        logging.info(
            f"✓ {self.project_name} structure created successfully at: {self.base_path.absolute()}"
        )
        logging.info(f"✓ Total new files and directories created: {self.created_count}")

    def _create_directories(self) -> None:
        """
        Create essential project directories.

        Enterprise-grade structure for multi-agent AI system.
        """
        directories = [
            # Configuration
            "config/prompts",
            # Core source code
            "src/marketing_agents/core",
            "src/marketing_agents/agents",
            "src/marketing_agents/tools",
            "src/marketing_agents/memory",
            "src/marketing_agents/evaluation",
            "src/marketing_agents/learning",
            "src/marketing_agents/utils",
            # API
            "api/routes",
            "api/schemas",
            "api/middleware",
            # UI
            "ui/components",
            "ui/static/css",
            "ui/static/js",
            "ui/static/images",
            # Workflows
            "workflows",
            # Tests
            "tests/unit",
            "tests/integration",
            "tests/evaluation",
            # Data
            "data/raw",
            "data/processed",
            "data/embeddings",
            "data/knowledge_base",
            # Models
            "models/prompts",
            "models/fine_tuned",
            "models/embeddings",
            # Logs
            "logs/agents",
            "logs/api",
            "logs/evaluation",
            # Notebooks
            "notebooks",
            # Scripts
            "scripts",
            # Documentation
            "docs",
        ]

        for directory in directories:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.created_count += 1
                logging.info(f"✓ Created: {directory}")

    def _create_files(self) -> None:
        """
        Create essential project files.

        Comprehensive file structure for enterprise multi-agent system.
        """
        files = [
            # Root configuration
            ".gitignore",
            ".env.example",
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "Makefile",
            # Config
            "config/__init__.py",
            "config/settings.py",
            "config/agents_config.yaml",
            "config/models_config.yaml",
            "config/prompts/content_strategist.txt",
            "config/prompts/seo_specialist.txt",
            "config/prompts/copywriter.txt",
            "config/prompts/campaign_manager.txt",
            # Source - Core
            "src/__init__.py",
            "src/marketing_agents/__init__.py",
            "src/marketing_agents/core/__init__.py",
            "src/marketing_agents/core/base_agent.py",
            "src/marketing_agents/core/orchestrator.py",
            "src/marketing_agents/core/handoff_manager.py",
            "src/marketing_agents/core/message_bus.py",
            # Source - Agents
            "src/marketing_agents/agents/__init__.py",
            "src/marketing_agents/agents/content_strategist.py",
            "src/marketing_agents/agents/seo_specialist.py",
            "src/marketing_agents/agents/copywriter.py",
            "src/marketing_agents/agents/campaign_manager.py",
            "src/marketing_agents/agents/data_analyst.py",
            # Source - Tools
            "src/marketing_agents/tools/__init__.py",
            "src/marketing_agents/tools/web_search.py",
            "src/marketing_agents/tools/content_analysis.py",
            "src/marketing_agents/tools/keyword_research.py",
            "src/marketing_agents/tools/competitor_analysis.py",
            "src/marketing_agents/tools/analytics_retrieval.py",
            # Source - Memory
            "src/marketing_agents/memory/__init__.py",
            "src/marketing_agents/memory/vector_store.py",
            "src/marketing_agents/memory/conversation_memory.py",
            "src/marketing_agents/memory/context_manager.py",
            "src/marketing_agents/memory/knowledge_base.py",
            # Source - Evaluation
            "src/marketing_agents/evaluation/__init__.py",
            "src/marketing_agents/evaluation/metrics.py",
            "src/marketing_agents/evaluation/evaluators.py",
            "src/marketing_agents/evaluation/performance_tracker.py",
            "src/marketing_agents/evaluation/ab_testing.py",
            # Source - Learning
            "src/marketing_agents/learning/__init__.py",
            "src/marketing_agents/learning/feedback_loop.py",
            "src/marketing_agents/learning/model_tuning.py",
            "src/marketing_agents/learning/optimization.py",
            # Source - Utils
            "src/marketing_agents/utils/__init__.py",
            "src/marketing_agents/utils/logging.py",
            "src/marketing_agents/utils/validators.py",
            "src/marketing_agents/utils/formatters.py",
            "src/marketing_agents/utils/exceptions.py",
            # API
            "api/__init__.py",
            "api/main.py",
            "api/dependencies.py",
            "api/routes/__init__.py",
            "api/routes/agents.py",
            "api/routes/campaigns.py",
            "api/routes/content.py",
            "api/routes/analytics.py",
            "api/schemas/__init__.py",
            "api/schemas/request.py",
            "api/schemas/response.py",
            "api/schemas/models.py",
            "api/middleware/__init__.py",
            "api/middleware/auth.py",
            "api/middleware/rate_limit.py",
            "api/middleware/error_handler.py",
            # UI
            "ui/__init__.py",
            "ui/streamlit_app.py",
            "ui/gradio_app.py",
            "ui/components/__init__.py",
            "ui/components/chat_interface.py",
            "ui/components/dashboard.py",
            "ui/components/visualizations.py",
            "ui/static/css/styles.css",
            # Workflows
            "workflows/__init__.py",
            "workflows/content_creation.py",
            "workflows/campaign_planning.py",
            "workflows/seo_optimization.py",
            "workflows/performance_analysis.py",
            # Tests
            "tests/__init__.py",
            "tests/conftest.py",
            "tests/unit/test_agents.py",
            "tests/unit/test_tools.py",
            "tests/unit/test_memory.py",
            "tests/unit/test_orchestrator.py",
            "tests/integration/test_workflows.py",
            "tests/integration/test_api.py",
            "tests/integration/test_agent_handoffs.py",
            "tests/evaluation/test_metrics.py",
            "tests/evaluation/test_evaluators.py",
            # Scripts
            "scripts/setup_environment.sh",
            "scripts/run_evaluation.py",
            "scripts/deploy_agents.py",
            "scripts/generate_report.py",
            # Notebooks
            "notebooks/01_agent_design.ipynb",
            "notebooks/02_workflow_testing.ipynb",
            "notebooks/03_evaluation_analysis.ipynb",
            "notebooks/04_performance_optimization.ipynb",
            # Documentation
            "docs/architecture.md",
            "docs/agents_overview.md",
            "docs/api_reference.md",
            "docs/deployment.md",
            "docs/evaluation_framework.md",
        ]

        for file_path_str in files:
            file_path = self.base_path / file_path_str
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if not file_path.exists():
                file_path.touch()
                self.created_count += 1
                logging.info(f"✓ Created: {file_path_str}")

    def _create_gitkeep_files(self) -> None:
        """Create .gitkeep files in empty directories."""
        gitkeep_dirs = [
            "data/raw",
            "data/processed",
            "data/embeddings",
            "data/knowledge_base",
            "models/fine_tuned",
            "logs/agents",
            "logs/api",
            "logs/evaluation",
        ]

        for gitkeep_dir in gitkeep_dirs:
            gitkeep_path = self.base_path / gitkeep_dir / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
                self.created_count += 1
                logging.info(f"✓ Created: {gitkeep_dir}/.gitkeep")

    def generate_summary_report(self) -> Dict[str, any]:
        """
        Generate a summary report of the project structure.

        Returns:
            Dict containing project statistics and metadata
        """
        return {
            "project_name": self.project_name,
            "base_path": str(self.base_path.absolute()),
            "new_items_created": self.created_count,
            "timestamp": datetime.now().isoformat(),
            "status": "Enterprise-grade multi-agent marketing AI system",
            "portfolio_quality": "Production-ready architecture for interviews",
            "key_features": [
                "Multi-agent orchestration (Content, SEO, Copywriter, Campaign Manager)",
                "Agent handoff and message bus architecture",
                "Vector-based memory and context management",
                "Comprehensive evaluation and metrics framework",
                "Learning and optimization loops",
                "FastAPI REST API with authentication",
                "Streamlit & Gradio UI interfaces",
                "End-to-end marketing workflows",
                "Unit, integration, and evaluation testing",
                "Production-ready logging and monitoring",
            ],
            "agents": [
                "Content Strategist",
                "SEO Specialist",
                "Copywriter",
                "Campaign Manager",
                "Data Analyst",
            ],
        }


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the project generator.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    setup_logging(log_level="INFO")

    generator = MarketingAgentsProjectGenerator()

    try:
        generator.create_project_structure()

        summary = generator.generate_summary_report()
        print("\n" + "=" * 80)
        print("🚀 ENTERPRISE MARKETING AI AGENTS - PROJECT STRUCTURE")
        print("=" * 80)
        print(f"Project Name: {summary['project_name']}")
        print(f"Location: {summary['base_path']}")
        print(f"Items Created: {summary['new_items_created']}")
        print(f"Status: {summary['status']}")
        print(f"Quality: {summary['portfolio_quality']}")
        print("\n🤖 AI Agents:")
        for agent in summary["agents"]:
            print(f"   ✓ {agent}")
        print("\n📋 Key Features:")
        for feature in summary["key_features"]:
            print(f"   ✓ {feature}")
        print("\n💡 Next Steps:")
        print("   1. Update .env.example with your API keys (OpenAI, etc.)")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Configure agents: edit config/agents_config.yaml")
        print("   4. Review architecture: docs/architecture.md")
        print("   5. Run tests: pytest tests/")
        print("   6. Start API: uvicorn api.main:app --reload")
        print("   7. Launch UI: streamlit run ui/streamlit_app.py")
        print("   8. Explore notebooks: jupyter lab notebooks/")
        print("\n📚 Documentation:")
        print("   - Architecture: docs/architecture.md")
        print("   - Agents Overview: docs/agents_overview.md")
        print("   - API Reference: docs/api_reference.md")
        print("=" * 80 + "\n")

    except Exception as e:
        logging.error(f"Error creating project structure: {e}")
        raise
