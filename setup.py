#!/usr/bin/env python3
"""
Setup script for Enterprise Marketing AI Agents

This setup file allows the project to be installed as a package,
making imports cleaner and enabling development mode installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (
    (this_directory / "README.md").read_text(encoding="utf-8")
    if (this_directory / "README.md").exists()
    else ""
)

__version__ = "0.0.0"
REPO_NAME = "enterprise-marketing-ai-agents"
AUTHOR_NAME = "Srikanth Gali"
AUTHOR_USER_NAME = "srikanthgali"
SRC_REPO = "enterprise-marketing-ai-agents"
AUTHOR_EMAIL = "srikanthgali137@gmail.com"


# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt file."""
    requirements_file = this_directory / "requirements.txt"
    if not requirements_file.exists():
        return []

    with open(requirements_file, encoding="utf-8") as f:
        requirements = []
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and editable installs
            if line and not line.startswith("#") and not line.startswith("-e"):
                requirements.append(line)
        return requirements


setup(
    name="enterprise-marketing-ai-agents",
    version="0.1.0",
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description="Enterprise-grade multi-agent AI system for marketing platforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "pytest-asyncio>=0.24.0",
            "pytest-cov>=6.0.0",
            "pytest-mock>=3.14.0",
            "black>=24.10.0",
            "isort>=5.13.2",
            "flake8>=7.1.0",
            "mypy>=1.13.0",
            "pylint>=3.3.0",
            "pre-commit>=4.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
            "mkdocstrings[python]>=0.24.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipykernel>=6.27.0",
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0",
            "plotly>=5.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "marketing-agents=marketing_agents.cli:main",
            "marketing-api=api.main:start_server",
        ],
    },
    include_package_data=True,
    package_data={
        "marketing_agents": [
            "config/*.yaml",
            "config/prompts/*.txt",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai",
        "agents",
        "multi-agent",
        "marketing",
        "llm",
        "langchain",
        "openai",
        "rag",
        "orchestration",
    ],
    project_urls={
        "Bug Reports": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        "Source": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
        "Documentation": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/docs",
    },
)
