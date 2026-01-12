"""Utility functions for RAG document processing."""

import re
from html.parser import HTMLParser
from typing import Optional


class HTMLTextExtractor(HTMLParser):
    """Extract plain text from HTML content."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {"script", "style", "meta", "link"}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            self.text_parts.append(data)

    def get_text(self) -> str:
        return "".join(self.text_parts)


def clean_html(html_content: str) -> str:
    """
    Remove HTML tags and extract plain text.

    Args:
        html_content: HTML content to clean

    Returns:
        Plain text with HTML tags removed
    """
    parser = HTMLTextExtractor()
    try:
        parser.feed(html_content)
        return parser.get_text()
    except Exception:
        # Fallback to regex-based cleaning if parser fails
        return re.sub(r"<[^>]+>", "", html_content)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text content.

    Args:
        text: Text to normalize

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with at most two newlines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def extract_title_from_content(content: str) -> Optional[str]:
    """
    Extract title from document content.

    Tries multiple strategies:
    1. H1 heading (# Title in Markdown)
    2. First line if it looks like a title
    3. First non-empty line

    Args:
        content: Document content

    Returns:
        Extracted title or None
    """
    lines = content.strip().split("\n")

    # Strategy 1: Look for markdown H1
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()

    # Strategy 2: Look for HTML h1
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.IGNORECASE | re.DOTALL)
    if h1_match:
        return clean_html(h1_match.group(1)).strip()

    # Strategy 3: First non-empty line that looks like a title
    for line in lines[:5]:
        line = line.strip()
        if line and len(line) < 200:  # Titles are usually short
            # Remove common markdown/HTML artifacts
            line = re.sub(r"^[#\*\-=]+\s*", "", line)
            line = re.sub(r"\s*[#\*\-=]+$", "", line)
            if line:
                return line

    return None


def categorize_document_content(content: str, file_path: str = "") -> str:
    """
    Categorize document based on content and file path.

    Categories:
    - API Reference: API docs, endpoints, parameters
    - Tutorial: Step-by-step guides
    - Guide: General how-to documentation
    - FAQ: Frequently asked questions
    - Conceptual: Overview and concept explanations

    Args:
        content: Document content
        file_path: Optional file path for additional context

    Returns:
        Document category
    """
    content_lower = content.lower()
    path_lower = file_path.lower()

    # Check file path first
    if "api" in path_lower or "reference" in path_lower:
        return "API Reference"
    if "tutorial" in path_lower:
        return "Tutorial"
    if "faq" in path_lower:
        return "FAQ"
    if "guide" in path_lower:
        return "Guide"

    # Content-based classification
    api_keywords = [
        "endpoint",
        "api",
        "request",
        "response",
        "parameter",
        "method",
        "curl",
    ]
    tutorial_keywords = ["step 1", "step 2", "first,", "next,", "finally,", "tutorial"]
    faq_keywords = ["question:", "q:", "a:", "frequently asked", "faq"]
    guide_keywords = ["how to", "getting started", "guide", "introduction"]

    # Count keyword matches
    api_score = sum(1 for kw in api_keywords if kw in content_lower)
    tutorial_score = sum(1 for kw in tutorial_keywords if kw in content_lower)
    faq_score = sum(1 for kw in faq_keywords if kw in content_lower)
    guide_score = sum(1 for kw in guide_keywords if kw in content_lower)

    # Determine category based on highest score
    scores = {
        "API Reference": api_score,
        "Tutorial": tutorial_score,
        "FAQ": faq_score,
        "Guide": guide_score,
    }

    max_score = max(scores.values())
    if max_score > 0:
        return max(scores.items(), key=lambda x: x[1])[0]

    # Default to conceptual if no strong signals
    return "Conceptual"


def clean_content(content: str, file_extension: str) -> str:
    """
    Clean content based on file type.

    Args:
        content: Raw content
        file_extension: File extension (.md, .html, .txt)

    Returns:
        Cleaned content
    """
    # Remove HTML tags if present (regardless of extension, some .md files have HTML)
    # Check for any HTML tags, not just html/body tags
    if file_extension == ".html" or re.search(r"<[a-zA-Z][^>]*>", content):
        content = clean_html(content)

    # Normalize whitespace
    content = normalize_whitespace(content)

    return content


def calculate_reading_time(word_count: int, words_per_minute: int = 200) -> int:
    """
    Calculate estimated reading time in minutes.

    Args:
        word_count: Number of words
        words_per_minute: Average reading speed (default: 200 wpm)

    Returns:
        Reading time in minutes
    """
    return max(1, round(word_count / words_per_minute))
