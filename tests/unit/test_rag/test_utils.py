"""Tests for RAG utility functions."""

import pytest
from src.marketing_agents.rag.utils import (
    calculate_reading_time,
    categorize_document_content,
    clean_content,
    clean_html,
    extract_title_from_content,
    normalize_whitespace,
)


class TestCleanHTML:
    """Tests for HTML cleaning functionality."""

    def test_simple_html(self):
        """Test basic HTML tag removal."""
        html = "<p>Hello <strong>world</strong>!</p>"
        result = clean_html(html)
        assert "Hello world!" in result
        assert "<p>" not in result
        assert "<strong>" not in result

    def test_html_with_scripts(self):
        """Test that script tags are removed."""
        html = "<div>Content</div><script>alert('test')</script>"
        result = clean_html(html)
        assert "Content" in result
        assert "alert" not in result

    def test_nested_html(self):
        """Test nested HTML elements."""
        html = "<div><p>Paragraph in <span>div</span></p></div>"
        result = clean_html(html)
        assert "Paragraph in div" in result


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""

    def test_multiple_spaces(self):
        """Test multiple spaces are collapsed."""
        text = "Hello    world"
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_multiple_newlines(self):
        """Test multiple newlines are collapsed."""
        text = "Line 1\n\n\n\nLine 2"
        result = normalize_whitespace(text)
        assert result == "Line 1\n\nLine 2"

    def test_leading_trailing_whitespace(self):
        """Test leading/trailing whitespace is removed."""
        text = "  \n  Hello  \n  "
        result = normalize_whitespace(text)
        assert result == "Hello"


class TestExtractTitle:
    """Tests for title extraction."""

    def test_markdown_h1(self):
        """Test extraction from markdown H1."""
        content = "# My Document Title\n\nSome content here."
        result = extract_title_from_content(content)
        assert result == "My Document Title"

    def test_html_h1(self):
        """Test extraction from HTML H1."""
        content = "<h1>HTML Document Title</h1><p>Content</p>"
        result = extract_title_from_content(content)
        assert result == "HTML Document Title"

    def test_first_line_fallback(self):
        """Test fallback to first line."""
        content = "Document Title\n\nRest of content"
        result = extract_title_from_content(content)
        assert result == "Document Title"

    def test_no_title(self):
        """Test when no clear title exists."""
        content = ""
        result = extract_title_from_content(content)
        assert result is None


class TestCategorizeDocument:
    """Tests for document categorization."""

    def test_api_reference(self):
        """Test API reference detection."""
        content = "API Endpoint: POST /v1/customers\nParameters: name, email"
        result = categorize_document_content(content)
        assert result == "API Reference"

    def test_tutorial(self):
        """Test tutorial detection."""
        content = "Step 1: Install the package\nStep 2: Configure settings"
        result = categorize_document_content(content)
        assert result == "Tutorial"

    def test_faq(self):
        """Test FAQ detection."""
        content = "Q: How do I get started?\nA: Follow these steps..."
        result = categorize_document_content(content)
        assert result == "FAQ"

    def test_guide(self):
        """Test guide detection."""
        content = "How to integrate with Stripe: A comprehensive guide"
        result = categorize_document_content(content)
        assert result == "Guide"

    def test_path_based_categorization(self):
        """Test categorization based on file path."""
        content = "Generic content"
        result = categorize_document_content(content, "/docs/api/reference.md")
        assert result == "API Reference"


class TestCleanContent:
    """Tests for content cleaning."""

    def test_html_file(self):
        """Test HTML file cleaning."""
        content = "<html><body><p>Test content</p></body></html>"
        result = clean_content(content, ".html")
        assert "<html>" not in result
        assert "Test content" in result

    def test_markdown_with_html(self):
        """Test markdown with embedded HTML."""
        content = "# Title\n\n<div>HTML content</div>\n\nMore text"
        result = clean_content(content, ".md")
        # Should still clean HTML tags
        assert "<div>" not in result

    def test_plain_text(self):
        """Test plain text file."""
        content = "Simple   text   content\n\n\n\nWith spacing"
        result = clean_content(content, ".txt")
        # Should normalize whitespace
        assert "Simple text content" in result


class TestCalculateReadingTime:
    """Tests for reading time calculation."""

    def test_short_text(self):
        """Test reading time for short text."""
        result = calculate_reading_time(50)
        assert result == 1  # Minimum 1 minute

    def test_long_text(self):
        """Test reading time for long text."""
        result = calculate_reading_time(1000)  # 1000 words
        assert result == 5  # 1000 / 200 = 5 minutes

    def test_custom_wpm(self):
        """Test with custom words per minute."""
        result = calculate_reading_time(400, words_per_minute=100)
        assert result == 4  # 400 / 100 = 4 minutes
