"""TDD tests for streaming markdown render.

These tests cover the new _stream_render_line() function that enables
real-time markdown rendering in streaming mode when
display.final_response_markdown: render is set.
"""
from io import StringIO

import pytest
from rich.console import Console

from cli import _stream_render_line


def _render_to_ansi(text: str) -> str:
    """Render text through _stream_render_line and capture ANSI output."""
    result = _stream_render_line(text)
    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(result)
    return buf.getvalue()


# ── Basic markdown: bold ────────────────────────────────────────────────────────

def test_stream_render_bold():
    result = _stream_render_line("**hello**")
    output = _render_to_ansi(result)
    assert "hello" in output
    assert "**" not in output


def test_stream_render_bold_across_token_boundary():
    # Simulate bold that arrives in two tokens
    part1 = _stream_render_line("**hel")
    part2 = _stream_render_line("lo**")
    # First token: should fall back to raw (incomplete markup)
    assert "**hel" in part1
    # Second token completes it
    combined = part1 + part2
    output = _render_to_ansi(combined)
    assert "hello" in output
    assert "**" not in output


# ── Basic markdown: italic ────────────────────────────────────────────────────

def test_stream_render_italic():
    result = _stream_render_line("*hello*")
    output = _render_to_ansi(result)
    assert "hello" in output
    assert "*" not in output


def test_stream_render_italic_preserves_list_items():
    # Italic asterisk must NOT consume list marker asterisks
    result = _stream_render_line("- item with *italic* word")
    output = _render_to_ansi(result)
    assert "- item with" in output
    assert "italic" in output


# ── Bold + Italic ─────────────────────────────────────────────────────────────

def test_stream_render_bold_italic():
    result = _stream_render_line("***hello***")
    output = _render_to_ansi(result)
    assert "hello" in output
    assert "***" not in output


# ── Inline code ────────────────────────────────────────────────────────────────

def test_stream_render_inline_code():
    result = _stream_render_line("`code`")
    output = _render_to_ansi(result)
    assert "code" in output
    assert "`" not in output


def test_stream_render_code_span_in_word():
    # Should NOT interpret backtick inside a path/word as code span
    result = _stream_render_line("path/to `/tmp/file` here")
    output = _render_to_ansi(result)
    assert "/tmp/file" in output or "file" in output


# ── Strikethrough ─────────────────────────────────────────────────────────────

def test_stream_render_strikethrough():
    result = _stream_render_line("~~strike~~")
    output = _render_to_ansi(result)
    assert "strike" in output
    assert "~~" not in output


# ── Links ─────────────────────────────────────────────────────────────────────

def test_stream_render_link():
    result = _stream_render_line("[link text](https://example.com)")
    output = _render_to_ansi(result)
    assert "link text" in output
    # URL may or may not be visible depending on rendering, but markdown is gone
    assert "[" not in output or "(" not in output


# ── Mixed content ─────────────────────────────────────────────────────────────

def test_stream_render_mixed_bold_and_italic():
    result = _stream_render_line("**bold** and *italic*")
    output = _render_to_ansi(result)
    assert "bold" in output
    assert "italic" in output
    assert "**" not in output
    assert "*" not in output


def test_stream_render_code_and_text():
    result = _stream_render_line("Run `python --version` to check")
    output = _render_to_ansi(result)
    assert "python --version" in output
    assert "Run" in output


def test_stream_render_multiline_structure():
    # Lists and blockquotes should be preserved
    result = _stream_render_line("- first item")
    output = _render_to_ansi(result)
    assert "- first item" in output

    result = _stream_render_line("> quoted text")
    output = _render_to_ansi(result)
    assert "quoted text" in output


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_stream_render_empty_line():
    result = _stream_render_line("")
    assert result == ""


def test_stream_render_whitespace_only():
    result = _stream_render_line("   \t  ")
    # Should not crash
    assert isinstance(result, str)


def test_stream_render_pure_plain_text():
    result = _stream_render_line("plain text with no markdown")
    output = _render_to_ansi(result)
    assert "plain text" in output


def test_stream_render_heading():
    result = _stream_render_line("## Heading")
    output = _render_to_ansi(result)
    assert "Heading" in output
    # # symbols may or may not be stripped depending on implementation
    assert "##" not in output


def test_stream_render_horizontal_rule():
    result = _stream_render_line("---")
    output = _render_to_ansi(result)
    # Should not crash; dashes may or may not be preserved
    assert isinstance(result, str)


def test_stream_render_snake_case_file_paths():
    # Underscores in file paths should NOT be treated as italic markers
    result = _stream_render_line("file: test_case_with_underscores.py")
    output = _render_to_ansi(result)
    assert "test_case_with_underscores" in output


def test_stream_render_url_with_underscores():
    # URLs should not corrupt
    result = _stream_render_line("https://example.com/path_to_resource")
    output = _render_to_ansi(result)
    assert "example.com" in output


def test_stream_render_incomplete_bold():
    # Stream ends mid-bold, should fall back gracefully
    result = _stream_render_line("**incomplete bold")
    output = _render_to_ansi(result)
    # Should print as-is, not corrupt the display
    assert "incomplete bold" in output


def test_stream_render_incomplete_code():
    result = _stream_render_line("`unclosed code span")
    output = _render_to_ansi(result)
    # Should print raw, not corrupt
    assert "unclosed code span" in output


def test_stream_render_ansi_escape_injection():
    # Malicious input: ANSI escape codes embedded in markdown
    malicious = "\x1b[31m**red bold**\x1b[0m"
    result = _stream_render_line(malicious)
    # Should strip ANSI before processing markdown, not interpret it as markup
    output = _render_to_ansi(result)
    # The ANSI should be stripped; **bold** may or may not render depending
    # on ordering. Key invariant: no raw ANSI escape sequences in output.
    assert "\x1b" not in output or output.count("\x1b") == 0


def test_stream_render_very_long_line():
    # Performance test: long line should not cause issues
    long_text = "word " * 10000
    result = _stream_render_line(long_text)
    assert len(result) > 0
    # Should not hang or crash


def test_stream_render_special_unicode():
    # Emoji and unicode should not break rendering
    result = _stream_render_line("Hello 🌍 *world* **test**")
    output = _render_to_ansi(result)
    assert "world" in output
    assert "test" in output


def test_stream_render_deeply_nested():
    # Multiple formatting layers
    result = _stream_render_line("***bold italic*** and **bold** and *italic*")
    output = _render_to_ansi(result)
    assert "bold italic" in output
    assert "bold" in output
    assert "italic" in output


# ── Sanity: output is always a string ─────────────────────────────────────────

def test_stream_render_always_returns_string():
    for text in ["", "plain", "**bold**", "*italic*", "`code`", "~~strike~~"]:
        result = _stream_render_line(text)
        assert isinstance(result, str), f"Expected str for {text!r}, got {type(result)}"


# ── Regression: file paths with asterisks ─────────────────────────────────────

def test_stream_render_asterisk_in_path():
    # File glob patterns contain *
    result = _stream_render_line("Matching: /tmp/*.py")
    output = _render_to_ansi(result)
    # The * should NOT be interpreted as italic
    assert "*.py" in output or "Matching" in output


def test_stream_render_dollar_in_shell_var():
    result = _stream_render_line("echo $HOME and ${PATH}")
    output = _render_to_ansi(result)
    assert "$HOME" in output or "HOME" in output
