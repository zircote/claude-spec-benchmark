"""Configurable patterns for specification degradation.

This module defines the regex patterns and rules used by the DegradationEngine
to remove specific types of information from SWE-bench issues.

Patterns are organized by what they match (code blocks, file paths, etc.)
and can be customized per-repository if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DegradationPatterns:
    """Configurable patterns for specification degradation.

    Defines regex patterns that identify different types of technical content
    to be removed or obscured during degradation. Patterns are compiled lazily
    and cached.

    Example:
        >>> patterns = DegradationPatterns.default()
        >>> patterns.matches_code_block("```python\\nprint()\\n```")
        True

        >>> django_patterns = DegradationPatterns.for_repo("django/django")
        >>> # Returns Django-specific patterns
    """

    # Code-related patterns
    code_block_pattern: str = r"```[\s\S]*?```"
    inline_code_pattern: str = r"`[^`]+`"

    # Error/stack trace patterns
    stack_trace_pattern: str = (
        r"Traceback \(most recent call last\):[\s\S]*?(?=\n\n|\Z)"
    )
    exception_pattern: str = (
        r"(?:^|\n)(\w*(?:Error|Exception|Warning|Fault):\s*.+?)(?=\n|$)"
    )

    # File system patterns
    file_path_pattern: str = r"(?:^|\s)([a-zA-Z_][\w/.-]*\.(?:py|js|ts|jsx|tsx|java|go|rs|rb|c|cpp|h|hpp|cs|php|swift|kt))\b"
    line_number_pattern: str = r"(?:line\s*|L)(\d+)(?:\s*[-,]\s*\d+)?"

    # Technical identifier patterns
    function_ref_pattern: str = r"\b(?:def\s+|function\s+|fn\s+|func\s+)(\w+)\s*\("
    class_ref_pattern: str = r"\b(?:class\s+)(\w+)(?:\s*[:\(])"
    method_call_pattern: str = r"\b(\w+)\.(\w+)\s*\("

    # Version/commit patterns
    commit_sha_pattern: str = r"\b[a-f0-9]{7,40}\b"
    version_pattern: str = r"\b(?:v?\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?)\b"

    # URL patterns
    url_pattern: str = r"https?://[^\s<>\"']+"
    github_ref_pattern: str = r"(?:#|GH-|gh-)(\d+)"

    # Output/logs patterns
    log_line_pattern: str = r"(?:^|\n)(?:\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}|(?:DEBUG|INFO|WARN|ERROR|CRITICAL):).+?(?=\n|$)"

    # Compiled pattern cache (not serialized, not part of init)
    _compiled_cache: dict[str, re.Pattern[str]] = field(
        default_factory=dict, repr=False, compare=False, init=False
    )

    def _compile(self, pattern_name: str) -> re.Pattern[str]:
        """Get or compile a pattern by name.

        Args:
            pattern_name: Name of the pattern attribute (e.g., 'code_block_pattern')

        Returns:
            Compiled regex pattern with MULTILINE and IGNORECASE flags.
        """
        if pattern_name not in self._compiled_cache:
            pattern = getattr(self, pattern_name)
            self._compiled_cache[pattern_name] = re.compile(
                pattern, re.MULTILINE | re.IGNORECASE
            )
        return self._compiled_cache[pattern_name]

    def find_all(self, pattern_name: str, text: str) -> list[tuple[str, int, int]]:
        """Find all matches of a pattern in text.

        Args:
            pattern_name: Name of the pattern attribute
            text: Text to search

        Returns:
            List of (matched_text, start_index, end_index) tuples
        """
        compiled = self._compile(pattern_name)
        return [
            (match.group(0), match.start(), match.end())
            for match in compiled.finditer(text)
        ]

    def remove_all(self, pattern_name: str, text: str, replacement: str = "") -> str:
        """Remove all matches of a pattern from text.

        Args:
            pattern_name: Name of the pattern attribute
            text: Text to modify
            replacement: String to replace matches with

        Returns:
            Modified text with matches removed/replaced
        """
        compiled = self._compile(pattern_name)
        return compiled.sub(replacement, text)

    def matches_code_block(self, text: str) -> bool:
        """Check if text contains a code block."""
        return bool(self._compile("code_block_pattern").search(text))

    def matches_stack_trace(self, text: str) -> bool:
        """Check if text contains a stack trace."""
        return bool(self._compile("stack_trace_pattern").search(text))

    def matches_file_path(self, text: str) -> bool:
        """Check if text contains a file path."""
        return bool(self._compile("file_path_pattern").search(text))

    @classmethod
    def default(cls) -> DegradationPatterns:
        """Return default patterns suitable for most Python repositories.

        Returns:
            DegradationPatterns with sensible defaults for Python projects.
        """
        return cls()

    @classmethod
    def for_repo(cls, repo: str) -> DegradationPatterns:
        """Return patterns customized for a specific repository.

        Some repositories have unique conventions that require adjusted patterns.

        Args:
            repo: Repository name in 'owner/name' format (e.g., 'django/django')

        Returns:
            DegradationPatterns customized for the repository, or defaults.
        """
        # Repository-specific overrides
        repo_patterns = _REPO_SPECIFIC_PATTERNS.get(repo.lower())
        if repo_patterns:
            return cls(**repo_patterns)

        # Check for framework-specific patterns
        if "django" in repo.lower():
            return cls(**_DJANGO_PATTERNS)
        if "flask" in repo.lower():
            return cls(**_FLASK_PATTERNS)
        if "pytorch" in repo.lower() or "torch" in repo.lower():
            return cls(**_PYTORCH_PATTERNS)

        return cls.default()

    def get_patterns_for_level(self, level: str) -> list[str]:
        """Get the pattern names to apply for a given degradation level.

        Args:
            level: Degradation level ('partial', 'vague', 'minimal', 'ambiguous')

        Returns:
            List of pattern attribute names to apply for this level.
        """
        level_patterns = {
            "full": [],  # No patterns applied
            "partial": [
                "code_block_pattern",
                "stack_trace_pattern",
                "file_path_pattern",
                "line_number_pattern",
                "commit_sha_pattern",
            ],
            "vague": [
                "code_block_pattern",
                "inline_code_pattern",
                "stack_trace_pattern",
                "exception_pattern",
                "file_path_pattern",
                "line_number_pattern",
                "function_ref_pattern",
                "class_ref_pattern",
                "method_call_pattern",
                "commit_sha_pattern",
                "version_pattern",
                "log_line_pattern",
            ],
            "minimal": [
                # All patterns - maximum removal
                "code_block_pattern",
                "inline_code_pattern",
                "stack_trace_pattern",
                "exception_pattern",
                "file_path_pattern",
                "line_number_pattern",
                "function_ref_pattern",
                "class_ref_pattern",
                "method_call_pattern",
                "commit_sha_pattern",
                "version_pattern",
                "url_pattern",
                "github_ref_pattern",
                "log_line_pattern",
            ],
            "ambiguous": [
                # Same as minimal, but engine will add contradictions
                "code_block_pattern",
                "inline_code_pattern",
                "stack_trace_pattern",
                "exception_pattern",
                "file_path_pattern",
                "line_number_pattern",
                "function_ref_pattern",
                "class_ref_pattern",
                "method_call_pattern",
                "commit_sha_pattern",
                "version_pattern",
                "url_pattern",
                "github_ref_pattern",
                "log_line_pattern",
            ],
        }
        return level_patterns.get(level.lower(), [])


# =============================================================================
# Repository-specific pattern overrides
# =============================================================================

_DJANGO_PATTERNS: dict[str, str] = {
    # Django-specific patterns
    "file_path_pattern": r"(?:^|\s)([a-zA-Z_][\w/.-]*\.(?:py|html|txt|js|css))\b",
    "function_ref_pattern": r"\b(?:def\s+)(\w+)\s*\(",
    # Django settings references
    "log_line_pattern": r"(?:^|\n)(?:\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}|(?:DEBUG|INFO|WARN|ERROR|CRITICAL)|django\.).+?(?=\n|$)",
}

_FLASK_PATTERNS: dict[str, str] = {
    # Flask-specific patterns
    "file_path_pattern": r"(?:^|\s)([a-zA-Z_][\w/.-]*\.(?:py|html|jinja2|js|css))\b",
}

_PYTORCH_PATTERNS: dict[str, str] = {
    # PyTorch-specific patterns (includes CUDA errors, tensor shapes)
    "exception_pattern": r"(?:^|\n)(\w*(?:Error|Exception|Warning|RuntimeError|CudaError):\s*.+?)(?=\n|$)",
    "log_line_pattern": r"(?:^|\n)(?:\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}|(?:DEBUG|INFO|WARN|ERROR|CRITICAL)|CUDA|tensor\().+?(?=\n|$)",
}

# Direct repository overrides (highest priority)
_REPO_SPECIFIC_PATTERNS: dict[str, dict[str, str]] = {
    # Add specific repository overrides here if needed
    # "owner/repo": {"pattern_name": "pattern_value", ...}
}
