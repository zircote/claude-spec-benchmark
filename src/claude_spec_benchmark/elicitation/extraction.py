"""Requirements extraction module for SDD-Bench.

This module provides the RequirementsExtractor class that parses SWE-bench
issue text into atomic requirements. These requirements become the "hidden
knowledge" that the ElicitationOracle uses to simulate stakeholder responses.

The extractor uses heuristic patterns to identify:
- Functional requirements (what the system should do)
- Non-functional requirements (how well it should do it)
- Constraints (limitations and boundaries)

Example:
    >>> from claude_spec_benchmark.elicitation.extraction import RequirementsExtractor
    >>> extractor = RequirementsExtractor()
    >>> requirements = extractor.extract('''
    ...     Bug: Login fails with special characters in password
    ...     Users cannot log in when password contains @ or #
    ...     Expected: Login should accept all printable ASCII
    ... ''')
    >>> for req in requirements:
    ...     print(f"{req.id}: {req.text}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from claude_spec_benchmark.elicitation.scoring import extract_keywords
from claude_spec_benchmark.models import Requirement

# Patterns for identifying requirement-like sentences
REQUIREMENT_PATTERNS: list[re.Pattern[str]] = [
    # Should/must/shall statements
    re.compile(r"(?:should|must|shall|needs?\s+to)\s+\w+", re.IGNORECASE),
    # Expected behavior
    re.compile(r"expected\s*:?\s*.+", re.IGNORECASE),
    # Bug descriptions (implicit requirements)
    re.compile(r"(?:bug|issue|error|problem)\s*:?\s*.+", re.IGNORECASE),
    # When/then patterns
    re.compile(r"when\s+.+,?\s+(?:then\s+)?(?:it\s+)?(?:should|must|will)", re.IGNORECASE),
    # Acceptance criteria style
    re.compile(r"(?:given|as\s+a|so\s+that)\s+.+", re.IGNORECASE),
    # Can/cannot statements
    re.compile(r"(?:can(?:not|'t)?|unable\s+to|fails?\s+to)\s+\w+", re.IGNORECASE),
]

# Patterns for categorizing requirements
NON_FUNCTIONAL_KEYWORDS: frozenset[str] = frozenset({
    "performance", "speed", "fast", "slow", "latency", "throughput",
    "security", "secure", "encrypt", "authentication", "authorization",
    "scalable", "scale", "load", "concurrent",
    "reliable", "availability", "uptime", "downtime",
    "maintainable", "readable", "clean", "refactor",
    "usable", "intuitive", "user-friendly", "accessible",
})

CONSTRAINT_KEYWORDS: frozenset[str] = frozenset({
    "limit", "maximum", "minimum", "at most", "at least",
    "only", "never", "always", "must not", "shall not",
    "backward compatible", "compatibility", "legacy",
    "within", "before", "after", "deadline",
    "constraint", "restriction", "limitation",
})


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling code blocks and edge cases.

    Args:
        text: Input text to split.

    Returns:
        List of sentence strings.
    """
    # Remove code blocks to avoid splitting on periods in code
    code_block_pattern = re.compile(r"```[\s\S]*?```|`[^`]+`")
    placeholders: list[str] = []

    def replace_code(match: re.Match[str]) -> str:
        placeholders.append(match.group(0))
        return f"__CODE_BLOCK_{len(placeholders) - 1}__"

    cleaned = code_block_pattern.sub(replace_code, text)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)

    # Also split on newlines that look like separate items
    result: list[str] = []
    for sentence in sentences:
        # Split on double newlines or bullet points
        parts = re.split(r"\n\n+|\n\s*[-*•]\s*", sentence)
        result.extend(p.strip() for p in parts if p.strip())

    # Restore code blocks (we keep them for context)
    final: list[str] = []
    for sentence in result:
        for i, placeholder in enumerate(placeholders):
            sentence = sentence.replace(f"__CODE_BLOCK_{i}__", placeholder)
        if sentence.strip():
            final.append(sentence.strip())

    return final


def _categorize_requirement(
    text: str,
) -> Literal["functional", "non-functional", "constraint"]:
    """Categorize a requirement based on keywords.

    Args:
        text: Requirement text to categorize.

    Returns:
        Category string.
    """
    text_lower = text.lower()

    # Check for constraint indicators
    for keyword in CONSTRAINT_KEYWORDS:
        if keyword in text_lower:
            return "constraint"

    # Check for non-functional indicators
    for keyword in NON_FUNCTIONAL_KEYWORDS:
        if keyword in text_lower:
            return "non-functional"

    return "functional"


def _compute_text_span(full_text: str, sentence: str) -> tuple[int, int] | None:
    """Find the character span of a sentence in the full text.

    Args:
        full_text: Original full text.
        sentence: Sentence to find.

    Returns:
        Tuple of (start, end) indices or None if not found.
    """
    # Try exact match first
    idx = full_text.find(sentence)
    if idx >= 0:
        return (idx, idx + len(sentence))

    # Try with normalized whitespace
    normalized_sentence = " ".join(sentence.split())
    normalized_text = " ".join(full_text.split())
    idx = normalized_text.find(normalized_sentence)
    if idx >= 0:
        # Approximate position (won't be exact due to normalization)
        return (idx, idx + len(normalized_sentence))

    return None


@dataclass
class RequirementsExtractor:
    """Extracts atomic requirements from issue text.

    Parses SWE-bench issues into structured requirements that can be
    used by the ElicitationOracle for stakeholder simulation.

    Attributes:
        min_requirement_length: Minimum characters for a valid requirement.
        max_requirements: Maximum requirements to extract per issue.
        include_implicit: Whether to extract implicit requirements from bug descriptions.

    Example:
        >>> extractor = RequirementsExtractor()
        >>> requirements = extractor.extract(issue_text)
        >>> print(f"Found {len(requirements)} requirements")
    """

    min_requirement_length: int = 15
    max_requirements: int = 30
    include_implicit: bool = True
    _id_counter: int = field(default=0, init=False, repr=False)

    def extract(self, issue_text: str) -> list[Requirement]:
        """Extract requirements from issue text.

        Args:
            issue_text: Full issue text from SWE-bench.

        Returns:
            List of Requirement objects extracted from the text.

        Example:
            >>> requirements = extractor.extract('''
            ...     Users should be able to reset their password.
            ...     The reset link should expire after 24 hours.
            ... ''')
            >>> len(requirements)
            2
        """
        self._id_counter = 0
        requirements: list[Requirement] = []
        seen_texts: set[str] = set()  # Deduplicate

        # Split into sentences
        sentences = _split_into_sentences(issue_text)

        for sentence in sentences:
            # Skip if too short
            if len(sentence) < self.min_requirement_length:
                continue

            # Check if sentence matches requirement patterns
            is_requirement = False
            for pattern in REQUIREMENT_PATTERNS:
                if pattern.search(sentence):
                    is_requirement = True
                    break

            if not is_requirement:
                continue

            # Normalize for deduplication
            normalized = " ".join(sentence.lower().split())
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            # Create requirement
            req = self._create_requirement(sentence, issue_text)
            requirements.append(req)

            # Check limit
            if len(requirements) >= self.max_requirements:
                break

        return requirements

    def _create_requirement(self, sentence: str, full_text: str) -> Requirement:
        """Create a Requirement object from a sentence.

        Args:
            sentence: The requirement sentence.
            full_text: Full issue text for span calculation.

        Returns:
            Populated Requirement object.
        """
        self._id_counter += 1
        req_id = f"REQ-{self._id_counter:03d}"

        # Categorize
        category = _categorize_requirement(sentence)

        # Extract keywords
        keywords = extract_keywords(sentence)

        # Find source span
        span = _compute_text_span(full_text, sentence)

        return Requirement(
            id=req_id,
            text=sentence,
            category=category,
            keywords=keywords,
            discoverable=True,
            source_span=span,
        )

    def extract_with_context(
        self, issue_text: str, title: str | None = None
    ) -> list[Requirement]:
        """Extract requirements with additional context from title.

        Args:
            issue_text: Full issue text.
            title: Optional issue title for additional context.

        Returns:
            List of Requirement objects.
        """
        # Combine title and text if provided
        combined = f"{title}\n\n{issue_text}" if title else issue_text
        return self.extract(combined)

    def get_summary(self, requirements: list[Requirement]) -> dict[str, int]:
        """Get a summary of extracted requirements by category.

        Args:
            requirements: List of extracted requirements.

        Returns:
            Dict mapping category to count.

        Example:
            >>> summary = extractor.get_summary(requirements)
            >>> print(summary)
            {'functional': 5, 'non-functional': 2, 'constraint': 1}
        """
        summary: dict[str, int] = {
            "functional": 0,
            "non-functional": 0,
            "constraint": 0,
        }
        for req in requirements:
            summary[req.category] += 1
        return summary
